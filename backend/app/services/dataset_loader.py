"""
Central Dataset Loader for AI-MADAC.
Provides universal dataset ingestion, automatic delimiter and encoding detection,
structural validation, data type inference, and normalization for CSV and Excel files.
"""

import csv
import io
import logging
import re
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Supported delimiters to evaluate
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]

# Encodings to try in priority order
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "iso-8859-1"]


def detect_csv_encoding_and_delimiter(
    content: bytes,
) -> Tuple[str, str, dict[str, Any]]:
    """
    Automatically detect the text encoding and the most structurally consistent delimiter.
    
    Evaluates candidate delimiters based on:
    - Number of columns produced in the header
    - Consistency of field count across sample data rows
    - Header vs row field count match
    - Parser validation
    """
    sample_bytes = content[:100000]
    decoded_text: str | None = None
    detected_encoding: str | None = None

    for enc in CANDIDATE_ENCODINGS:
        try:
            decoded_text = sample_bytes.decode(enc)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None or detected_encoding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The CSV could not be decoded with a supported text encoding (UTF-8, CP1252, ISO-8859-1).",
        )

    # Take representative non-empty sample lines (up to 50 lines)
    sample_lines = [line.strip() for line in decoded_text.splitlines() if line.strip()][:50]
    if not sample_lines:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty.",
        )

    best_delimiter = ","
    best_score = -1.0
    delimiter_evaluations: dict[str, Any] = {}

    for sep in CANDIDATE_DELIMITERS:
        try:
            parsed_rows = [row for row in csv.reader(sample_lines, delimiter=sep) if row]
            if not parsed_rows:
                continue

            header_fields = len(parsed_rows[0])
            data_rows = parsed_rows[1:]
            
            if header_fields < 2:
                # Delimiter did not split header into multiple fields
                score = 0.0
                consistency = 0.0
            else:
                row_lengths = [len(r) for r in data_rows]
                if row_lengths:
                    exact_matches = sum(1 for length in row_lengths if length == header_fields)
                    consistency = exact_matches / len(row_lengths)
                else:
                    consistency = 1.0

                # Score combines column count and row consistency
                score = (header_fields * 100.0) * (consistency ** 2)

            delimiter_evaluations[sep] = {
                "header_fields": header_fields,
                "consistency": consistency,
                "score": score,
            }

            if score > best_score:
                best_score = score
                best_delimiter = sep

        except Exception as e:
            logger.debug(f"Failed delimiter evaluation for '{sep}': {e}")
            continue

    # Structural Validation:
    # If the winning delimiter produced only 1 column, but the header contains candidate delimiter chars,
    # investigate whether an alternative candidate delimiter was overlooked.
    first_line = sample_lines[0]
    if best_score <= 0.0:
        found_candidates = [sep for sep in CANDIDATE_DELIMITERS if sep in first_line]
        if found_candidates:
            for sep in found_candidates:
                try:
                    test_df = pd.read_csv(
                        io.BytesIO(sample_bytes),
                        sep=sep,
                        encoding=detected_encoding,
                        nrows=10,
                        engine="python",
                    )
                    if len(test_df.columns) > 1:
                        best_delimiter = sep
                        best_score = len(test_df.columns) * 100.0
                        break
                except Exception:
                    continue

        if best_score <= 0.0 and len(found_candidates) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to reliably determine the CSV delimiter. Please verify that the file is a valid CSV.",
            )

    logger.info(
        f"Detected CSV encoding '{detected_encoding}', delimiter '{best_delimiter}' "
        f"(score: {best_score}, evals: {delimiter_evaluations})"
    )

    return detected_encoding, best_delimiter, delimiter_evaluations


def infer_and_convert_column_types(df: pd.DataFrame, detected_delimiter: str = ",") -> Tuple[pd.DataFrame, list[str]]:
    """
    Infer and convert column data types on the current DataFrame.
    Supports integer, float (including European comma-decimals), datetime, boolean, and string.
    """
    converted_columns: list[str] = []

    for col in df.columns:
        # Only process object / string columns for type inference
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            continue

        non_null_series = df[col].dropna()
        if non_null_series.empty:
            continue

        str_series = non_null_series.astype(str).str.strip()

        # 1. European decimal detection (e.g. "2,55" or "-123,45")
        # Prevalent when delimiter is ';' or '\t'
        euro_float_pattern = r"^-?\d+,\d+$"
        euro_matches = str_series.str.match(euro_float_pattern).mean()
        
        if euro_matches >= 0.8:
            converted = pd.to_numeric(
                df[col].astype("string").str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if converted.notna().mean() >= 0.8:
                df[col] = converted
                converted_columns.append(col)
                continue

        # If values contain letters (e.g. cancellation 'C' in InvoiceNo or alphanumeric StockCodes),
        # do not treat as numeric measures.
        has_letters = str_series.str.contains(r"[a-zA-Z]").any()
        if has_letters:
            # Check for datetime with month names (e.g. "01-Dec-2010")
            if any(term in col.lower() for term in ("date", "time", "year", "month", "day")):
                try:
                    sample_dates = pd.to_datetime(str_series.head(100), errors="coerce", format="mixed", dayfirst=True)
                    if sample_dates.notna().mean() >= 0.8:
                        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=True)
                        converted_columns.append(col)
                        continue
                except Exception:
                    pass
            continue

        # 2. Standard numeric detection (integers or floats with dot decimal)
        cleaned_numeric_str = (
            str_series.str.replace(r"[$€£¥]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        if cleaned_numeric_str.str.contains(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$").mean() >= 0.8:
            cleaned_numeric_str = cleaned_numeric_str.str.replace(",", "", regex=False)

        numeric_converted = pd.to_numeric(cleaned_numeric_str, errors="coerce")
        if numeric_converted.notna().mean() >= 0.95:
            valid_nums = numeric_converted.dropna()
            if not valid_nums.empty and (valid_nums % 1 == 0).all():
                if numeric_converted.isna().sum() == 0:
                    df[col] = numeric_converted.astype("int64")
                else:
                    df[col] = numeric_converted
            else:
                df[col] = numeric_converted
            converted_columns.append(col)
            continue

        # 3. Datetime detection
        has_date_delimiters = str_series.str.contains(r"[\./\-:]").mean() >= 0.8
        if has_date_delimiters:
            try:
                # Use format="mixed" for fast, robust multi-format date parsing
                sample_dates = pd.to_datetime(str_series.head(100), errors="coerce", format="mixed", dayfirst=True)
                if sample_dates.notna().mean() >= 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=True)
                    converted_columns.append(col)
                    continue
            except Exception:
                pass

        # 4. Boolean detection
        lower_vals = str_series.str.lower()
        bool_map = {"true": True, "false": False, "yes": True, "no": False, "t": True, "f": False}
        if lower_vals.isin(bool_map.keys()).mean() >= 0.95:
            df[col] = lower_vals.map(bool_map)
            converted_columns.append(col)
            continue

    return df, converted_columns


def load_dataset(
    content: bytes | None = None,
    file_path: str | Path | None = None,
    filename: str | None = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Central dataset loading function.
    
    Accepts raw bytes or a file path, detects delimiters/encoding, validates structure,
    and returns a clean, typed Pandas DataFrame along with ingestion metadata.
    """
    if content is None and file_path is not None:
        p = Path(file_path)
        filename = filename or p.name
        with open(p, "rb") as f:
            content = f.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file content is empty.",
        )

    filename = filename or "dataset"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to parse Excel file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to parse Excel file: {e}",
            )
        detected_encoding = None
        detected_delimiter = None
        evaluations = {}
    elif extension == ".csv" or not extension:
        detected_encoding, detected_delimiter, evaluations = detect_csv_encoding_and_delimiter(content)
        
        # Read the CSV with detected parameters
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                sep=detected_delimiter,
                encoding=detected_encoding,
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.warning(f"C-engine parse failed with {e}; retrying with python engine...")
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    sep=detected_delimiter,
                    encoding=detected_encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e2:
                logger.error(f"Failed to parse CSV file: {e2}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unable to process CSV file with detected delimiter '{detected_delimiter}': {e2}",
                )

        # Structural Validation:
        # If the file produces 1 column, but the header contains candidate delimiter characters,
        # it is an invalid parse. Retry using candidate delimiters.
        if len(df.columns) == 1:
            col_name = str(df.columns[0])
            for sep in CANDIDATE_DELIMITERS:
                if sep in col_name and sep != detected_delimiter:
                    logger.info(f"Retrying single-column parse with delimiter '{sep}'...")
                    try:
                        retry_df = pd.read_csv(
                            io.BytesIO(content),
                            sep=sep,
                            encoding=detected_encoding,
                            on_bad_lines="skip",
                        )
                        if len(retry_df.columns) > 1:
                            df = retry_df
                            detected_delimiter = sep
                            break
                    except Exception:
                        continue

            if len(df.columns) == 1 and any(s in col_name for s in CANDIDATE_DELIMITERS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Detected 1 column from a file whose header appears to contain multiple fields. Unable to parse table structure.",
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{extension}'. Allowed: .csv, .xlsx, .xls",
        )

    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file contains no readable tabular data.",
        )

    # Universal normalization: clean column whitespace and missing values
    df.columns = [re.sub(r"\s+", " ", str(col).strip()) for col in df.columns]

    for col in df.select_dtypes(include=["object", "string"]).columns:
        cleaned = df[col].astype("string").str.strip()
        df[col] = cleaned.replace({
            "": pd.NA,
            "N/A": pd.NA,
            "NA": pd.NA,
            "null": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "nan": pd.NA,
        })

    # Drop completely blank rows
    original_rows = len(df)
    df = df.dropna(how="all")
    blank_rows_dropped = original_rows - len(df)

    # Infer and convert column data types
    df, converted_cols = infer_and_convert_column_types(df, detected_delimiter=detected_delimiter or ",")

    metadata = {
        "file_type": extension.lstrip("."),
        "detected_encoding": detected_encoding,
        "detected_delimiter": detected_delimiter,
        "original_rows": original_rows,
        "blank_rows_dropped": blank_rows_dropped,
        "converted_columns": converted_cols,
        "evaluations": evaluations,
    }

    return df, metadata
