"""
Dataset service — handles file upload, validation, and storage in PostgreSQL.
"""

import logging
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import sync_engine
from app.db.models import Dataset

logger = logging.getLogger(__name__)

# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/octet-stream",  # Some systems send this for xlsx
}


def _generate_table_name() -> str:
    """Generate a safe, unique PostgreSQL table name for the dataset."""
    short_id = uuid.uuid4().hex[:10]
    return f"dataset_{short_id}"


def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names: lowercase, replace spaces/special chars with underscores."""
    new_columns = {}
    for col in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip())
        clean = re.sub(r"_+", "_", clean).strip("_").lower()
        if not clean:
            clean = f"col_{df.columns.get_loc(col)}"
        new_columns[col] = clean
    df = df.rename(columns=new_columns)
    return df


def _extract_schema_info(df: pd.DataFrame) -> dict[str, Any]:
    """Extract column names, types, and sample values for the schema."""
    schema = {"columns": []}
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
            "unique_count": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
        }
        schema["columns"].append(col_info)
    return schema


def _read_csv_with_recovery(content: bytes) -> tuple[pd.DataFrame, str]:
    """Read common CSV encodings instead of assuming UTF-8 only."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding, on_bad_lines="skip"), encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError("The CSV could not be decoded with a supported text encoding.") from last_error


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply conservative quality fixes while preserving usable source records."""
    original_rows = len(df)
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(column).strip()) for column in df.columns]

    for column in df.select_dtypes(include=["object", "string"]).columns:
        cleaned = df[column].astype("string").str.strip()
        df[column] = cleaned.replace({"": pd.NA, "N/A": pd.NA, "NA": pd.NA, "null": pd.NA, "None": pd.NA})

    # Source exports often include blank separator rows and a publisher footer.
    df = df.dropna(how="all")
    if len(df.columns) > 1:
        first_column = df.columns[0]
        footer_mask = (
            df[first_column].astype("string").str.contains("development bank group|data portal", case=False, na=False)
            & df.drop(columns=[first_column]).isna().all(axis=1)
        )
        df = df.loc[~footer_mask]

    converted_columns: list[str] = []
    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        normalized = values.str.replace(r"[,$]", "", regex=True).str.replace(" ", "", regex=False)
        numeric = pd.to_numeric(normalized, errors="coerce")
        if numeric.notna().mean() >= 0.95 and ("amount" in column.lower() or "commitment" in column.lower() or "value" in column.lower()):
            df[column] = pd.to_numeric(df[column].astype("string").str.replace(r"[,$]", "", regex=True), errors="coerce")
            converted_columns.append(column)
        elif "date" in column.lower():
            parsed_dates = pd.to_datetime(values, errors="coerce", dayfirst=False)
            if parsed_dates.notna().mean() >= 0.8:
                df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d")
                converted_columns.append(column)

    before_duplicates = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {
        "original_rows": original_rows,
        "clean_rows": len(df),
        "removed_blank_or_footer_rows": original_rows - before_duplicates,
        "removed_duplicate_rows": before_duplicates - len(df),
        "converted_columns": converted_columns,
    }


async def process_upload(
    file: UploadFile,
    user_id: uuid.UUID,
    dataset_name: str,
    db: AsyncSession,
) -> Dataset:
    """
    Validate, read, and store an uploaded dataset file.

    Steps:
    1. Validate file extension and size
    2. Read into Pandas DataFrame
    3. Sanitize column names
    4. Create a PostgreSQL table and insert data
    5. Create Dataset metadata record

    Returns the created Dataset ORM object.
    """
    # --- Validate extension ---
    file_name = file.filename or "unnamed"
    extension = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{extension}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # --- Read file content ---
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    # --- Parse and clean into a DataFrame ---
    try:
        if extension == ".csv":
            df, detected_encoding = _read_csv_with_recovery(content)
        else:  # .xlsx
            df = pd.read_excel(BytesIO(content))
            detected_encoding = None
    except Exception as e:
        logger.error(f"Failed to parse uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to process this file. Ensure it contains valid tabular data.",
        )

    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file contains no readable tabular data.",
        )

    df, cleaning_summary = _clean_dataframe(df)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file contains no usable tabular records after data-quality cleaning.",
        )

    # --- Sanitize columns ---
    df = _sanitize_column_names(df)

    # --- Generate safe table name ---
    table_name = _generate_table_name()

    # --- Extract schema ---
    schema_info = _extract_schema_info(df)
    cleaned_file_name = f"cleaned_{uuid.uuid4().hex}.csv"
    settings.ensure_directories()
    df.to_csv(Path(settings.UPLOAD_DIR) / cleaned_file_name, index=False, encoding="utf-8")
    schema_info["cleaning"] = {
        **cleaning_summary,
        "source_encoding": detected_encoding,
        "cleaned_file_name": cleaned_file_name,
    }

    # --- Store data in PostgreSQL ---
    try:
        df.to_sql(
            name=table_name,
            con=sync_engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=500,
        )
        logger.info(f"Dataset stored in table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")
    except Exception as e:
        logger.error(f"Failed to store dataset in PostgreSQL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store dataset in database.",
        )

    # --- Create Dataset metadata ---
    dataset = Dataset(
        user_id=user_id,
        name=dataset_name or file_name.rsplit(".", 1)[0],
        file_name=file_name,
        file_type=extension.lstrip("."),
        table_name=table_name,
        row_count=len(df),
        column_count=len(df.columns),
        schema_info=schema_info,
    )
    db.add(dataset)
    await db.commit()
    logger.info(f"Dataset '{dataset.name}' created with ID {dataset.id}")

    # --- Automatically trigger Data Analyst Agent Profiling ---
    try:
        from app.agents.data_analyst.agent import data_analyst_agent
        from app.core.vector_store import vector_store_service
        from app.db.models import DatasetProfile

        profile_json = data_analyst_agent.profile_dataset(table_name)
        profile_record = DatasetProfile(
            dataset_id=dataset.id,
            profile_json=profile_json,
        )
        db.add(profile_record)
        await db.commit()
        vector_store_service.index_dataset_schema(
            dataset_id=str(dataset.id),
            dataset_name=dataset.name,
            schema_info=schema_info,
        )
        logger.info(f"Dataset profile created and stored for dataset ID {dataset.id}")
    except Exception as e:
        logger.error(f"Failed to generate profile for dataset '{dataset.id}': {e}")
        # Profiling failure should not abort upload, but log error

    return dataset



async def delete_dataset_table(table_name: str) -> None:
    """Drop the data table associated with a dataset (using sync engine)."""
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "", table_name)
    try:
        with sync_engine.connect() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{safe_name}"'))
            conn.commit()
        logger.info(f"Dropped table '{safe_name}'")
    except Exception as e:
        logger.error(f"Failed to drop table '{safe_name}': {e}")
