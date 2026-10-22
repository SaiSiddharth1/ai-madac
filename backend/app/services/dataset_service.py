"""
Dataset service — handles file upload, validation, and storage in PostgreSQL.
"""

import json
import logging
import re
import time
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
    """Generate a safe, unique PostgreSQL/SQLite table name for the dataset."""
    short_id = uuid.uuid4().hex[:10]
    return f"dataset_{short_id}"


def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names: replace spaces/special chars with underscores, ensure uniqueness, and preserve casing."""
    sanitized = []
    seen = {}
    for i, col in enumerate(df.columns):
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip())
        clean = re.sub(r"_+", "_", clean).strip("_")
        if not clean:
            clean = f"col_{i + 1}"

        lower_clean = clean.lower()
        if lower_clean in seen:
            seen[lower_clean] += 1
            clean = f"{clean}_{seen[lower_clean]}"
        else:
            seen[lower_clean] = 0

        sanitized.append(clean)

    df.columns = sanitized
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


from app.services.dataset_loader import load_dataset


def _read_csv_with_recovery(content: bytes) -> tuple[pd.DataFrame, str]:
    """Compatibility wrapper around central dataset loader."""
    df, meta = load_dataset(content=content, filename="dataset.csv")
    return df, meta.get("detected_encoding") or "utf-8"


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply conservative quality fixes while preserving usable source records."""
    original_rows = len(df)
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(column).strip()) for column in df.columns]

    for column in df.select_dtypes(include=["object", "string"]).columns:
        cleaned = df[column].astype("string").str.strip()
        df[column] = cleaned.replace({
            "": pd.NA,
            "N/A": pd.NA,
            "NA": pd.NA,
            "null": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "nan": pd.NA,
        })

    # Remove completely blank rows
    df = df.dropna(how="all")
    if len(df.columns) > 1:
        first_column = df.columns[0]
        footer_mask = (
            df[first_column].astype("string").str.contains("development bank group|data portal", case=False, na=False)
            & df.drop(columns=[first_column]).isna().all(axis=1)
        )
        df = df.loc[~footer_mask]

    before_duplicates = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {
        "original_rows": original_rows,
        "clean_rows": len(df),
        "removed_blank_or_footer_rows": original_rows - before_duplicates,
        "removed_duplicate_rows": before_duplicates - len(df),
        "converted_columns": [],
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
    2. Read into Pandas DataFrame using central dataset loader
    3. Sanitize column names
    4. Create a database table and insert data
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

    # --- Parse and clean into a DataFrame using central dataset_loader ---
    try:
        df, loader_metadata = load_dataset(content=content, filename=file_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to process this file: {e}",
        )

    detected_encoding = loader_metadata.get("detected_encoding")
    detected_delimiter = loader_metadata.get("detected_delimiter")

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
        "detected_delimiter": detected_delimiter,
        "cleaned_file_name": cleaned_file_name,
    }

    # --- Store data in database ---
    # Ensure nested objects (e.g. lists, dicts) are stringified to prevent SQL serialization errors
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(20)
            if any(isinstance(v, (list, dict, set, tuple)) for v in sample):
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict, set, tuple)) else (str(x) if pd.notna(x) else None)
                )

    is_sqlite = sync_engine.dialect.name == "sqlite"
    num_cols = max(len(df.columns), 1)

    # In SQLite, parameter limit is 999 (old) or 32766 (new).
    # method="multi" creates (chunksize * num_cols) parameters per statement.
    # To avoid "sqlite3.OperationalError: too many SQL variables":
    # For SQLite, use method=None (native executemany, robust for any number of columns and rows).
    # For PostgreSQL, use method="multi" with chunksize=2000.
    if is_sqlite:
        insert_method = None
        chunk_size = 1000
    else:
        insert_method = "multi"
        chunk_size = 2000

    max_retries = 3
    last_err = None
    for attempt in range(max_retries):
        try:
            df.to_sql(
                name=table_name,
                con=sync_engine,
                if_exists="replace",
                index=False,
                method=insert_method,
                chunksize=chunk_size,
            )
            logger.info(f"Dataset stored in table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")
            last_err = None
            break
        except Exception as e:
            last_err = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to store dataset in database failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                # Fallback: try smaller chunk size without method
                try:
                    df.to_sql(
                        name=table_name,
                        con=sync_engine,
                        if_exists="replace",
                        index=False,
                        method=None,
                        chunksize=250,
                    )
                    logger.info(f"Fallback insert succeeded for table '{table_name}'")
                    last_err = None
                    break
                except Exception as final_e:
                    last_err = final_e

    if last_err is not None:
        logger.error(f"Failed to store dataset in database: {last_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store dataset in database: {str(last_err)}",
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
