"""
Dataset API endpoints — upload, list, detail, delete.
All endpoints require authentication and enforce user-scoped access.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.config import settings
from app.db.models import User, Dataset, DatasetProfile
from app.schemas.dataset import (
    DatasetResponse,
    DatasetListResponse,
    DatasetProfileResponse,
    DatasetUploadResponse,
)
from app.services.dataset_service import process_upload, delete_dataset_table
from app.core.vector_store import vector_store_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a CSV or XLSX dataset file."""
    dataset = await process_upload(
        file=file,
        user_id=current_user.id,
        dataset_name=name,
        db=db,
    )
    return DatasetUploadResponse(
        message="Dataset uploaded and stored successfully",
        dataset=DatasetResponse.model_validate(dataset),
    )


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all datasets belonging to the current user."""
    result = await db.execute(
        select(Dataset)
        .where(Dataset.user_id == current_user.id)
        .order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()

    return DatasetListResponse(
        datasets=[DatasetResponse.model_validate(d) for d in datasets],
        total=len(datasets),
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific dataset's metadata. Enforces user ownership."""
    dataset = await _get_user_dataset(dataset_id, current_user.id, db)
    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a dataset, its data table, and all related records."""
    dataset = await _get_user_dataset(dataset_id, current_user.id, db)

    # Drop the actual data table
    await delete_dataset_table(dataset.table_name)
    vector_store_service.delete_dataset_schema(str(dataset.id))

    # Delete the metadata record (cascades to profile, queries, etc.)
    await db.delete(dataset)
    await db.commit()

    logger.info(f"Dataset '{dataset.name}' (ID: {dataset.id}) deleted by user {current_user.id}")


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def get_dataset_profile(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the dataset profile (auto-generated on upload)."""
    dataset = await _get_user_dataset(dataset_id, current_user.id, db)

    result = await db.execute(
        select(DatasetProfile).where(DatasetProfile.dataset_id == dataset.id)
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset profile not yet generated. Please wait or re-upload.",
        )

    return DatasetProfileResponse.model_validate(profile)


@router.get("/{dataset_id}/cleaned-file")
async def download_cleaned_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the normalized UTF-8 CSV used by the analysis pipeline."""
    dataset = await _get_user_dataset(dataset_id, current_user.id, db)
    cleaning = (dataset.schema_info or {}).get("cleaning", {})
    cleaned_file_name = cleaning.get("cleaned_file_name")
    if not cleaned_file_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No cleaned file is available for this dataset")

    path = Path(settings.UPLOAD_DIR) / Path(cleaned_file_name).name
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cleaned file is no longer available")
    return FileResponse(path, media_type="text/csv", filename=f"cleaned_{dataset.file_name.rsplit('.', 1)[0]}.csv")


# ---- Helper ----

async def _get_user_dataset(dataset_id: UUID, user_id: UUID, db: AsyncSession) -> Dataset:
    """Fetch a dataset and verify it belongs to the user. Raises 404 if not found."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    return dataset
