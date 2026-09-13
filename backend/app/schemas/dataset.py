"""
Pydantic schemas for dataset endpoints.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class DatasetResponse(BaseModel):
    """Dataset metadata returned to client."""
    id: UUID
    name: str
    file_name: str
    file_type: str
    table_name: str
    row_count: int
    column_count: int
    schema_info: Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    """List of datasets."""
    datasets: list[DatasetResponse]
    total: int


class DatasetProfileResponse(BaseModel):
    """Dataset profile returned to client."""
    id: UUID
    dataset_id: UUID
    profile_json: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    """Response after successful dataset upload."""
    message: str
    dataset: DatasetResponse
