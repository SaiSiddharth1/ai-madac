"""
ML API Endpoints.
Train machine learning models, list trained models, and execute model predictions.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.agents.ml.agent import ml_agent
from app.db.models import User, Dataset, MLModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["Machine Learning"])


class TrainModelRequest(BaseModel):
    target_column: str
    feature_columns: Optional[list[str]] = None


class PredictRequest(BaseModel):
    model_id: UUID
    input_features: dict[str, Any]


@router.post("/{dataset_id}/train", status_code=status.HTTP_201_CREATED)
async def train_dataset_model(
    dataset_id: UUID,
    body: TrainModelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Train a machine learning model on a dataset for a target column.
    """
    # Verify dataset ownership
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
    )
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    try:
        res = ml_agent.train_model(
            table_name=dataset.table_name,
            target_column=body.target_column,
            feature_columns=body.feature_columns,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )

    # Save metadata to ml_models DB table
    model_record = MLModel(
        dataset_id=dataset_id,
        model_type=res["model_type"],
        target_column=res["target_column"],
        feature_columns=res["feature_columns"],
        metrics=res["metrics"],
        model_path=res["model_path"],
    )
    db.add(model_record)
    await db.commit()
    await db.refresh(model_record)

    logger.info(f"Trained model '{model_record.id}' for dataset {dataset_id}")

    return {
        "message": "Model trained and saved successfully",
        "model_id": model_record.id,
        "metrics": res["metrics"],
        "feature_importances": res["feature_importances"],
        "task_type": res["task_type"],
    }


@router.get("/{dataset_id}/models")
async def list_dataset_models(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all trained ML models for a dataset.
    """
    # Verify ownership
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
    )
    if not ds_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    result = await db.execute(
        select(MLModel).where(MLModel.dataset_id == dataset_id).order_by(MLModel.created_at.desc())
    )
    models = result.scalars().all()

    return [
        {
            "id": m.id,
            "model_type": m.model_type,
            "target_column": m.target_column,
            "feature_columns": m.feature_columns,
            "metrics": m.metrics,
            "created_at": m.created_at,
        }
        for m in models
    ]


@router.post("/{dataset_id}/predict")
async def predict_with_model(
    dataset_id: UUID,
    body: PredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Make a prediction using a trained model.
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == body.model_id, MLModel.dataset_id == dataset_id)
    )
    model_record = result.scalar_one_or_none()

    if not model_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    try:
        prediction_result = ml_agent.predict(
            model_path=model_record.model_path,
            input_features=body.input_features,
        )
        return {
            "model_id": model_record.id,
            "target_column": model_record.target_column,
            **prediction_result,
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )
