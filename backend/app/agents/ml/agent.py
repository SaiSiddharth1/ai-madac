"""
ML Agent.
Automated Machine Learning agent for dataset model training, evaluation, and predictions.
"""

import json
import logging
import os
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.agents.state import AgentState
from app.config import settings
from app.db.database import sync_engine
from app.utils.sql_validator import sanitize_table_name

logger = logging.getLogger(__name__)


class MLAgent:
    """
    Automated ML Agent handling feature engineering, model selection,
    training, metric evaluation, model persistence, and inference.
    """

    def train_model(
        self,
        table_name: str,
        target_column: str,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Train a Machine Learning model on a dataset table.

        Returns metadata dict containing:
        - model_id, model_type, task_type
        - metrics (accuracy/f1 or MAE/R2)
        - feature_importances
        - model_path
        """
        safe_table = sanitize_table_name(table_name)
        logger.info(f"ML Agent training model on table '{safe_table}', target: '{target_column}'")

        try:
            df = pd.read_sql(f'SELECT * FROM "{safe_table}"', con=sync_engine)
        except Exception as e:
            raise RuntimeError(f"Database error during ML training: {e}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Drop rows where target is NaN
        df = df.dropna(subset=[target_column]).copy()

        # Select features
        if not feature_columns:
            feature_columns = [c for c in df.columns if c != target_column]

        X_df = df[feature_columns].copy()
        y_raw = df[target_column].copy()

        # Preprocessing: encode categorical features & target
        encoders = {}
        for col in X_df.columns:
            if X_df[col].dtype == "object" or X_df[col].dtype.name == "category":
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                encoders[col] = le
            else:
                # Impute numeric NaNs with median
                X_df[col] = X_df[col].fillna(X_df[col].median())

        # Determine task type (classification vs regression)
        unique_targets = y_raw.nunique()
        if y_raw.dtype == "object" or y_raw.dtype.name == "category" or unique_targets <= 15:
            task_type = "classification"
            target_le = LabelEncoder()
            y = target_le.fit_transform(y_raw.astype(str))
            encoders["__target__"] = target_le
        else:
            task_type = "regression"
            y = y_raw.astype(float).values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )

        # Instantiate model
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
            }
            model_type = "RandomForestClassifier"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            metrics = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2_score": round(r2, 4),
            }
            model_type = "RandomForestRegressor"

        # Feature importances
        importances = model.feature_importances_
        feature_importances = {
            col: round(float(imp), 4)
            for col, imp in zip(feature_columns, importances)
        }
        # Sort importances descending
        feature_importances = dict(
            sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        )

        # Save model + encoders bundle
        settings.ensure_directories()
        model_filename = f"model_{uuid.uuid4().hex[:10]}.joblib"
        model_filepath = os.path.join(settings.MODELS_DIR, model_filename)

        artifact = {
            "model": model,
            "encoders": encoders,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "task_type": task_type,
            "metrics": metrics,
        }
        joblib.dump(artifact, model_filepath)
        logger.info(f"Model saved to '{model_filepath}'")

        return {
            "model_type": model_type,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "feature_importances": feature_importances,
            "model_path": model_filepath,
        }

    def predict(self, model_path: str, input_features: dict[str, Any]) -> dict[str, Any]:
        """
        Execute prediction on user input features using a saved model artifact.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found at path '{model_path}'")

        artifact = joblib.load(model_path)
        model = artifact["model"]
        encoders = artifact["encoders"]
        feature_columns = artifact["feature_columns"]
        task_type = artifact["task_type"]

        # Build feature vector
        row = []
        for col in feature_columns:
            val = input_features.get(col)
            if col in encoders:
                try:
                    val_encoded = encoders[col].transform([str(val)])[0]
                except ValueError:
                    # Unseen category fallback
                    val_encoded = 0
                row.append(val_encoded)
            else:
                row.append(float(val) if val is not None else 0.0)

        X_input = pd.DataFrame([row], columns=feature_columns)
        raw_pred = model.predict(X_input)[0]

        if task_type == "classification" and "__target__" in encoders:
            prediction_label = str(encoders["__target__"].inverse_transform([int(raw_pred)])[0])
            probas = model.predict_proba(X_input)[0].tolist()
            confidence = round(float(max(probas)), 4)
        else:
            prediction_label = round(float(raw_pred), 3)
            confidence = None

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "task_type": task_type,
        }

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Agent node handler for graph execution.
        """
        question = state.get("question", "")
        table_name = state.get("table_name", "")
        schema = state.get("schema", {})

        # Default prediction target heuristic if not explicitly set
        columns = [c.get("name") for c in schema.get("columns", [])]
        target_col = columns[-1] if columns else "target"

        try:
            res = self.train_model(table_name, target_col)
            agents = state.get("agents_used", []) + ["ml_agent"]
            return {
                "ml_result": res,
                "agents_used": agents,
            }
        except Exception as e:
            logger.error(f"ML Agent execution error: {e}")
            return {"ml_result": {"error": str(e)}}


# Singleton instance
ml_agent = MLAgent()
