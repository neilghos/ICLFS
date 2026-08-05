from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.metrics as skm


CLASSIFICATION = "classification"
REGRESSION = "regression"
RANKING = "ranking"


def calculate_metrics(
    task: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)

    if task == CLASSIFICATION:
        labels = prediction.astype(np.int64).reshape(-1)
        accuracy = float(skm.accuracy_score(y_true, labels))
        return {
            "accuracy": accuracy,
            "score": accuracy,
        }

    if task in {REGRESSION, RANKING}:
        rmse = float(np.sqrt(skm.mean_squared_error(y_true, prediction.reshape(-1))))
        mae = float(skm.mean_absolute_error(y_true, prediction.reshape(-1)))
        r2 = float(skm.r2_score(y_true, prediction.reshape(-1)))
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "score": -rmse,
        }

    raise ValueError(f"Unknown task '{task}'.")


def summarize_metrics(task: str, metrics: dict[str, Any]) -> str:
    if task == CLASSIFICATION:
        return (
            f"score={metrics['score']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}"
        )
    if task in {REGRESSION, RANKING}:
        return (
            f"score={metrics['score']:.4f} "
            f"rmse={metrics['rmse']:.4f} "
            f"mae={metrics['mae']:.4f} "
            f"r2={metrics['r2']:.4f}"
        )
    raise ValueError(f"Unknown task '{task}'.")
