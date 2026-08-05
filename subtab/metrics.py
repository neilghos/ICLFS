from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.special
import sklearn.metrics

from .util import PredictionType, TaskType


def _normalize_task_type(task: str | TaskType, y_true: np.ndarray | None = None) -> TaskType:
    if isinstance(task, TaskType):
        return task
    if task in {"binclass", "multiclass", "regression"}:
        return TaskType(task)
    if task == "classification":
        if y_true is not None and np.unique(np.asarray(y_true)).size <= 2:
            return TaskType.BINCLASS
        return TaskType.MULTICLASS
    if task == "ranking":
        return TaskType.REGRESSION
    raise ValueError(f"Unknown task type: {task}")


def _get_labels_and_probs(
    prediction: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType,
) -> tuple[np.ndarray, None | np.ndarray]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type == PredictionType.LABELS:
        return np.asarray(prediction), None
    if prediction_type == PredictionType.PROBS:
        probs = np.asarray(prediction)
    elif prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(prediction)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(prediction, axis=1)
        )
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype(np.int64), probs


def _tabm_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType,
) -> dict[str, Any]:
    task_type = _normalize_task_type(task_type, y_true)
    prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type == PredictionType.LABELS
        return {
            'rmse': float(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5),
            'mae': float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
            'r2': float(sklearn.metrics.r2_score(y_true, y_pred)),
        }

    labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
    result = cast(
        dict[str, Any],
        sklearn.metrics.classification_report(
            y_true,
            labels,
            output_dict=True,
            zero_division=0,
        ),
    )
    if probs is not None:
        result['cross-entropy'] = float(sklearn.metrics.log_loss(y_true, probs))
    if task_type == TaskType.BINCLASS and probs is not None:
        binary_probs = probs if probs.ndim == 1 else probs[:, 1]
        result['roc-auc'] = float(sklearn.metrics.roc_auc_score(y_true, binary_probs))
    return result


def calculate_metrics(task: str | TaskType, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    task_type = _normalize_task_type(task, y_true)
    if task_type == TaskType.REGRESSION:
        metrics = _tabm_metrics(y_true, y_pred, task_type, PredictionType.LABELS)
        metrics['score'] = -float(metrics['rmse'])
        return metrics

    metrics = _tabm_metrics(y_true, y_pred, task_type, PredictionType.LABELS)
    metrics['accuracy'] = float(metrics['accuracy'])
    metrics['macro_f1'] = float(metrics['macro avg']['f1-score'])
    metrics['score'] = float(metrics['accuracy'])
    return metrics


def summarize_metrics(task: str | TaskType, metrics: dict[str, Any]) -> str:
    task_type = _normalize_task_type(task)
    if task_type == TaskType.REGRESSION:
        return (
            f"score={metrics['score']:.4f} rmse={metrics['rmse']:.4f} "
            f"mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}"
        )
    return (
        f"score={metrics['score']:.4f} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
