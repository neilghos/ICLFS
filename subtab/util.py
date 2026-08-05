from __future__ import annotations

import enum


class TaskType(enum.Enum):
    REGRESSION = "regression"
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"


class PredictionType(enum.Enum):
    LABELS = "labels"
    PROBS = "probs"
    LOGITS = "logits"
