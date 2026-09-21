"""Shared, dependency-light segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


@dataclass(frozen=True)
class SegmentationMetrics:
    """Serializable metrics calculated from one accumulated confusion matrix."""

    pixel_accuracy: float
    mean_iou: float
    mean_f1: float
    mean_precision: float
    mean_recall: float
    frequency_weighted_iou: float
    per_class_iou: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    evaluated_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "mean_f1": self.mean_f1,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "frequency_weighted_iou": self.frequency_weighted_iou,
            "per_class_iou": list(self.per_class_iou),
            "per_class_f1": list(self.per_class_f1),
            "per_class_precision": list(self.per_class_precision),
            "per_class_recall": list(self.per_class_recall),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "evaluated_pixels": self.evaluated_pixels,
        }


class ConfusionMatrix:
    """Accumulate integer segmentation predictions (rows=true, columns=predicted)."""

    def __init__(self, num_classes: int, ignore_index: int | None = None) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        predicted = np.asarray(prediction)
        expected = np.asarray(target)
        if predicted.shape != expected.shape:
            raise ValueError(
                f"prediction and target shapes differ: {predicted.shape} vs {expected.shape}"
            )
        predicted = predicted.reshape(-1).astype(np.int64, copy=False)
        expected = expected.reshape(-1).astype(np.int64, copy=False)
        valid = (expected >= 0) & (expected < self.num_classes)
        if self.ignore_index is not None:
            valid &= expected != self.ignore_index
        if np.any((predicted[valid] < 0) | (predicted[valid] >= self.num_classes)):
            values = np.unique(
                predicted[valid][(predicted[valid] < 0) | (predicted[valid] >= self.num_classes)]
            )
            raise ValueError(f"predictions contain invalid class indices: {values.tolist()}")
        encoded = self.num_classes * expected[valid] + predicted[valid]
        self.matrix += np.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def merge(self, other: ConfusionMatrix) -> None:
        if (self.num_classes, self.ignore_index) != (
            other.num_classes,
            other.ignore_index,
        ):
            raise ValueError("cannot merge confusion matrices with different settings")
        self.matrix += other.matrix

    def compute(self) -> SegmentationMetrics:
        matrix = self.matrix.astype(np.float64)
        total = float(matrix.sum())
        true_positive = np.diag(matrix)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive
        support = matrix.sum(axis=1)
        iou = _safe_divide(true_positive, true_positive + false_positive + false_negative)
        f1 = _safe_divide(
            2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        present = support > 0

        def present_mean(values: np.ndarray) -> float:
            return float(np.mean(values[present])) if np.any(present) else 0.0

        frequency = _safe_divide(support, np.full_like(support, total))
        return SegmentationMetrics(
            pixel_accuracy=float(true_positive.sum() / total) if total else 0.0,
            mean_iou=present_mean(iou),
            mean_f1=present_mean(f1),
            mean_precision=present_mean(precision),
            mean_recall=present_mean(recall),
            frequency_weighted_iou=float(np.sum(frequency * iou)),
            per_class_iou=tuple(float(value) for value in iou),
            per_class_f1=tuple(float(value) for value in f1),
            per_class_precision=tuple(float(value) for value in precision),
            per_class_recall=tuple(float(value) for value in recall),
            confusion_matrix=tuple(tuple(int(value) for value in row) for row in self.matrix),
            evaluated_pixels=int(total),
        )


def evaluate_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> SegmentationMetrics:
    matrix = ConfusionMatrix(num_classes, ignore_index)
    matrix.update(prediction, target)
    return matrix.compute()
