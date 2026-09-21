"""Evaluation and report-generation public API."""

from bsa_benchmark.evaluation.metrics import (
    ConfusionMatrix,
    SegmentationMetrics,
    evaluate_prediction,
)
from bsa_benchmark.evaluation.reporting import (
    EvaluationReport,
    SampleEvaluation,
    evaluate_predictions,
    write_cross_validation_summary,
    write_evaluation_report,
)

__all__ = [
    "ConfusionMatrix",
    "EvaluationReport",
    "SampleEvaluation",
    "SegmentationMetrics",
    "evaluate_prediction",
    "evaluate_predictions",
    "write_cross_validation_summary",
    "write_evaluation_report",
]
