"""Write transparent JSON, CSV, and Markdown benchmark reports."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bsa_benchmark.evaluation.metrics import ConfusionMatrix, SegmentationMetrics


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    metrics: SegmentationMetrics


@dataclass(frozen=True)
class EvaluationReport:
    dataset_id: str
    method_id: str
    fold_id: str
    class_names: tuple[str, ...]
    aggregate: SegmentationMetrics
    samples: tuple[SampleEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "method_id": self.method_id,
            "fold_id": self.fold_id,
            "class_names": list(self.class_names),
            "aggregate": self.aggregate.to_dict(),
            "samples": [
                {"sample_id": sample.sample_id, **sample.metrics.to_dict()}
                for sample in self.samples
            ],
        }


def evaluate_predictions(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    dataset_id: str,
    method_id: str,
    fold_id: str,
    class_names: Sequence[str],
    ignore_index: int | None = None,
) -> EvaluationReport:
    missing = sorted(set(targets) - set(predictions))
    unexpected = sorted(set(predictions) - set(targets))
    if missing or unexpected:
        raise ValueError(
            f"prediction IDs do not match targets; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    aggregate = ConfusionMatrix(len(class_names), ignore_index)
    samples: list[SampleEvaluation] = []
    for sample_id in sorted(targets):
        per_sample = ConfusionMatrix(len(class_names), ignore_index)
        per_sample.update(predictions[sample_id], targets[sample_id])
        aggregate.update(predictions[sample_id], targets[sample_id])
        samples.append(SampleEvaluation(sample_id, per_sample.compute()))
    return EvaluationReport(
        dataset_id=dataset_id,
        method_id=method_id,
        fold_id=fold_id,
        class_names=tuple(class_names),
        aggregate=aggregate.compute(),
        samples=tuple(samples),
    )


def write_evaluation_report(report: EvaluationReport, directory: Path) -> dict[str, Path]:
    """Write a report into a new fold directory without overwriting files."""

    directory.mkdir(parents=True, exist_ok=False)
    json_path = directory / "metrics.json"
    csv_path = directory / "per_sample.csv"
    markdown_path = directory / "summary.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = (
        "sample_id",
        "pixel_accuracy",
        "mean_iou",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "frequency_weighted_iou",
        "evaluated_pixels",
    )
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in report.samples:
            values = sample.metrics.to_dict()
            writer.writerow(
                {name: sample.sample_id if name == "sample_id" else values[name] for name in fields}
            )

    aggregate = report.aggregate
    rows = [
        ("Pixel accuracy", aggregate.pixel_accuracy),
        ("Mean IoU", aggregate.mean_iou),
        ("Mean F1 (Dice)", aggregate.mean_f1),
        ("Mean precision", aggregate.mean_precision),
        ("Mean recall", aggregate.mean_recall),
        ("Frequency-weighted IoU", aggregate.frequency_weighted_iou),
    ]
    lines = [
        f"# {report.method_id} — {report.dataset_id} — {report.fold_id}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *(f"| {name} | {value:.6f} |" for name, value in rows),
        "",
        "| Class | IoU | F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(report.class_names):
        lines.append(
            f"| {name} | {aggregate.per_class_iou[index]:.6f} | "
            f"{aggregate.per_class_f1[index]:.6f} | "
            f"{aggregate.per_class_precision[index]:.6f} | "
            f"{aggregate.per_class_recall[index]:.6f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def write_cross_validation_summary(reports: Sequence[EvaluationReport], destination: Path) -> Path:
    if not reports:
        raise ValueError("at least one fold report is required")
    keys = (
        "pixel_accuracy",
        "mean_iou",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "frequency_weighted_iou",
    )
    folds = []
    for report in reports:
        metrics = report.aggregate.to_dict()
        folds.append({"fold_id": report.fold_id, **{key: metrics[key] for key in keys}})
    summary = {
        key: {
            "mean": float(np.mean([fold[key] for fold in folds])),
            "standard_deviation": float(np.std([fold[key] for fold in folds])),
        }
        for key in keys
    }
    payload = {
        "schema_version": 1,
        "dataset_id": reports[0].dataset_id,
        "method_id": reports[0].method_id,
        "fold_count": len(reports),
        "folds": folds,
        "summary": summary,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination
