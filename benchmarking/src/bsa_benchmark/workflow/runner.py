"""End-to-end configuration-driven benchmark execution."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bsa_benchmark.config.schemas import ResolvedExperimentConfig
from bsa_benchmark.core.artifacts import ExperimentDirectory
from bsa_benchmark.core.interfaces import MethodContext, SampleReference
from bsa_benchmark.core.seeding import seed_everything
from bsa_benchmark.data import (
    AugmentationPipeline,
    ImageMaskLoader,
    discover_dataset,
    generate_protocol_splits,
    validate_manifest_against_index,
    write_split_manifest,
)
from bsa_benchmark.data.loading import LoadedSample
from bsa_benchmark.evaluation import (
    EvaluationReport,
    evaluate_predictions,
    write_cross_validation_summary,
    write_evaluation_report,
)
from bsa_benchmark.methods import load_method


@dataclass(frozen=True)
class WorkflowResult:
    run_directory: Path
    reports: tuple[EvaluationReport, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_directory": str(self.run_directory),
            "status": self.status,
            "folds": [report.fold_id for report in self.reports],
            "method": self.reports[0].method_id if self.reports else None,
            "dataset": self.reports[0].dataset_id if self.reports else None,
        }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_status(run_directory: Path, state: str, **details: Any) -> None:
    _write_json(
        run_directory / "status.json",
        {"state": state, "updated_at": datetime.now(UTC).isoformat(), **details},
    )


def _select_references(
    references: dict[str, SampleReference], sample_ids: Sequence[str]
) -> tuple[SampleReference, ...]:
    return tuple(references[sample_id] for sample_id in sample_ids)


def _load_role(
    loader: ImageMaskLoader,
    references: Sequence[SampleReference],
    *,
    role: str,
    augmentation: AugmentationPipeline | None = None,
    epoch: int = 0,
) -> tuple[LoadedSample, ...]:
    return tuple(
        loader.load_for_role(
            reference,
            role=role,
            augmentation=augmentation if role == "train" else None,
            epoch=epoch,
        )
        for reference in references
    )


def _save_predictions(predictions: dict[str, np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for sample_id, prediction in predictions.items():
        array = np.asarray(prediction)
        if array.ndim != 2:
            raise ValueError(f"prediction for {sample_id!r} is not a 2D class-index mask")
        Image.fromarray(array.astype(np.uint8), mode="L").save(directory / f"{sample_id}.png")


def run_experiment(
    resolved: ResolvedExperimentConfig,
    *,
    project_root: Path,
    output_root: Path | None = None,
    run_id: str | None = None,
    fold_index: int | None = None,
    command: Sequence[str] = (),
) -> WorkflowResult:
    """Prepare splits, train, infer, evaluate, and report one configured experiment."""

    seed_everything(
        resolved.experiment.seed,
        deterministic_tensorflow=resolved.experiment.deterministic_operations,
        include_tensorflow=resolved.method.family.value == "deep_learning",
    )
    plan = ExperimentDirectory.plan(resolved, output_root=output_root, run_id=run_id).create(
        resolved, project_root=project_root, command=tuple(command)
    )
    _set_status(plan.path, "preparing")
    index = discover_dataset(resolved.dataset)
    manifests = generate_protocol_splits(index, resolved.experiment.protocol)
    if fold_index is not None:
        if fold_index < 0 or fold_index >= len(manifests):
            raise ValueError(f"fold index {fold_index} is outside 0..{len(manifests) - 1}")
        manifests = (manifests[fold_index],)
    for manifest in manifests:
        validate_manifest_against_index(manifest, index, require_complete=True)
        write_split_manifest(manifest, plan.path / "splits" / f"{manifest.fold_id}.json")

    method = load_method(resolved.method)
    loader = ImageMaskLoader(resolved.dataset)
    by_id = index.by_id()
    augmentation = (
        AugmentationPipeline(resolved.dataset.augmentation, resolved.experiment.seed)
        if resolved.dataset.augmentation.enabled
        else None
    )
    reports: list[EvaluationReport] = []
    for manifest_number, manifest in enumerate(manifests):
        fold_seed = resolved.experiment.seed + manifest_number
        context = MethodContext(
            run_directory=plan.path,
            seed=fold_seed,
            deterministic_operations=resolved.experiment.deterministic_operations,
            fold_index=fold_index if fold_index is not None else manifest_number,
        )
        _set_status(plan.path, "training", fold_id=manifest.fold_id)
        train_samples = _load_role(
            loader,
            _select_references(by_id, manifest.train_ids),
            role="train",
            augmentation=augmentation,
            epoch=manifest_number,
        )
        validation_samples = _load_role(
            loader,
            _select_references(by_id, manifest.validation_ids),
            role="validation",
        )
        test_samples = _load_role(
            loader,
            _select_references(by_id, manifest.test_ids),
            role="test",
        )
        model, history = method.fit(train_samples, validation_samples, context)
        suffix = ".keras" if resolved.method.family.value == "deep_learning" else ".joblib"
        method.save_model(model, plan.path / "checkpoints" / f"{manifest.fold_id}{suffix}")
        _write_json(plan.path / "logs" / f"{manifest.fold_id}_history.json", history)

        _set_status(plan.path, "inference", fold_id=manifest.fold_id)
        predictions = dict(method.predict(model, test_samples, context))
        _save_predictions(predictions, plan.path / "predictions" / manifest.fold_id)
        targets = {sample.sample_id: sample.mask for sample in test_samples}
        report = evaluate_predictions(
            predictions,
            targets,
            dataset_id=resolved.dataset.id,
            method_id=resolved.method.id,
            fold_id=manifest.fold_id,
            class_names=resolved.dataset.masks.classes,
            ignore_index=resolved.dataset.masks.ignore_index,
        )
        write_evaluation_report(report, plan.path / "metrics" / manifest.fold_id)
        reports.append(report)

        del model, train_samples, validation_samples, test_samples

    write_cross_validation_summary(reports, plan.path / "reports" / "summary.json")
    _set_status(plan.path, "complete", completed_folds=[item.fold_id for item in reports])
    return WorkflowResult(plan.path, tuple(reports), "complete")
