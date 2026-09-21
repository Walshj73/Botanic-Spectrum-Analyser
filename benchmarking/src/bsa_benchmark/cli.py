"""Command-line foundation for the BSA benchmarking workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from bsa_benchmark.config.loader import (
    ConfigLoadError,
    check_dataset_paths,
    find_project_root,
    load_dataset_config,
    load_method_config,
    resolve_experiment,
)
from bsa_benchmark.config.schemas import ResolvedExperimentConfig
from bsa_benchmark.core.artifacts import (
    ExperimentDirectory,
    configuration_fingerprint,
    describe_directory,
)
from bsa_benchmark.data import (
    discover_dataset,
    generate_protocol_splits,
    validate_dataset_contents,
    write_split_manifest,
)
from bsa_benchmark.methods import OptionalDependencyError
from bsa_benchmark.workflow import run_experiment


def _config_path(value: str, kind: str, project_root: Path) -> Path:
    supplied = Path(value)
    if supplied.exists() or supplied.suffix in {".yaml", ".yml"} or supplied.parent != Path("."):
        return supplied if supplied.is_absolute() else project_root / supplied
    plural = {
        "dataset": "datasets",
        "method": "methods",
        "experiment": "experiments",
    }[kind]
    return project_root / "configs" / plural / f"{value}.yaml"


def _apply_overrides(
    resolved: ResolvedExperimentConfig,
    args: argparse.Namespace,
    project_root: Path,
) -> ResolvedExperimentConfig:
    dataset = resolved.dataset
    method = resolved.method
    experiment_updates: dict[str, Path] = {}
    if args.dataset:
        path = _config_path(args.dataset, "dataset", project_root)
        dataset = load_dataset_config(path, project_root)
        experiment_updates["dataset_config"] = path.resolve()
    if args.method:
        path = _config_path(args.method, "method", project_root)
        method = load_method_config(path)
        experiment_updates["method_config"] = path.resolve()
    experiment = resolved.experiment.model_copy(update=experiment_updates)
    try:
        return ResolvedExperimentConfig(
            experiment=experiment,
            dataset=dataset,
            method=method,
        )
    except ValidationError as exc:
        raise ConfigLoadError(f"invalid command-line configuration override:\n{exc}") from exc


def _resolve_from_args(args: argparse.Namespace, project_root: Path) -> ResolvedExperimentConfig:
    experiment_path = Path(args.experiment)
    if not experiment_path.is_absolute():
        candidate = project_root / experiment_path
        if candidate.exists() or experiment_path.suffix in {".yaml", ".yml"}:
            experiment_path = candidate
        else:
            experiment_path = project_root / "configs" / "experiments" / f"{args.experiment}.yaml"
    return _apply_overrides(resolve_experiment(experiment_path, project_root), args, project_root)


def _validate(args: argparse.Namespace, project_root: Path) -> dict[str, object]:
    path = Path(args.config)
    if not path.is_absolute():
        candidate = project_root / path
        path = (
            candidate if candidate.exists() else _config_path(args.config, args.kind, project_root)
        )

    if args.kind == "dataset":
        config = load_dataset_config(path, project_root)
        result: dict[str, object] = {"kind": "dataset", "id": config.id, "valid": True}
        if args.check_paths:
            report = check_dataset_paths(config)
            result["paths"] = {
                "valid": report.valid,
                "image_directory": str(report.image_directory),
                "mask_directory": str(report.mask_directory),
                "image_count": report.image_count,
                "mask_count": report.mask_count,
                "paired_count": report.paired_count,
                "unmatched_images": list(report.unmatched_images),
                "unmatched_masks": list(report.unmatched_masks),
            }
            if not report.valid:
                raise ConfigLoadError(f"dataset pairing check failed for {config.id}")
        if args.check_contents:
            content = validate_dataset_contents(config)
            result["contents"] = {
                "sample_count": content.sample_count,
                "fingerprint": content.fingerprint,
                "image_shapes": content.image_shapes,
                "mask_shapes": content.mask_shapes,
                "image_dtypes": content.image_dtypes,
                "mask_values": content.mask_values,
            }
        return result
    if args.kind == "method":
        config = load_method_config(path)
        return {"kind": "method", "id": config.id, "valid": True}

    resolved = resolve_experiment(path, project_root)
    result = {
        "kind": "experiment",
        "id": resolved.experiment.id,
        "dataset": resolved.dataset.id,
        "method": resolved.method.id,
        "protocol": resolved.experiment.protocol.type.value,
        "valid": True,
    }
    if args.check_paths:
        report = check_dataset_paths(resolved.dataset)
        result["dataset_paths_valid"] = report.valid
        result["paired_samples"] = report.paired_count
        if not report.valid:
            raise ConfigLoadError(f"dataset pairing check failed for {resolved.dataset.id}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsa-benchmark",
        description="Configuration-driven benchmarking for the BSA manuscript",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a YAML configuration")
    validate.add_argument("kind", choices=("dataset", "method", "experiment"))
    validate.add_argument("config", help="configuration path or short name")
    validate.add_argument("--check-paths", action="store_true")
    validate.add_argument("--check-contents", action="store_true")

    for command, help_text in (
        ("dry-run", "resolve an experiment and show the planned benchmark output"),
        ("init-run", "create a new metadata-only benchmark directory"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--experiment", required=True, help="path or short name")
        subparser.add_argument("--dataset", help="optional dataset config override")
        subparser.add_argument("--method", help="optional method config override")
        subparser.add_argument("--output-root", type=Path)
        subparser.add_argument("--run-id", help="explicit unique run identifier")
        subparser.add_argument("--check-paths", action="store_true")

    split = subparsers.add_parser(
        "make-splits", help="generate reproducible, group-safe split manifests"
    )
    split.add_argument("--experiment", required=True, help="path or short name")
    split.add_argument("--dataset", help="optional dataset config override")
    split.add_argument("--method", help="optional method config override")
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser(
        "run", help="prepare, train, infer, evaluate, and report one experiment"
    )
    run.add_argument("--experiment", required=True, help="path or short name")
    run.add_argument("--dataset", help="optional dataset config override")
    run.add_argument("--method", help="optional method config override")
    run.add_argument("--output-root", type=Path)
    run.add_argument("--run-id", help="explicit unique run identifier")
    run.add_argument("--fold", type=int, help="run only this zero-based cross-validation fold")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = find_project_root()
        if args.command == "validate":
            print(json.dumps(_validate(args, project_root), indent=2, sort_keys=True))
            return 0

        resolved = _resolve_from_args(args, project_root)
        if args.command == "run":
            output_root = args.output_root
            if output_root is not None and not output_root.is_absolute():
                output_root = (project_root / output_root).resolve()
            completed = run_experiment(
                resolved,
                project_root=project_root,
                output_root=output_root,
                run_id=args.run_id,
                fold_index=args.fold,
                command=tuple(sys.argv if argv is None else ("bsa-benchmark", *argv)),
            )
            print(json.dumps(completed.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "make-splits":
            index = discover_dataset(resolved.dataset)
            generated = generate_protocol_splits(index, resolved.experiment.protocol)
            output = args.output if args.output.is_absolute() else project_root / args.output
            manifests = [(item, output / f"{item.fold_id}.json") for item in generated]
            summary = {
                "dataset": resolved.dataset.id,
                "dataset_fingerprint": index.fingerprint,
                "experiment": resolved.experiment.id,
                "protocol": resolved.experiment.protocol.type.value,
                "seed": resolved.experiment.protocol.split_seed,
                "sample_count": len(index.samples),
                "manifest_count": len(manifests),
                "output": str(output.resolve()),
                "manifest_paths": [str(path.resolve()) for _, path in manifests],
                "creates_files": not args.dry_run,
            }
            if not args.dry_run:
                output.mkdir(parents=True, exist_ok=False)
                for manifest, path in manifests:
                    write_split_manifest(manifest, path)
                summary["configuration_fingerprint"] = configuration_fingerprint(resolved)
                (output / "split_index.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.check_paths:
            report = check_dataset_paths(resolved.dataset)
            if not report.valid:
                raise ConfigLoadError(f"dataset pairing check failed for {resolved.dataset.id}")
        output_root = args.output_root
        if output_root is not None and not output_root.is_absolute():
            output_root = (project_root / output_root).resolve()
        directory = ExperimentDirectory.plan(
            resolved,
            output_root=output_root,
            run_id=args.run_id,
        )
        result = describe_directory(directory)
        result.update(
            dataset=resolved.dataset.id,
            method=resolved.method.id,
            experiment=resolved.experiment.id,
            seed=resolved.experiment.seed,
            creates_files=args.command == "init-run",
        )
        if args.command == "init-run":
            directory.create(resolved, project_root=project_root, command=tuple(sys.argv))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ConfigLoadError,
        FileExistsError,
        OSError,
        OptionalDependencyError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
