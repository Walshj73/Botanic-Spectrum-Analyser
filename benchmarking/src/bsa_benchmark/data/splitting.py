"""Group-aware, reproducible development and cross-validation splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bsa_benchmark.config.schemas import EvaluationProtocol, ProtocolConfig
from bsa_benchmark.core.interfaces import SampleReference
from bsa_benchmark.data.index import DatasetIndex, DatasetValidationError


@dataclass(frozen=True)
class SplitManifest:
    dataset_id: str
    dataset_fingerprint: str
    protocol: str
    seed: int
    fold_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    source_groups: Mapping[str, str]
    schema_version: int = 1

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol": self.protocol,
            "seed": self.seed,
            "fold_id": self.fold_id,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "source_groups": dict(sorted(self.source_groups.items())),
        }
        if include_fingerprint:
            data["manifest_fingerprint"] = self.fingerprint
        return data

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _grouped(samples: Sequence[SampleReference]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    sample_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in sample_ids:
            raise DatasetValidationError(f"duplicate sample ID: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        group = sample.group_id or sample.sample_id
        grouped.setdefault(group, []).append(sample.sample_id)
    return {group: tuple(sorted(ids)) for group, ids in grouped.items()}


def _balanced_group_folds(
    grouped: Mapping[str, tuple[str, ...]], folds: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    if folds < 2:
        raise ValueError("fold count must be at least two")
    if len(grouped) < folds:
        raise ValueError(f"cannot create {folds} folds from {len(grouped)} source groups")
    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: len(grouped[group]), reverse=True)
    buckets: list[list[str]] = [[] for _ in range(folds)]
    sizes = [0] * folds
    for group in groups:
        destination = min(range(folds), key=lambda index: (sizes[index], index))
        buckets[destination].append(group)
        sizes[destination] += len(grouped[group])
    return tuple(tuple(bucket) for bucket in buckets)


def _select_fraction(
    grouped: Mapping[str, tuple[str, ...]],
    groups: Iterable[str],
    fraction: float,
    seed: int,
) -> tuple[str, ...]:
    candidates = list(groups)
    if not candidates or fraction <= 0:
        return ()
    random.Random(seed).shuffle(candidates)
    total_samples = sum(len(grouped[group]) for group in candidates)
    target = max(1, round(total_samples * fraction))
    selected: list[str] = []
    selected_samples = 0
    for group in candidates:
        if len(selected) >= len(candidates) - 1:
            break
        selected.append(group)
        selected_samples += len(grouped[group])
        if selected_samples >= target:
            break
    return tuple(selected)


def _ids(grouped: Mapping[str, tuple[str, ...]], groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(sample_id for group in groups for sample_id in grouped[group]))


def _manifest(
    index: DatasetIndex,
    protocol: str,
    seed: int,
    fold_id: str,
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
) -> SplitManifest:
    sample_groups = {
        sample.sample_id: sample.group_id or sample.sample_id for sample in index.samples
    }
    included = set(train_ids) | set(validation_ids) | set(test_ids)
    manifest = SplitManifest(
        dataset_id=index.dataset_id,
        dataset_fingerprint=index.fingerprint,
        protocol=protocol,
        seed=seed,
        fold_id=fold_id,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        source_groups={sample_id: sample_groups[sample_id] for sample_id in sorted(included)},
    )
    assert_no_source_leakage(manifest)
    return manifest


def assert_no_source_leakage(manifest: SplitManifest) -> None:
    roles = {
        "train": set(manifest.train_ids),
        "validation": set(manifest.validation_ids),
        "test": set(manifest.test_ids),
    }
    role_names = tuple(roles)
    for index, left_name in enumerate(role_names):
        for right_name in role_names[index + 1 :]:
            overlap = roles[left_name] & roles[right_name]
            if overlap:
                raise DatasetValidationError(
                    f"sample leakage between {left_name} and {right_name}: {sorted(overlap)[:5]}"
                )
            left_groups = {manifest.source_groups[sample] for sample in roles[left_name]}
            right_groups = {manifest.source_groups[sample] for sample in roles[right_name]}
            group_overlap = left_groups & right_groups
            if group_overlap:
                raise DatasetValidationError(
                    f"source-level leakage between {left_name} and {right_name}: "
                    f"{sorted(group_overlap)[:5]}"
                )


def validate_manifest_against_index(
    manifest: SplitManifest,
    index: DatasetIndex,
    *,
    require_complete: bool = False,
) -> None:
    """Reject stale manifests or manifests referring to different source groups."""

    if manifest.dataset_id != index.dataset_id:
        raise DatasetValidationError(
            f"manifest dataset {manifest.dataset_id!r} does not match {index.dataset_id!r}"
        )
    if manifest.dataset_fingerprint != index.fingerprint:
        raise DatasetValidationError("manifest dataset fingerprint does not match current data")
    indexed_groups = {
        sample.sample_id: sample.group_id or sample.sample_id for sample in index.samples
    }
    manifest_ids = set(manifest.train_ids + manifest.validation_ids + manifest.test_ids)
    unknown = manifest_ids - set(indexed_groups)
    if unknown:
        raise DatasetValidationError(f"manifest contains unknown samples: {sorted(unknown)[:5]}")
    for sample_id in manifest_ids:
        if manifest.source_groups.get(sample_id) != indexed_groups[sample_id]:
            raise DatasetValidationError(f"source group changed for sample {sample_id!r}")
    if require_complete and manifest_ids != set(indexed_groups):
        missing = set(indexed_groups) - manifest_ids
        raise DatasetValidationError(
            f"manifest does not cover the complete dataset: {sorted(missing)[:5]}"
        )
    assert_no_source_leakage(manifest)


def generate_development_split(
    index: DatasetIndex,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitManifest:
    grouped = _grouped(index.samples)
    all_groups = tuple(grouped)
    test_groups = _select_fraction(grouped, all_groups, test_fraction, seed)
    remaining = tuple(group for group in all_groups if group not in set(test_groups))
    validation_groups = _select_fraction(grouped, remaining, validation_fraction, seed + 1)
    train_groups = tuple(group for group in remaining if group not in set(validation_groups))
    if not train_groups or not validation_groups or not test_groups:
        raise ValueError("development split requires non-empty train, validation, and test sets")
    return _manifest(
        index,
        EvaluationProtocol.DEVELOPMENT.value,
        seed,
        "development",
        _ids(grouped, train_groups),
        _ids(grouped, validation_groups),
        _ids(grouped, test_groups),
    )


def generate_cross_validation(
    index: DatasetIndex,
    *,
    folds: int,
    validation_fraction: float,
    seed: int,
) -> tuple[SplitManifest, ...]:
    grouped = _grouped(index.samples)
    buckets = _balanced_group_folds(grouped, folds, seed)
    manifests: list[SplitManifest] = []
    for fold_index, test_groups in enumerate(buckets):
        test_set = set(test_groups)
        remaining = tuple(group for group in grouped if group not in test_set)
        validation_groups = _select_fraction(
            grouped, remaining, validation_fraction, seed + fold_index + 1
        )
        validation_set = set(validation_groups)
        train_groups = tuple(group for group in remaining if group not in validation_set)
        manifests.append(
            _manifest(
                index,
                EvaluationProtocol.CROSS_VALIDATION.value,
                seed,
                f"fold_{fold_index}",
                _ids(grouped, train_groups),
                _ids(grouped, validation_groups),
                _ids(grouped, test_groups),
            )
        )
    return tuple(manifests)


def generate_protocol_splits(
    index: DatasetIndex, protocol: ProtocolConfig
) -> tuple[SplitManifest, ...]:
    if protocol.type == EvaluationProtocol.DEVELOPMENT:
        if protocol.validation_fraction is None or protocol.test_fraction is None:
            raise ValueError("development splitting requires validation and test fractions")
        return (
            generate_development_split(
                index,
                validation_fraction=protocol.validation_fraction,
                test_fraction=protocol.test_fraction,
                seed=protocol.split_seed,
            ),
        )
    if protocol.type == EvaluationProtocol.CROSS_VALIDATION:
        if protocol.folds is None or protocol.validation_fraction is None:
            raise ValueError("cross-validation requires folds and validation_fraction")
        return generate_cross_validation(
            index,
            folds=protocol.folds,
            validation_fraction=protocol.validation_fraction,
            seed=protocol.split_seed,
        )
    raise ValueError(
        "held-out test manifests must be reused from their originating development split"
    )


def write_split_manifest(manifest: SplitManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def read_split_manifest(path: Path) -> SplitManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"could not read split manifest {path}: {exc}") from exc
    try:
        recorded_fingerprint = raw.pop("manifest_fingerprint", None)
        manifest = SplitManifest(
            schema_version=int(raw["schema_version"]),
            dataset_id=str(raw["dataset_id"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            protocol=str(raw["protocol"]),
            seed=int(raw["seed"]),
            fold_id=str(raw["fold_id"]),
            train_ids=tuple(raw["train_ids"]),
            validation_ids=tuple(raw["validation_ids"]),
            test_ids=tuple(raw["test_ids"]),
            source_groups=dict(raw["source_groups"]),
        )
        assert_no_source_leakage(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"invalid split manifest {path}: {exc}") from exc
    if recorded_fingerprint != manifest.fingerprint:
        raise DatasetValidationError(f"split manifest fingerprint mismatch: {path}")
    return manifest
