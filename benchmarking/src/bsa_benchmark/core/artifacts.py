"""Non-overwriting experiment directory planning and metadata creation."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from bsa_benchmark import __version__
from bsa_benchmark.config.schemas import ResolvedExperimentConfig


def _serializable(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def configuration_fingerprint(value: BaseModel | Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _serializable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentDirectory:
    path: Path
    run_id: str
    fingerprint: str

    @classmethod
    def plan(
        cls,
        resolved: ResolvedExperimentConfig,
        *,
        output_root: Path | None = None,
        run_id: str | None = None,
        now: datetime | None = None,
    ) -> ExperimentDirectory:
        fingerprint = configuration_fingerprint(resolved)
        timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S.%fZ")
        chosen_id = run_id or f"{timestamp}_{fingerprint[:10]}"
        root = output_root or resolved.experiment.output_root
        path = root / resolved.dataset.id / resolved.method.id / resolved.experiment.id / chosen_id
        return cls(path=path, run_id=chosen_id, fingerprint=fingerprint)

    def create(
        self,
        resolved: ResolvedExperimentConfig,
        *,
        project_root: Path,
        command: tuple[str, ...] = (),
    ) -> ExperimentDirectory:
        self.path.mkdir(parents=True, exist_ok=False)
        subdirectories = (
            "splits",
            "checkpoints",
            "predictions",
            "metrics",
            "reports",
            "logs",
        )
        for name in subdirectories:
            (self.path / name).mkdir()

        resolved_data = resolved.model_dump(mode="json")
        (self.path / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_data, sort_keys=False), encoding="utf-8"
        )
        metadata = {
            "run_id": self.run_id,
            "workflow": "benchmark",
            "created_at": datetime.now(UTC).isoformat(),
            "seed": resolved.experiment.seed,
            "deterministic_operations": resolved.experiment.deterministic_operations,
            "configuration_fingerprint": self.fingerprint,
            "workbench_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "project_root": str(project_root.resolve()),
            "command": list(command),
        }
        (self.path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        status = {"state": "initialised", "updated_at": metadata["created_at"]}
        (self.path / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self


def describe_directory(directory: ExperimentDirectory) -> dict[str, Any]:
    data = asdict(directory)
    data["path"] = str(directory.path)
    return data
