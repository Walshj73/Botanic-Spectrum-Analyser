"""Central seed control for BSA Benchmark random-number generators."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python: bool
    numpy: bool
    tensorflow: bool
    tensorflow_determinism: bool
    notes: tuple[str, ...]


def seed_everything(
    seed: int,
    *,
    deterministic_tensorflow: bool = True,
    include_tensorflow: bool = True,
) -> SeedReport:
    """Seed available libraries before dataset shuffling or model construction.

    Setting ``PYTHONHASHSEED`` affects child processes; the current interpreter's
    hash seed is fixed at startup. Data APIs must still receive this same seed
    explicitly, which is why it is part of all split and method contexts.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    notes: list[str] = []

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        notes.append("NumPy is not installed")

    tensorflow_seeded = False
    tensorflow_determinism = False
    if include_tensorflow:
        try:
            if deterministic_tensorflow:
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            tensorflow_seeded = True
            if deterministic_tensorflow:
                try:
                    tf.config.experimental.enable_op_determinism()
                    tensorflow_determinism = True
                except (AttributeError, RuntimeError) as exc:
                    notes.append(f"TensorFlow deterministic operations unavailable: {exc}")
        except ImportError:
            notes.append("TensorFlow is not installed")

    return SeedReport(
        seed=seed,
        python=True,
        numpy=numpy_seeded,
        tensorflow=tensorflow_seeded,
        tensorflow_determinism=tensorflow_determinism,
        notes=tuple(notes),
    )


def numpy_generator(seed: int) -> Any:
    """Return a local NumPy generator for deterministic data-processing code."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to create a local generator") from exc
    return np.random.default_rng(seed)
