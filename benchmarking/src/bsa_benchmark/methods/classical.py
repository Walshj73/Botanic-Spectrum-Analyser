"""Random Forest, linear SVM, and the reconstructed SLIC-RF baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bsa_benchmark.core.interfaces import MethodContext
from bsa_benchmark.data.loading import LoadedSample
from bsa_benchmark.methods.base import OptionalDependencyError, SegmentationMethod
from bsa_benchmark.methods.features import extract_colour_texture_features, sample_pixel_matrix


def _sklearn() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise OptionalDependencyError(
            "classical methods require: pip install 'bsa-benchmark[classical]'"
        ) from exc
    return joblib, RandomForestClassifier, SGDClassifier, make_pipeline, StandardScaler


class PixelClassifierSegmentation(SegmentationMethod):
    estimator_kind = ""

    def _build_estimator(self, seed: int) -> Any:
        _, random_forest, sgd_classifier, make_pipeline, standard_scaler = _sklearn()
        parameters = dict(self.config.hyperparameters)
        if self.estimator_kind == "random_forest":
            return random_forest(random_state=seed, **parameters)
        if self.estimator_kind == "svm":
            return make_pipeline(standard_scaler(), sgd_classifier(random_state=seed, **parameters))
        raise RuntimeError(f"unsupported estimator kind: {self.estimator_kind}")

    def fit(
        self,
        train_samples: Sequence[LoadedSample],
        validation_samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> tuple[Any, Mapping[str, Sequence[float]]]:
        del validation_samples
        training = self.config.training
        matrix, target = sample_pixel_matrix(
            [(sample.image, sample.mask) for sample in train_samples],
            pixels_per_image=training.get("pixels_per_image"),
            maximum_pixels=training.get("max_training_pixels"),
            ignore_index=training.get("ignore_index"),
            seed=context.seed,
        )
        model = self._build_estimator(context.seed)
        model.fit(matrix, target)
        return model, {"training_pixels": (float(len(target)),)}

    def predict(
        self,
        model: Any,
        samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> Mapping[str, np.ndarray]:
        del context
        predictions: dict[str, np.ndarray] = {}
        for sample in samples:
            features = extract_colour_texture_features(sample.image)
            height, width, channels = features.shape
            predicted = model.predict(features.reshape(-1, channels))
            predictions[sample.sample_id] = (
                np.asarray(predicted).reshape(height, width).astype(np.uint8)
            )
        return predictions

    def save_model(self, model: Any, destination: Path) -> Path:
        joblib, *_ = _sklearn()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        joblib.dump(model, destination, compress=3)
        return destination


class RandomForestSegmentation(PixelClassifierSegmentation):
    estimator_kind = "random_forest"


class SVMSegmentation(PixelClassifierSegmentation):
    estimator_kind = "svm"


class SuperpixelRandomForestSegmentation(SegmentationMethod):
    """New SLIC-region classifier; not the missing manuscript implementation."""

    def _segments(self, image: np.ndarray) -> np.ndarray:
        try:
            from skimage.segmentation import slic
        except ImportError as exc:
            raise OptionalDependencyError(
                "SLIC-RF requires: pip install 'bsa-benchmark[classical]'"
            ) from exc
        settings = dict(self.config.hyperparameters.get("slic", {}))
        return slic(np.asarray(image)[..., :3], channel_axis=-1, **settings).astype(np.int32)

    @staticmethod
    def _region_features(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image)[..., :3].astype(np.float32)
        if float(np.max(rgb, initial=0.0)) > 1.0:
            rgb /= 255.0
        height, width = segments.shape
        yy, xx = np.mgrid[:height, :width]
        region_ids = np.unique(segments)
        rows: list[np.ndarray] = []
        for region_id in region_ids:
            selected = segments == region_id
            pixels = rgb[selected]
            geometry = np.array(
                [
                    float(np.mean(xx[selected])) / max(width - 1, 1),
                    float(np.mean(yy[selected])) / max(height - 1, 1),
                    float(np.mean(selected)),
                ],
                dtype=np.float32,
            )
            rows.append(np.concatenate([pixels.mean(axis=0), pixels.std(axis=0), geometry]))
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _region_labels(mask: np.ndarray, segments: np.ndarray) -> np.ndarray:
        labels = []
        for region_id in np.unique(segments):
            values, counts = np.unique(mask[segments == region_id], return_counts=True)
            labels.append(int(values[np.argmax(counts)]))
        return np.asarray(labels, dtype=np.int32)

    def fit(
        self,
        train_samples: Sequence[LoadedSample],
        validation_samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> tuple[Any, Mapping[str, Sequence[float]]]:
        del validation_samples
        _, random_forest, *_ = _sklearn()
        matrices, labels = [], []
        for sample in train_samples:
            segments = self._segments(sample.image)
            matrices.append(self._region_features(sample.image, segments))
            labels.append(self._region_labels(sample.mask, segments))
        classifier = random_forest(
            random_state=context.seed,
            **dict(self.config.hyperparameters.get("classifier", {})),
        )
        target = np.concatenate(labels)
        classifier.fit(np.concatenate(matrices), target)
        return classifier, {"training_superpixels": (float(len(target)),)}

    def predict(
        self,
        model: Any,
        samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> Mapping[str, np.ndarray]:
        del context
        predictions: dict[str, np.ndarray] = {}
        for sample in samples:
            segments = self._segments(sample.image)
            region_ids = np.unique(segments)
            region_predictions = model.predict(self._region_features(sample.image, segments))
            output = np.zeros(segments.shape, dtype=np.uint8)
            for region_id, label in zip(region_ids, region_predictions, strict=True):
                output[segments == region_id] = int(label)
            predictions[sample.sample_id] = output
        return predictions

    def save_model(self, model: Any, destination: Path) -> Path:
        joblib, *_ = _sklearn()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        joblib.dump(model, destination, compress=3)
        return destination
