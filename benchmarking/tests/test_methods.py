import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bsa_benchmark.config.loader import find_project_root, load_method_config
from bsa_benchmark.config.schemas import MethodConfig, MethodFamily
from bsa_benchmark.core.interfaces import MethodContext
from bsa_benchmark.data.loading import LoadedSample
from bsa_benchmark.methods import SegmentationMethod, load_method
from bsa_benchmark.methods.classical import (
    RandomForestSegmentation,
    SuperpixelRandomForestSegmentation,
    SVMSegmentation,
)
from bsa_benchmark.methods.deep_learning import SDAUNetSegmentation, deeplab_rates

PROJECT_ROOT = find_project_root(Path(__file__))


class MethodConfigurationTests(unittest.TestCase):
    method_ids = (
        "random_forest",
        "svm",
        "slic_rf_v1",
        "fcn_resnet50",
        "deeplabv3plus",
        "pspnet",
        "bsa_sda_unet",
    )

    def test_every_documented_method_resolves_without_loading_optional_frameworks(self) -> None:
        for method_id in self.method_ids:
            with self.subTest(method=method_id):
                config = load_method_config(
                    PROJECT_ROOT / "configs" / "methods" / f"{method_id}.yaml"
                )
                self.assertIsInstance(load_method(config), SegmentationMethod)

    def test_deeplab_rates_follow_input_resolution(self) -> None:
        self.assertEqual((2, 4, 6), deeplab_rates(128))
        self.assertEqual((6, 12, 18), deeplab_rates(512))


class ReconstructedSlicTests(unittest.TestCase):
    def test_region_features_and_majority_labels(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:, 2:, 0] = 1.0
        segments = np.zeros((4, 4), dtype=np.int32)
        segments[:, 2:] = 1
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, 2:] = 1
        features = SuperpixelRandomForestSegmentation._region_features(image, segments)
        labels = SuperpixelRandomForestSegmentation._region_labels(mask, segments)
        self.assertEqual((2, 9), features.shape)
        np.testing.assert_array_equal(np.array([0, 1]), labels)
        self.assertLess(features[0, 0], features[1, 0])


class DeepInferenceConfigurationTests(unittest.TestCase):
    def test_configured_inference_batch_size_is_used(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def predict(self, images, verbose=0):
                del verbose
                self.batch_sizes.append(len(images))
                return np.zeros((*images.shape[:3], 1), dtype=np.float32)

        config = MethodConfig(
            id="bsa",
            display_name="BSA",
            family=MethodFamily.DEEP_LEARNING,
            implementation="unused:unused",
            implementation_version="test",
            accepted_input_channels=(3,),
            inference={"batch_size": 2},
        )
        samples = tuple(
            LoadedSample(
                f"sample_{index}",
                f"sample_{index}",
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((8, 8), dtype=np.uint8),
            )
            for index in range(3)
        )
        model = FakeModel()
        predictions = SDAUNetSegmentation(config).predict(
            model, samples, MethodContext(Path("."), 42, True)
        )
        self.assertEqual([2, 1], model.batch_sizes)
        self.assertEqual({"sample_0", "sample_1", "sample_2"}, set(predictions))


@unittest.skipUnless(
    importlib.util.find_spec("sklearn") and importlib.util.find_spec("skimage"),
    "classical extras are not installed",
)
class ClassicalMethodSmokeTests(unittest.TestCase):
    @staticmethod
    def samples() -> tuple[LoadedSample, ...]:
        samples = []
        for index in range(4):
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[:, 8:] = 1
            image = np.zeros((16, 16, 3), dtype=np.float32)
            image[..., 1] = 0.1 * index
            image[:, 8:, 0] = 1.0
            samples.append(LoadedSample(f"sample_{index}", f"sample_{index}", image, mask))
        return tuple(samples)

    def test_random_forest_and_svm_fit_and_predict(self) -> None:
        configurations = (
            (
                RandomForestSegmentation,
                MethodConfig(
                    id="rf",
                    display_name="RF",
                    family=MethodFamily.CLASSICAL,
                    implementation="unused:unused",
                    implementation_version="test",
                    accepted_input_channels=(3,),
                    hyperparameters={"n_estimators": 4, "max_depth": 4, "n_jobs": 1},
                    training={"pixels_per_image": 128, "max_training_pixels": 512},
                ),
            ),
            (
                SVMSegmentation,
                MethodConfig(
                    id="svm",
                    display_name="SVM",
                    family=MethodFamily.CLASSICAL,
                    implementation="unused:unused",
                    implementation_version="test",
                    accepted_input_channels=(3,),
                    hyperparameters={"loss": "hinge", "max_iter": 100, "tol": 0.001},
                    training={"pixels_per_image": 128, "max_training_pixels": 512},
                ),
            ),
        )
        with TemporaryDirectory() as temporary:
            context = MethodContext(Path(temporary), 42, True)
            for implementation, config in configurations:
                with self.subTest(method=config.id):
                    method = implementation(config)
                    model, history = method.fit(self.samples()[:3], self.samples()[3:], context)
                    predictions = method.predict(model, self.samples()[3:], context)
                    self.assertEqual((16, 16), predictions["sample_3"].shape)
                    self.assertTrue(history)

    def test_reconstructed_slic_rf_fit_and_predict(self) -> None:
        config = MethodConfig(
            id="slic",
            display_name="SLIC-RF",
            family=MethodFamily.CLASSICAL,
            implementation="unused:unused",
            implementation_version="new-test",
            accepted_input_channels=(3,),
            hyperparameters={
                "slic": {"n_segments": 8, "compactness": 5.0, "start_label": 0},
                "classifier": {"n_estimators": 4, "n_jobs": 1},
            },
        )
        with TemporaryDirectory() as temporary:
            context = MethodContext(Path(temporary), 42, True)
            method = SuperpixelRandomForestSegmentation(config)
            model, history = method.fit(self.samples()[:3], self.samples()[3:], context)
            predictions = method.predict(model, self.samples()[3:], context)
            self.assertEqual((16, 16), predictions["sample_3"].shape)
            self.assertIn("training_superpixels", history)


if __name__ == "__main__":
    unittest.main()
