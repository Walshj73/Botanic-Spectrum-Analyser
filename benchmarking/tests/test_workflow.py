import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from bsa_benchmark.config.schemas import (
    AlphaHandling,
    AugmentationConfig,
    ColourMode,
    DatasetConfig,
    Dimensions,
    EvaluationProtocol,
    ExperimentConfig,
    ImageConfig,
    MaskConfig,
    MaskFormat,
    MethodConfig,
    MethodFamily,
    PairingConfig,
    PairingStrategy,
    PreprocessingConfig,
    ProtocolConfig,
    ResolvedExperimentConfig,
    SplitDefaultsConfig,
)
from bsa_benchmark.methods.base import SegmentationMethod
from bsa_benchmark.workflow import run_experiment


class ThresholdMethod(SegmentationMethod):
    def fit(self, train_samples, validation_samples, context):
        self.testcase.assertTrue(train_samples)
        self.testcase.assertTrue(validation_samples)
        return {"threshold": 0.5}, {"loss": (0.1,)}

    def predict(self, model, samples, context):
        return {
            sample.sample_id: (sample.image[..., 0] > model["threshold"]).astype(np.uint8)
            for sample in samples
        }

    def save_model(self, model, destination):
        destination.write_text(str(model), encoding="utf-8")
        return destination


class WorkflowTests(unittest.TestCase):
    def test_synthetic_end_to_end_workflow(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "dataset"
            (data_root / "images").mkdir(parents=True)
            (data_root / "masks").mkdir()
            for index in range(10):
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[:, index % 8 :] = 255
                image = np.repeat(mask[..., None], 3, axis=2)
                Image.fromarray(image).save(data_root / "images" / f"sample_{index}.png")
                Image.fromarray(mask).save(data_root / "masks" / f"sample_{index}.png")

            dimensions = Dimensions(height=8, width=8)
            dataset = DatasetConfig(
                id="synthetic",
                display_name="Synthetic",
                root=data_root,
                images=ImageConfig(
                    directory=Path("images"),
                    dimensions=dimensions,
                    source_channels=3,
                    model_channels=3,
                    colour_mode=ColourMode.RGB,
                    alpha_handling=AlphaHandling.ABSENT,
                ),
                masks=MaskConfig(
                    directory=Path("masks"),
                    dimensions=dimensions,
                    format=MaskFormat.BINARY_GRAYSCALE,
                    classes=("background", "foreground"),
                    threshold=127,
                ),
                pairing=PairingConfig(strategy=PairingStrategy.IDENTICAL_STEM),
                preprocessing=PreprocessingConfig(),
                augmentation=AugmentationConfig(enabled=False),
                splits=SplitDefaultsConfig(group_by_source=False),
            )
            method_config = MethodConfig(
                id="threshold",
                display_name="Threshold",
                family=MethodFamily.CLASSICAL,
                implementation="tests.test_workflow:ThresholdMethod",
                implementation_version="test",
                accepted_input_channels=(3,),
            )
            experiment = ExperimentConfig(
                id="synthetic_development",
                display_name="Synthetic development",
                dataset_config=Path("unused-dataset.yaml"),
                method_config=Path("unused-method.yaml"),
                protocol=ProtocolConfig(
                    type=EvaluationProtocol.DEVELOPMENT,
                    validation_fraction=0.2,
                    test_fraction=0.2,
                ),
                output_root=root / "runs",
            )
            resolved = ResolvedExperimentConfig(
                experiment=experiment, dataset=dataset, method=method_config
            )
            fake = ThresholdMethod(method_config)
            fake.testcase = self
            with patch("bsa_benchmark.workflow.runner.load_method", return_value=fake):
                result = run_experiment(
                    resolved,
                    project_root=root,
                    run_id="synthetic-run",
                )
            self.assertEqual("complete", result.status)
            self.assertTrue((result.run_directory / "splits/development.json").is_file())
            self.assertTrue((result.run_directory / "checkpoints/development.joblib").is_file())
            self.assertTrue((result.run_directory / "metrics/development/metrics.json").is_file())
            self.assertTrue((result.run_directory / "reports/summary.json").is_file())
            self.assertEqual(1.0, result.reports[0].aggregate.mean_iou)


if __name__ == "__main__":
    unittest.main()
