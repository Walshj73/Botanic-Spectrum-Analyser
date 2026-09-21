import unittest
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from bsa_benchmark.config.loader import (
    check_dataset_paths,
    find_project_root,
    load_dataset_config,
    resolve_experiment,
)
from bsa_benchmark.config.schemas import (
    EvaluationProtocol,
    PairingConfig,
    PairingStrategy,
    ProtocolConfig,
)

PROJECT_ROOT = find_project_root(Path(__file__))


class DatasetConfigurationTests(unittest.TestCase):
    expected_counts: ClassVar[dict[str, int]] = {
        "barley": 100,
        "barley_swir": 90,
        "dicot": 50,
        "wheat": 30,
    }

    def test_all_dataset_configs_validate_and_pair_existing_data(self) -> None:
        for dataset_id, expected_count in self.expected_counts.items():
            with self.subTest(dataset=dataset_id):
                path = PROJECT_ROOT / "configs" / "datasets" / f"{dataset_id}.yaml"
                config = load_dataset_config(path, PROJECT_ROOT)
                if not (config.root / config.images.directory).is_dir():
                    self.assertEqual(dataset_id, config.id)
                    continue
                report = check_dataset_paths(config)
                self.assertTrue(report.valid)
                self.assertEqual(expected_count, report.paired_count)

    def test_token_pairing_requires_both_tokens(self) -> None:
        with self.assertRaises(ValidationError):
            PairingConfig(strategy=PairingStrategy.TOKEN_REPLACEMENT, image_token="_rgb")

    def test_identical_pairing_rejects_unused_tokens(self) -> None:
        with self.assertRaises(ValidationError):
            PairingConfig(
                strategy=PairingStrategy.IDENTICAL_STEM,
                image_token="_rgb",
                mask_token="_label",
            )


class ExperimentConfigurationTests(unittest.TestCase):
    def test_cross_validation_experiment_composes(self) -> None:
        resolved = resolve_experiment(
            PROJECT_ROOT / "configs/experiments/barley_rf_cross_validation.yaml",
            PROJECT_ROOT,
        )
        self.assertEqual("barley", resolved.dataset.id)
        self.assertEqual("random_forest", resolved.method.id)
        self.assertEqual(EvaluationProtocol.CROSS_VALIDATION, resolved.experiment.protocol.type)
        self.assertEqual(5, resolved.experiment.protocol.folds)
        self.assertEqual(200, resolved.method.hyperparameters["n_estimators"])

    def test_cross_validation_requires_fold_count(self) -> None:
        with self.assertRaises(ValidationError):
            ProtocolConfig(
                type=EvaluationProtocol.CROSS_VALIDATION,
                validation_fraction=0.1,
            )

    def test_reconstructed_slic_experiment_is_standard_cross_validation(self) -> None:
        resolved = resolve_experiment(
            PROJECT_ROOT / "configs/experiments/wheat_slic_cross_validation.yaml",
            PROJECT_ROOT,
        )
        protocol = resolved.experiment.protocol
        self.assertEqual(EvaluationProtocol.CROSS_VALIDATION, protocol.type)
        self.assertEqual(5, protocol.folds)
        self.assertEqual("slic_rf_v1", resolved.method.id)


if __name__ == "__main__":
    unittest.main()
