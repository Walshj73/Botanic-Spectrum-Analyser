import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bsa_benchmark.config.loader import find_project_root, resolve_experiment
from bsa_benchmark.core.artifacts import ExperimentDirectory

PROJECT_ROOT = find_project_root(Path(__file__))


class ExperimentDirectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = resolve_experiment(
            PROJECT_ROOT / "configs/experiments/barley_rf_cross_validation.yaml",
            PROJECT_ROOT,
        )

    def test_plan_is_a_pure_dry_run(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            plan = ExperimentDirectory.plan(
                self.resolved,
                output_root=output_root,
                run_id="dry-run",
            )
            self.assertFalse(plan.path.exists())
            self.assertEqual(64, len(plan.fingerprint))

    def test_create_writes_required_metadata_and_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            plan = ExperimentDirectory.plan(
                self.resolved,
                output_root=Path(temporary),
                run_id="fixed-run",
            )
            plan.create(self.resolved, project_root=PROJECT_ROOT, command=("test",))
            self.assertTrue((plan.path / "resolved_config.yaml").is_file())
            self.assertTrue((plan.path / "splits").is_dir())
            self.assertTrue((plan.path / "checkpoints").is_dir())
            self.assertTrue((plan.path / "predictions").is_dir())
            self.assertTrue((plan.path / "metrics").is_dir())
            self.assertTrue((plan.path / "reports").is_dir())
            self.assertTrue((plan.path / "logs").is_dir())
            metadata = json.loads((plan.path / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(42, metadata["seed"])
            self.assertEqual("benchmark", metadata["workflow"])
            self.assertNotIn("git", metadata)
            self.assertEqual(plan.fingerprint, metadata["configuration_fingerprint"])
            with self.assertRaises(FileExistsError):
                plan.create(self.resolved, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
