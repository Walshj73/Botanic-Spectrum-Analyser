import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from bsa_benchmark.cli import main
from bsa_benchmark.config.loader import find_project_root

PROJECT_ROOT = find_project_root(Path(__file__))


class CommandLineTests(unittest.TestCase):
    def test_validate_experiment_short_name(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = main(["validate", "experiment", "barley_rf_cross_validation"])
        self.assertEqual(0, result)
        payload = json.loads(stdout.getvalue())
        self.assertEqual("barley_rf_cross_validation", payload["id"])
        self.assertEqual("cross_validation", payload["protocol"])

    def test_validate_dataset_short_name_and_paths(self) -> None:
        if not (PROJECT_ROOT / "data/raw/barley/images").is_dir():
            self.skipTest("raw integration-test data is not present")
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = main(["validate", "dataset", "barley", "--check-paths"])
        self.assertEqual(0, result)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(100, payload["paths"]["paired_count"])

    def test_dry_run_accepts_explicit_selection_without_creating_output(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "planned"
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "dry-run",
                        "--experiment",
                        "barley_rf_cross_validation",
                        "--dataset",
                        "barley",
                        "--method",
                        "random_forest",
                        "--output-root",
                        str(output_root),
                    ]
                )
            self.assertEqual(0, result)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["creates_files"])
            self.assertFalse(output_root.exists())

    def test_invalid_configuration_returns_nonzero(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = main(["validate", "dataset", "does_not_exist"])
        self.assertEqual(2, result)
        self.assertIn("error:", stderr.getvalue())

    def test_make_splits_writes_non_overwriting_reproducible_manifests(self) -> None:
        if not (PROJECT_ROOT / "data/raw/barley/images").is_dir():
            self.skipTest("raw integration-test data is not present")
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "splits"
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "make-splits",
                        "--experiment",
                        "barley_rf_cross_validation",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(0, result)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(5, payload["manifest_count"])
            self.assertTrue((output / "fold_0.json").is_file())
            self.assertTrue((output / "split_index.json").is_file())

            stderr = StringIO()
            with redirect_stderr(stderr):
                repeated = main(
                    [
                        "make-splits",
                        "--experiment",
                        "barley_rf_cross_validation",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(2, repeated)


if __name__ == "__main__":
    unittest.main()
