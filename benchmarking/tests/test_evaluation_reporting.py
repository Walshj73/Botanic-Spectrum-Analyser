import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bsa_benchmark.evaluation import (
    ConfusionMatrix,
    evaluate_predictions,
    write_cross_validation_summary,
    write_evaluation_report,
)


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_metrics_and_ignore_index(self) -> None:
        target = np.array([[0, 0, 1], [1, 1, 255]])
        prediction = np.array([[0, 1, 1], [0, 1, 1]])
        matrix = ConfusionMatrix(2, ignore_index=255)
        matrix.update(prediction, target)
        metrics = matrix.compute()
        self.assertEqual(((1, 1), (1, 2)), metrics.confusion_matrix)
        self.assertAlmostEqual(3 / 5, metrics.pixel_accuracy)
        self.assertAlmostEqual((1 / 3 + 1 / 2) / 2, metrics.mean_iou)
        self.assertEqual(5, metrics.evaluated_pixels)

    def test_invalid_prediction_class_is_rejected(self) -> None:
        matrix = ConfusionMatrix(2)
        with self.assertRaisesRegex(ValueError, "invalid class"):
            matrix.update(np.array([[2]]), np.array([[1]]))


class ReportingTests(unittest.TestCase):
    def _report(self, fold_id: str = "fold_0"):
        targets = {
            "a": np.array([[0, 1], [0, 1]], dtype=np.uint8),
            "b": np.array([[1, 1], [0, 0]], dtype=np.uint8),
        }
        predictions = {key: value.copy() for key, value in targets.items()}
        return evaluate_predictions(
            predictions,
            targets,
            dataset_id="synthetic",
            method_id="test_method",
            fold_id=fold_id,
            class_names=("background", "foreground"),
        )

    def test_reports_are_written_without_pandas_or_matplotlib(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = write_evaluation_report(self._report(), Path(temporary) / "fold_0")
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(1.0, payload["aggregate"]["mean_iou"])
            with paths["csv"].open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(["a", "b"], [row["sample_id"] for row in rows])
            self.assertIn("Frequency-weighted IoU", paths["markdown"].read_text())
            with self.assertRaises(FileExistsError):
                write_evaluation_report(self._report(), Path(temporary) / "fold_0")

    def test_cross_validation_summary(self) -> None:
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "summary.json"
            write_cross_validation_summary(
                (self._report("fold_0"), self._report("fold_1")), destination
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(2, payload["fold_count"])
            self.assertEqual(1.0, payload["summary"]["mean_iou"]["mean"])


if __name__ == "__main__":
    unittest.main()
