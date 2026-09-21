import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bsa_benchmark.config.loader import find_project_root, load_dataset_config
from bsa_benchmark.core.interfaces import SampleReference
from bsa_benchmark.data import discover_dataset
from bsa_benchmark.data.index import DatasetIndex, DatasetValidationError
from bsa_benchmark.data.splitting import (
    SplitManifest,
    assert_no_source_leakage,
    generate_cross_validation,
    generate_development_split,
    read_split_manifest,
    validate_manifest_against_index,
    write_split_manifest,
)


def grouped_index(groups: int = 12, variants: int = 2) -> DatasetIndex:
    samples = []
    for group_index in range(groups):
        for variant in range(variants):
            sample_id = f"source_{group_index:02d}_variant_{variant}"
            samples.append(
                SampleReference(
                    sample_id=sample_id,
                    image_path=Path(f"{sample_id}.png"),
                    mask_path=Path(f"{sample_id}_mask.png"),
                    group_id=f"source_{group_index:02d}",
                )
            )
    return DatasetIndex("synthetic", Path("."), tuple(samples), "a" * 64)


class SplitGenerationTests(unittest.TestCase):
    def test_explicit_source_overlap_is_rejected(self) -> None:
        manifest = SplitManifest(
            dataset_id="synthetic",
            dataset_fingerprint="a" * 64,
            protocol="development",
            seed=42,
            fold_id="bad",
            train_ids=("source_a_variant_0",),
            validation_ids=("source_a_variant_1",),
            test_ids=(),
            source_groups={
                "source_a_variant_0": "source_a",
                "source_a_variant_1": "source_a",
            },
        )
        with self.assertRaisesRegex(DatasetValidationError, "source-level leakage"):
            assert_no_source_leakage(manifest)

    def test_five_fold_generation_for_every_configured_dataset(self) -> None:
        project_root = find_project_root(Path(__file__))
        if not (project_root / "data/raw/barley/images").is_dir():
            self.skipTest("raw integration-test data is not present")
        for dataset_id in ("barley", "barley_swir", "dicot", "wheat"):
            with self.subTest(dataset=dataset_id):
                config = load_dataset_config(
                    project_root / "configs/datasets" / f"{dataset_id}.yaml",
                    project_root,
                )
                index = discover_dataset(config)
                if dataset_id in {"barley", "barley_swir"}:
                    self.assertLess(
                        len({sample.group_id for sample in index.samples}),
                        len(index.samples),
                    )
                manifests = generate_cross_validation(
                    index,
                    folds=config.splits.folds,
                    validation_fraction=config.splits.validation_fraction,
                    seed=config.splits.seed,
                )
                self.assertEqual(5, len(manifests))
                tested = [sample for manifest in manifests for sample in manifest.test_ids]
                self.assertEqual(len(index.samples), len(tested))
                self.assertEqual(len(tested), len(set(tested)))
                for manifest in manifests:
                    assert_no_source_leakage(manifest)

    def test_development_split_is_reproducible_and_group_safe(self) -> None:
        index = grouped_index()
        first = generate_development_split(
            index, validation_fraction=0.2, test_fraction=0.2, seed=42
        )
        second = generate_development_split(
            index, validation_fraction=0.2, test_fraction=0.2, seed=42
        )
        self.assertEqual(first, second)
        assert_no_source_leakage(first)
        self.assertEqual(24, len(first.train_ids + first.validation_ids + first.test_ids))

    def test_cross_validation_tests_every_source_once_without_leakage(self) -> None:
        index = grouped_index(groups=15)
        manifests = generate_cross_validation(index, folds=5, validation_fraction=0.2, seed=7)
        tested = []
        for manifest in manifests:
            assert_no_source_leakage(manifest)
            tested.extend(manifest.test_ids)
        self.assertEqual(sorted(sample.sample_id for sample in index.samples), sorted(tested))

    def test_manifest_round_trip_and_non_overwrite(self) -> None:
        index = grouped_index()
        manifest = generate_development_split(
            index, validation_fraction=0.2, test_fraction=0.2, seed=42
        )
        validate_manifest_against_index(manifest, index, require_complete=True)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            write_split_manifest(manifest, path)
            self.assertEqual(manifest, read_split_manifest(path))
            with self.assertRaises(FileExistsError):
                write_split_manifest(manifest, path)

            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["train_ids"] = raw["train_ids"][1:]
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "fingerprint mismatch"):
                read_split_manifest(path)


if __name__ == "__main__":
    unittest.main()
