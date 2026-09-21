import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import numpy as np
from PIL import Image

from bsa_benchmark.config.loader import find_project_root, load_dataset_config
from bsa_benchmark.core.interfaces import SampleReference
from bsa_benchmark.data import (
    AugmentationPipeline,
    DatasetValidationError,
    ImageMaskLoader,
    discover_dataset,
    validate_dataset_contents,
)

PROJECT_ROOT = find_project_root(Path(__file__))


@unittest.skipUnless(
    (PROJECT_ROOT / "data/raw/barley/images").is_dir(),
    "raw integration-test data is not present",
)
class RealDatasetLoadingTests(unittest.TestCase):
    expected: ClassVar[dict[str, tuple[int, tuple[int, ...]]]] = {
        "barley": (100, (512, 512, 3)),
        "barley_swir": (90, (512, 512, 3)),
        "dicot": (50, (512, 512, 3)),
        "wheat": (30, (512, 512, 3)),
    }

    def test_all_configured_datasets_decode_and_normalize_masks(self) -> None:
        for dataset_id, (count, image_shape) in self.expected.items():
            with self.subTest(dataset=dataset_id):
                config = load_dataset_config(
                    PROJECT_ROOT / "configs/datasets" / f"{dataset_id}.yaml",
                    PROJECT_ROOT,
                )
                index = discover_dataset(config)
                report = validate_dataset_contents(config, index)
                self.assertEqual(count, report.sample_count)
                self.assertEqual((image_shape,), report.image_shapes)
                self.assertEqual(((512, 512),), report.mask_shapes)
                self.assertEqual((0, 1), report.mask_values)
                self.assertEqual(64, len(report.fingerprint))

    def test_wheat_alpha_is_dropped_but_source_file_remains_rgba(self) -> None:
        config = load_dataset_config(PROJECT_ROOT / "configs/datasets/wheat.yaml", PROJECT_ROOT)
        index = discover_dataset(config)
        loaded = ImageMaskLoader(config).load(index.samples[0])
        self.assertEqual(3, loaded.image.shape[-1])
        with Image.open(index.samples[0].image_path) as source:
            self.assertEqual("RGBA", source.mode)


class MaskValidationTests(unittest.TestCase):
    def test_swir_auxiliary_transforms_are_available_and_bounded(self) -> None:
        base = load_dataset_config(
            PROJECT_ROOT / "configs/datasets/barley_swir_nexm.yaml", PROJECT_ROOT
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            image[..., 0] = 255
            image[..., 2] = 255
            mask = np.zeros((512, 512), dtype=np.uint8)
            Image.fromarray(image).save(root / "images/sample.png")
            Image.fromarray(mask).save(root / "masks/sample.png")
            config = base.model_copy(
                update={
                    "root": root,
                    "splits": base.splits.model_copy(update={"source_group_regex": None}),
                }
            )
            loaded = ImageMaskLoader(config).load(discover_dataset(config).samples[0])
            self.assertEqual((512, 512, 4), loaded.image.shape)
            self.assertGreaterEqual(float(loaded.image[..., 3].min()), 0.0)
            self.assertLessEqual(float(loaded.image[..., 3].max()), 1.0)

    def test_loader_forbids_augmentation_on_validation_and_test_roles(self) -> None:
        config = load_dataset_config(PROJECT_ROOT / "configs/datasets/barley.yaml", PROJECT_ROOT)
        loader = ImageMaskLoader(config)
        augmentation = AugmentationPipeline(config.augmentation, seed=42)
        missing = SampleReference(
            sample_id="unused",
            image_path=Path("missing-image.png"),
            mask_path=Path("missing-mask.png"),
            group_id="unused",
        )
        for role in ("validation", "test"):
            with (
                self.subTest(role=role),
                self.assertRaisesRegex(ValueError, "augmentation is forbidden"),
            ):
                loader.load_for_role(missing, role=role, augmentation=augmentation)

    def test_binary_rgb_mask_rejects_disagreeing_channels(self) -> None:
        base = load_dataset_config(PROJECT_ROOT / "configs/datasets/barley.yaml", PROJECT_ROOT)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512, 3), dtype=np.uint8)
            mask[0, 0, 1] = 255
            Image.fromarray(image).save(root / "images/sample.png")
            Image.fromarray(mask).save(root / "masks/sample.png")
            config = base.model_copy(
                update={
                    "root": root,
                    "splits": base.splits.model_copy(update={"source_group_regex": None}),
                }
            )
            index = discover_dataset(config)
            with self.assertRaisesRegex(DatasetValidationError, "channels differ"):
                ImageMaskLoader(config).load(index.samples[0])

    def test_binary_grayscale_mask_is_thresholded_to_zero_and_one(self) -> None:
        base = load_dataset_config(PROJECT_ROOT / "configs/datasets/barley_swir.yaml", PROJECT_ROOT)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[0, 0] = 255
            Image.fromarray(image).save(root / "images/sample.png")
            Image.fromarray(mask).save(root / "masks/sample.png")
            config = base.model_copy(
                update={
                    "root": root,
                    "splits": base.splits.model_copy(update={"source_group_regex": None}),
                }
            )
            loaded = ImageMaskLoader(config).load(discover_dataset(config).samples[0])
            self.assertEqual(np.uint8, loaded.mask.dtype)
            self.assertEqual({0, 1}, set(np.unique(loaded.mask).tolist()))
            self.assertEqual(1, int(loaded.mask[0, 0]))


if __name__ == "__main__":
    unittest.main()
