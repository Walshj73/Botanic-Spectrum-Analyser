import unittest

import numpy as np

from bsa_benchmark.config.schemas import AugmentationConfig
from bsa_benchmark.data.augmentation import AugmentationPipeline


class AugmentationTests(unittest.TestCase):
    def test_geometric_transform_is_identical_for_image_and_mask(self) -> None:
        mask = np.arange(20, dtype=np.uint8).reshape(4, 5)
        image = np.repeat(mask[..., None], 3, axis=2)
        config = AugmentationConfig(
            enabled=True,
            horizontal_flip_probability=1.0,
            vertical_flip_probability=1.0,
            rotations_degrees=(90,),
            brightness_delta=0.0,
        )
        transformed_image, transformed_mask, decision = AugmentationPipeline(config, seed=19).apply(
            image, mask, sample_id="source-a", epoch=2
        )
        np.testing.assert_array_equal(transformed_image[..., 0], transformed_mask)
        self.assertTrue(decision.horizontal_flip)
        self.assertTrue(decision.vertical_flip)
        self.assertEqual(90, decision.rotation_degrees)

    def test_decisions_and_outputs_are_reproducible_per_sample_and_epoch(self) -> None:
        image = np.linspace(0, 1, 48, dtype=np.float32).reshape(4, 4, 3)
        mask = np.arange(16, dtype=np.uint8).reshape(4, 4) % 2
        config = AugmentationConfig(
            enabled=True,
            rotations_degrees=(0, 90, 180, 270),
            brightness_delta=0.2,
            contrast_range=(0.8, 1.2),
        )
        pipeline = AugmentationPipeline(config, seed=42)
        first = pipeline.apply(image, mask, sample_id="sample", epoch=3)
        second = pipeline.apply(image, mask, sample_id="sample", epoch=3)
        self.assertEqual(first[2], second[2])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertGreaterEqual(float(first[0].min()), 0.0)
        self.assertLessEqual(float(first[0].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
