import importlib.util
import unittest

import numpy as np

from bsa_benchmark.methods.deep_learning import (
    binary_bce_dice_loss,
    build_deeplabv3plus,
    build_fcn_resnet50,
    build_pspnet,
    build_sda_unet,
)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class DeepModelSmokeTests(unittest.TestCase):
    def test_all_deep_models_build_and_run_inference(self) -> None:
        import tensorflow as tf

        builders = (
            ("fcn", build_fcn_resnet50, {}),
            ("deeplabv3plus", build_deeplabv3plus, {"aspp_filters": 8}),
            ("pspnet", build_pspnet, {"pyramid_filters": 16}),
            ("bsa_sda_unet", build_sda_unet, {"base_filters": 2}),
        )
        for name, builder, parameters in builders:
            with self.subTest(method=name):
                tf.keras.backend.clear_session()
                model = builder((32, 32, 3), 2, weights=None, **parameters)
                output = np.asarray(model(np.zeros((1, 32, 32, 3), np.float32), training=False))
                self.assertEqual((1, 32, 32), output.shape[:3])
                self.assertEqual(1 if name == "bsa_sda_unet" else 2, output.shape[-1])

    def test_bsa_supports_four_channel_swir_input_and_one_training_step(self) -> None:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        model = build_sda_unet((32, 32, 4), 2, base_filters=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )
        loss = model.train_on_batch(
            np.zeros((1, 32, 32, 4), dtype=np.float32),
            np.zeros((1, 32, 32, 1), dtype=np.float32),
        )
        self.assertTrue(np.isfinite(loss))
        combined_loss = binary_bce_dice_loss(tf.zeros((1, 2, 2, 1)), tf.zeros((1, 2, 2, 1)))
        self.assertTrue(np.isfinite(float(combined_loss)))


if __name__ == "__main__":
    unittest.main()
