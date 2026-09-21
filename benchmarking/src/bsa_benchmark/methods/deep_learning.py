"""TensorFlow implementations of the four manuscript deep-learning methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bsa_benchmark.core.interfaces import MethodContext
from bsa_benchmark.data.loading import LoadedSample
from bsa_benchmark.methods.base import OptionalDependencyError, SegmentationMethod


def require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise OptionalDependencyError(
            "deep-learning methods require: pip install 'bsa-benchmark[deep]'"
        ) from exc
    return tf


def binary_bce_dice_loss(y_true: Any, logits: Any) -> Any:
    """Stable manuscript BSA objective: binary cross-entropy plus soft Dice loss."""

    tf = require_tensorflow()
    target = tf.cast(y_true, tf.float32)
    prediction_logits = tf.cast(logits, tf.float32)
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=prediction_logits)
    )
    probabilities = tf.math.sigmoid(prediction_logits)
    axes = tuple(range(1, len(probabilities.shape)))
    intersection = tf.reduce_sum(target * probabilities, axis=axes)
    denominator = tf.reduce_sum(target + probabilities, axis=axes)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return cross_entropy + tf.reduce_mean(1.0 - dice)


def _resnet50_endpoints(
    input_shape: tuple[int, int, int], weights: str | None
) -> tuple[Any, Mapping[str, Any]]:
    tf = require_tensorflow()
    if input_shape[-1] != 3:
        raise ValueError("ResNet-50 baselines require exactly three input channels")
    inputs = tf.keras.Input(input_shape, name="image")
    scaled = tf.keras.layers.Lambda(
        lambda value: tf.keras.applications.resnet50.preprocess_input(value * 255.0),
        name="imagenet_preprocessing",
    )(inputs)
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights=weights, input_tensor=scaled
    )
    return inputs, {
        "c2": backbone.get_layer("conv2_block3_out").output,
        "c3": backbone.get_layer("conv3_block4_out").output,
        "c4": backbone.get_layer("conv4_block6_out").output,
        "c5": backbone.get_layer("conv5_block3_out").output,
    }


def build_fcn_resnet50(
    input_shape: tuple[int, int, int], num_classes: int, *, weights: str | None = None, **_: Any
) -> Any:
    tf = require_tensorflow()
    layers = tf.keras.layers
    inputs, endpoints = _resnet50_endpoints(input_shape, weights)
    score32 = layers.Conv2D(num_classes, 1, name="score32")(endpoints["c5"])
    up32 = layers.Conv2DTranspose(num_classes, 4, strides=2, padding="same")(score32)
    score16 = layers.Conv2D(num_classes, 1, name="score16")(endpoints["c4"])
    up16 = layers.Conv2DTranspose(num_classes, 4, strides=2, padding="same")(
        layers.Add()([up32, score16])
    )
    score8 = layers.Conv2D(num_classes, 1, name="score8")(endpoints["c3"])
    logits = layers.Conv2DTranspose(
        num_classes, 16, strides=8, padding="same", name="logits", dtype="float32"
    )(layers.Add()([up16, score8]))
    return tf.keras.Model(inputs, logits, name="fcn8s_resnet50")


def deeplab_rates(image_size: int) -> tuple[int, ...]:
    feature_size = max(image_size // 16, 1)
    canonical = tuple(rate for rate in (6, 12, 18) if rate < feature_size)
    if len(canonical) >= 2:
        return canonical
    compact = tuple(rate for rate in (2, 4, 6) if rate < feature_size)
    return compact or (1,)


def _conv_bn_relu(value: Any, filters: int, kernel: int = 3, dilation: int = 1) -> Any:
    tf = require_tensorflow()
    value = tf.keras.layers.Conv2D(
        filters,
        kernel,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
    )(value)
    value = tf.keras.layers.BatchNormalization()(value)
    return tf.keras.layers.Activation("relu")(value)


def build_deeplabv3plus(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    weights: str | None = None,
    aspp_filters: int = 256,
    **_: Any,
) -> Any:
    tf = require_tensorflow()
    layers = tf.keras.layers
    inputs, endpoints = _resnet50_endpoints(input_shape, weights)
    low, high = endpoints["c2"], endpoints["c4"]
    branches = [_conv_bn_relu(high, aspp_filters, 1)]
    branches.extend(
        _conv_bn_relu(high, aspp_filters, 3, rate) for rate in deeplab_rates(input_shape[0])
    )
    pooled = layers.GlobalAveragePooling2D(keepdims=True)(high)
    # No batch normalization on the 1x1 pooled branch: singleton batches are valid.
    pooled = layers.Conv2D(aspp_filters, 1, activation="relu")(pooled)
    pooled = layers.Resizing(high.shape[1], high.shape[2], interpolation="bilinear")(pooled)
    branches.append(pooled)
    value = _conv_bn_relu(layers.Concatenate()(branches), aspp_filters, 1)
    value = layers.Resizing(low.shape[1], low.shape[2], interpolation="bilinear")(value)
    low = _conv_bn_relu(low, 48, 1)
    value = _conv_bn_relu(layers.Concatenate()([value, low]), aspp_filters)
    value = _conv_bn_relu(value, aspp_filters)
    value = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(value)
    logits = layers.Conv2D(num_classes, 1, name="logits", dtype="float32")(value)
    return tf.keras.Model(inputs, logits, name="deeplabv3plus_resnet50")


def build_pspnet(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    weights: str | None = None,
    pyramid_bins: Sequence[int] = (1, 2, 3, 6),
    pyramid_filters: int = 512,
    **_: Any,
) -> Any:
    tf = require_tensorflow()
    layers = tf.keras.layers
    inputs, endpoints = _resnet50_endpoints(input_shape, weights)
    c2, c3, c4 = endpoints["c2"], endpoints["c3"], endpoints["c4"]
    height, width = int(c4.shape[1]), int(c4.shape[2])
    branches = [c4]
    for bin_size in pyramid_bins:
        pool_height, pool_width = max(height // bin_size, 1), max(width // bin_size, 1)
        pooled = layers.AveragePooling2D(
            pool_size=(pool_height, pool_width), strides=(pool_height, pool_width)
        )(c4)
        pooled = _conv_bn_relu(pooled, pyramid_filters // len(pyramid_bins), 1)
        branches.append(layers.Resizing(height, width, interpolation="bilinear")(pooled))
    value = _conv_bn_relu(layers.Concatenate()(branches), pyramid_filters)
    value = layers.Dropout(0.1)(value)
    value = layers.Resizing(c3.shape[1], c3.shape[2], interpolation="bilinear")(value)
    value = _conv_bn_relu(layers.Concatenate()([value, _conv_bn_relu(c3, 128, 1)]), 256)
    value = layers.Resizing(c2.shape[1], c2.shape[2], interpolation="bilinear")(value)
    value = _conv_bn_relu(layers.Concatenate()([value, _conv_bn_relu(c2, 48, 1)]), 256)
    value = _conv_bn_relu(value, 256)
    value = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(value)
    logits = layers.Conv2D(num_classes, 1, name="logits", dtype="float32")(value)
    return tf.keras.Model(inputs, logits, name="pspnet_resnet50")


def build_sda_unet(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    base_filters: int = 24,
    dropout_rate: float = 0.2,
    stem: bool = True,
    separable: bool = False,
    **_: Any,
) -> Any:
    """Migrated stable SDA-UNet/BSA architecture with logits output."""

    tf = require_tensorflow()
    layers = tf.keras.layers
    if stem and (input_shape[0] % 32 or input_shape[1] % 32):
        raise ValueError("SDA-UNet input height and width must be divisible by 32")

    def convolution(value: Any, filters: int, kernel: int = 3, dilation: int = 1) -> Any:
        layer = layers.SeparableConv2D if separable and kernel > 1 else layers.Conv2D
        kwargs: dict[str, Any] = {
            "filters": filters,
            "kernel_size": kernel,
            "padding": "same",
            "dilation_rate": dilation,
            "use_bias": False,
        }
        value = layer(**kwargs)(value)
        return layers.Activation("relu")(layers.BatchNormalization()(value))

    def residual(value: Any, filters: int, dropout: float = 0.0) -> Any:
        shortcut = value
        if int(value.shape[-1]) != filters:
            shortcut = layers.BatchNormalization()(
                layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
            )
        value = convolution(value, filters)
        value = convolution(value, filters)
        if dropout:
            value = layers.Dropout(dropout)(value)
        return layers.Activation("relu")(layers.Add()([value, shortcut]))

    def squeeze_excite(value: Any) -> Any:
        channels = int(value.shape[-1])
        excitation = layers.GlobalAveragePooling2D(keepdims=True)(value)
        excitation = layers.Dense(max(channels // 8, 1), activation="relu")(excitation)
        excitation = layers.Dense(channels, activation="sigmoid")(excitation)
        return layers.Multiply()([value, excitation])

    def attention(skip: Any, gating: Any, filters: int) -> Any:
        theta = layers.Conv2D(filters, 1, padding="same", use_bias=False)(skip)
        phi = layers.Conv2D(filters, 1, padding="same")(gating)
        score = layers.Activation("relu")(layers.Add()([theta, phi]))
        score = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(score)
        return layers.Multiply()([skip, score])

    def encoder(value: Any, filters: int, dropout: float = 0.0) -> tuple[Any, Any]:
        skip = residual(value, filters, dropout)
        return skip, layers.MaxPool2D((2, 2))(skip)

    def decoder(value: Any, skip: Any, filters: int, dropout: float = 0.0) -> Any:
        value = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(value)
        skip = attention(skip, value, max(filters // 2, 1))
        return squeeze_excite(residual(layers.Concatenate()([value, skip]), filters, dropout))

    inputs = tf.keras.Input(input_shape, name="image")
    full_resolution = None
    value = inputs
    if stem:
        full_resolution = convolution(value, base_filters)
        value = layers.Conv2D(base_filters, 3, strides=2, padding="same", use_bias=False)(
            full_resolution
        )
        value = layers.Activation("relu")(layers.BatchNormalization()(value))
    skip1, value = encoder(value, base_filters)
    skip2, value = encoder(value, base_filters * 2)
    skip3, value = encoder(value, base_filters * 4, dropout_rate)
    skip4, value = encoder(value, base_filters * 8, dropout_rate)

    height, width = int(value.shape[1]), int(value.shape[2])
    branches = [convolution(value, base_filters * 8, 1)]
    branches.extend(convolution(value, base_filters * 8, 3, rate) for rate in (2, 4))
    pooled = layers.GlobalAveragePooling2D(keepdims=True)(value)
    pooled = layers.Conv2D(base_filters * 8, 1, activation="relu")(pooled)
    branches.append(layers.Resizing(height, width, interpolation="bilinear")(pooled))
    value = layers.Dropout(0.3)(convolution(layers.Concatenate()(branches), base_filters * 8, 1))
    value = decoder(value, skip4, base_filters * 8, dropout_rate)
    value = decoder(value, skip3, base_filters * 4, dropout_rate)
    value = decoder(value, skip2, base_filters * 2)
    value = decoder(value, skip1, base_filters)
    if stem:
        value = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(value)
        value = residual(layers.Concatenate()([value, full_resolution]), base_filters)
    output_channels = 1 if num_classes == 2 else num_classes
    logits = layers.Conv2D(
        output_channels, 1, padding="same", name="segmentation_logits", dtype="float32"
    )(value)
    return tf.keras.Model(inputs, logits, name="SDA-UNet")


class TensorFlowSegmentation(SegmentationMethod):
    builder: Callable[..., Any]
    binary_logits = False

    def _dataset(
        self, samples: Sequence[LoadedSample], batch_size: int, shuffle: bool, seed: int
    ) -> Any:
        tf = require_tensorflow()
        if not samples:
            raise ValueError("deep-learning split must not be empty")
        images = np.stack([sample.image for sample in samples]).astype(np.float32)
        masks = np.stack([sample.mask for sample in samples])
        if self.binary_logits:
            masks = masks[..., None].astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        if shuffle:
            dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(
        self,
        train_samples: Sequence[LoadedSample],
        validation_samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> tuple[Any, Mapping[str, Sequence[float]]]:
        tf = require_tensorflow()
        training = self.config.training
        hyperparameters = dict(self.config.hyperparameters)
        num_classes = int(training.get("num_classes", 2))
        input_shape = tuple(int(value) for value in train_samples[0].image.shape)
        model = self.builder(input_shape, num_classes, **hyperparameters)
        learning_rate = float(training.get("learning_rate", 6e-4))
        if self.binary_logits:
            loss_name = str(training.get("loss", "bce_dice")).casefold()
            if loss_name == "bce_dice":
                loss: Any = binary_bce_dice_loss
            elif loss_name == "binary_crossentropy":
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                raise ValueError(f"unsupported binary training loss: {loss_name}")
            metrics: list[Any] = [
                tf.keras.metrics.BinaryIoU(target_class_ids=(0, 1), threshold=0.0)
            ]
            monitor = "val_binary_io_u"
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy")]
            monitor = "val_loss"
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=int(training.get("patience", 12)),
                mode="min" if monitor == "val_loss" else "max",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, int(training.get("patience", 12)) // 3),
            ),
        ]
        history = model.fit(
            self._dataset(train_samples, int(training.get("batch_size", 4)), True, context.seed),
            validation_data=self._dataset(
                validation_samples, int(training.get("batch_size", 4)), False, context.seed
            ),
            epochs=int(training.get("epochs", 150)),
            callbacks=callbacks,
            verbose=int(training.get("verbose", 1)),
        )
        return model, {
            key: tuple(float(value) for value in values) for key, values in history.history.items()
        }

    def predict(
        self,
        model: Any,
        samples: Sequence[LoadedSample],
        context: MethodContext,
    ) -> Mapping[str, np.ndarray]:
        del context
        predictions: dict[str, np.ndarray] = {}
        batch_size = int(self.config.inference.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("inference.batch_size must be at least one")
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = np.stack([sample.image for sample in batch]).astype(np.float32)
            batch_logits = np.asarray(model.predict(images, verbose=0))
            for sample, logits in zip(batch, batch_logits, strict=True):
                if self.binary_logits:
                    predicted = (logits[..., 0] >= 0.0).astype(np.uint8)
                else:
                    predicted = np.argmax(logits, axis=-1).astype(np.uint8)
                predictions[sample.sample_id] = predicted
        return predictions

    def save_model(self, model: Any, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        model.save(destination)
        return destination


class FCNResNet50Segmentation(TensorFlowSegmentation):
    builder = staticmethod(build_fcn_resnet50)


class DeepLabV3PlusSegmentation(TensorFlowSegmentation):
    builder = staticmethod(build_deeplabv3plus)


class PSPNetSegmentation(TensorFlowSegmentation):
    builder = staticmethod(build_pspnet)


class SDAUNetSegmentation(TensorFlowSegmentation):
    builder = staticmethod(build_sda_unet)
    binary_logits = True
