"""Processing primitives for the BSA Mask Creator.

The functions in this module preserve the processing sequence from the
authoritative GUI while remaining independent of Tkinter and TensorFlow.
Model-framework objects are supplied by the caller.
"""

from __future__ import annotations

import datetime as _datetime
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from bsa.utils.resources import load_model_file
from bsa.exporting.analysis import safe_filename_component
from bsa.exporting.output_safety import PlannedOutput, validate_output_plan, write_output
from bsa.segmentation.model_security import inspect_model_file_state
from bsa.utils.image_input import ImageInputError, decode_raster_image
from bsa.utils.read_paths import ResolvedFile, require_unchanged


MODEL_SIZE = (512, 512)


def create_directory(path: os.PathLike[str] | str) -> None:
    """Create *path* when it does not already exist."""

    if not os.path.exists(path):
        os.makedirs(path)


def discover_image_paths(input_dir: os.PathLike[str] | str) -> list[str]:
    """Return the baseline's sorted list of every entry in the input folder.

    The recovered implementation did not filter extensions or directories;
    OpenCV determines whether each discovered entry can be decoded.
    """

    return sorted(os.path.join(input_dir, name) for name in os.listdir(input_dir))


def load_color_image(
    image_path: os.PathLike[str] | str,
    *,
    expected_file: ResolvedFile | None = None,
) -> np.ndarray:
    """Decode one pinned regular raster and return OpenCV-compatible BGR pixels."""

    decoded = decode_raster_image(image_path, expected_file=expected_file)
    try:
        rgb = decoded.convert("RGB")
        try:
            image = np.asarray(rgb, dtype=np.uint8)[:, :, ::-1].copy()
        finally:
            rgb.close()
    finally:
        decoded.close()
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ImageInputError(f"Cannot decode an 8-bit RGB image: {image_path}.")
    return image


def pad_image(image: np.ndarray, new_size: tuple[int, int] = MODEL_SIZE) -> np.ndarray:
    """Centre-pad an image to ``new_size`` using black pixels."""

    old_size = image.shape[:2]
    delta_w = new_size[1] - old_size[1]
    delta_h = new_size[0] - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def unpad_image(image: np.ndarray, old_size: Sequence[int]) -> np.ndarray:
    """Return the centred ``old_size`` region from an image."""

    height, width = old_size
    top = (image.shape[0] - height) // 2
    bottom = top + height
    left = (image.shape[1] - width) // 2
    right = left + width
    return image[top:bottom, left:right]


def resize_image(image: np.ndarray, new_size: tuple[int, int] = MODEL_SIZE) -> np.ndarray:
    """Aspect-resize and centre-pad an image using the recovered algorithm."""

    old_size = image.shape[:2]
    ratio = min(new_size[0] / old_size[0], new_size[1] / old_size[1])
    intermediate_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
    resized_image = cv2.resize(image, intermediate_size, interpolation=cv2.INTER_AREA)

    delta_w = new_size[1] - intermediate_size[0]
    delta_h = new_size[0] - intermediate_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(
        resized_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def prepare_model_input(
    image: np.ndarray,
    model_size: tuple[int, int] = MODEL_SIZE,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize or pad, normalize to 0..1, and add the model batch axis."""

    original_size = image.shape[:2]
    processed_image = image
    if max(original_size) > model_size[0]:
        processed_image = resize_image(image, model_size)
    elif min(original_size) < model_size[0]:
        processed_image = pad_image(image, model_size)

    model_input = processed_image / 255.0
    return np.expand_dims(model_input, axis=0), original_size


def restore_prediction(
    prediction: np.ndarray,
    original_size: Sequence[int],
    model_size: tuple[int, int] = MODEL_SIZE,
) -> np.ndarray:
    """Restore a model-sized prediction using the recovered branch behavior."""

    if max(original_size) > model_size[0]:
        return cv2.resize(
            prediction,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return unpad_image(prediction, original_size)


def predict_mask(model: Any, model_input: np.ndarray, original_size: Sequence[int]) -> np.ndarray:
    """Run one prediction and restore it to the source image dimensions."""

    prediction = model.predict(model_input)[0]
    prediction = np.squeeze(prediction, axis=-1)
    return restore_prediction(prediction, original_size)


def mask_filename(image_path: os.PathLike[str] | str) -> str:
    """Keep the complete source stem, including interior dots."""

    stem = Path(image_path).stem
    return safe_filename_component(stem) + "-PlantMask.png"


def mask_output_path(
    output_dir: os.PathLike[str] | str,
    image_path: os.PathLike[str] | str,
) -> str:
    """Return the output path used by the Mask Creator."""

    return os.path.join(output_dir, mask_filename(image_path))


def mask_output_plan(
    image_paths: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
) -> tuple[PlannedOutput, ...]:
    """Preflight all mask names, including portable spelling collisions."""

    return validate_output_plan(
        PlannedOutput(str(source), "Segmentor mask", Path(mask_output_path(output_dir, source)))
        for source in image_paths
    )


def convert_mask_for_saving(prediction: np.ndarray) -> np.ndarray:
    """Preserve uint8 masks; otherwise scale prediction values by 255."""

    if prediction.dtype != np.uint8:
        return (prediction * 255).astype(np.uint8)
    return prediction


def save_mask(prediction: np.ndarray, save_image_path: os.PathLike[str] | str) -> str:
    """Write a completed PNG without replacing an existing mask."""

    path = Path(save_image_path)
    if not path.name.endswith("-PlantMask.png"):
        path = path.with_name(mask_filename(path))
    def write(staged: Path) -> None:
        if not cv2.imwrite(str(staged), convert_mask_for_saving(prediction)):
            raise OSError(f"Could not save mask {path.name!r}.")
    write_output(PlannedOutput(str(path), "Segmentor mask", path), write)
    return str(path)


def segment_image_file(
    model: Any,
    image_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    should_cancel: Callable[[], bool] | None = None,
    *,
    read_path: os.PathLike[str] | str | None = None,
    expected_file: ResolvedFile | None = None,
) -> str | None:
    """Load, prepare, predict, restore, convert, and save one source image."""

    image = load_color_image(
        read_path if read_path is not None else image_path,
        expected_file=expected_file,
    )
    model_input, original_size = prepare_model_input(image)
    prediction = predict_mask(model, model_input, original_size)
    if should_cancel is not None and should_cancel():
        return None
    return save_mask(prediction, mask_output_path(output_dir, image_path))


def segment_image_batch_cpu(
    model: Any,
    image_paths: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    should_cancel: Callable[[], bool] | None = None,
    *,
    read_paths: Sequence[os.PathLike[str] | str] | None = None,
    expected_files: Sequence[ResolvedFile] | None = None,
) -> list[tuple[os.PathLike[str] | str, str | None, Exception | None]]:
    """Segment one or two CPU images, always predicting with batch size two."""

    if not 1 <= len(image_paths) <= 2:
        raise ValueError("A CPU segmentation batch must contain one or two images.")
    mask_output_plan(image_paths, output_dir)

    results: list[tuple[os.PathLike[str] | str, str | None, Exception | None]] = [
        (path, None, None) for path in image_paths
    ]
    prepared: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    for index, path in enumerate(image_paths):
        if should_cancel is not None and should_cancel():
            return results
        try:
            read_path = read_paths[index] if read_paths is not None else path
            expected_file = expected_files[index] if expected_files is not None else None
            image = load_color_image(read_path, expected_file=expected_file)
            model_input, original_size = prepare_model_input(image)
            prepared.append((index, model_input, original_size))
        except Exception as exc:
            results[index] = (path, None, exc)

    if not prepared:
        return results
    if should_cancel is not None and should_cancel():
        return results

    try:
        model_input = np.concatenate([item[1] for item in prepared], axis=0)
        if len(prepared) == 1:
            model_input = np.concatenate((model_input, model_input), axis=0)
        predictions = model.predict(model_input, verbose=0)
        if len(predictions) < len(prepared):
            raise ValueError("The model returned fewer predictions than source images.")
    except Exception as exc:
        for index, _, _ in prepared:
            results[index] = (image_paths[index], None, exc)
        return results

    for prediction, (index, _, original_size) in zip(predictions, prepared):
        if should_cancel is not None and should_cancel():
            break
        try:
            restored = restore_prediction(np.squeeze(prediction, axis=-1), original_size)
            saved = save_mask(restored, mask_output_path(output_dir, image_paths[index]))
            results[index] = (image_paths[index], saved, None)
        except Exception as exc:
            results[index] = (image_paths[index], None, exc)
    return results


def segment_image_files_cpu(
    model: Any,
    image_paths: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    should_cancel: Callable[[], bool] | None = None,
    *,
    read_paths: Sequence[os.PathLike[str] | str] | None = None,
    expected_files: Sequence[ResolvedFile] | None = None,
):
    """Yield one result per source image while predicting in ordered pairs."""

    mask_output_plan(image_paths, output_dir)

    for start in range(0, len(image_paths), 2):
        if should_cancel is not None and should_cancel():
            return
        yield from segment_image_batch_cpu(
            model,
            image_paths[start : start + 2],
            output_dir,
            should_cancel,
            read_paths=read_paths[start : start + 2] if read_paths is not None else None,
            expected_files=expected_files[start : start + 2] if expected_files is not None else None,
        )


def load_segmentation_model(
    model_path: os.PathLike[str] | str,
    loader: Callable[[str], Any],
    custom_object_scope: Callable[[Mapping[str, Any]], Any],
    custom_objects: Mapping[str, Any],
    framework_seed: Callable[[int], None],
) -> Any:
    """Seed the two RNGs and load a validated model with custom metrics."""

    np.random.seed(42)
    framework_seed(42)
    with custom_object_scope(custom_objects):
        return load_model_file(model_path, loader)


def load_keras_model_for_inference(model_path: str, keras: Any) -> Any:
    """Load a Keras model, accepting legacy ungrouped transpose convolutions.

    Keras 2.10 wrote ``groups: 1`` into Conv2DTranspose H5 configs. Keras 3
    rejects that redundant field. The compatibility class changes only that
    deserialisation detail; the H5 file and its stored weights are untouched.
    """

    inspected = inspect_model_file_state(model_path, keras)
    inspected_path = inspected.path
    require_unchanged(inspected, "Model file")
    if inspected_path.suffix.lower() not in ('.h5', '.hdf5'):
        model = keras.models.load_model(str(inspected_path), compile=False, safe_mode=True)
        require_unchanged(inspected, "Model file")
        return _disable_model_jit_on_gpu(model)

    class LegacyConv2DTranspose(keras.layers.Conv2DTranspose):
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> Any:
            compatible_config = dict(config)
            if 'groups' in compatible_config:
                groups = compatible_config['groups']
                if type(groups) is not int or groups != 1:
                    raise ValueError(
                        'Legacy Conv2DTranspose compatibility supports only groups=1; '
                        f'found {groups!r}.'
                    )
                del compatible_config['groups']
            return super().from_config(compatible_config)

    model = keras.models.load_model(
        str(inspected_path),
        compile=False,
        safe_mode=True,
        custom_objects={'Conv2DTranspose': LegacyConv2DTranspose},
    )
    require_unchanged(inspected, "Model file")
    return _disable_model_jit_on_gpu(model)


def _disable_model_jit_on_gpu(model: Any) -> Any:
    """Avoid Keras GPU auto-JIT's oversized XLA allocations during inference."""

    import tensorflow as tf

    if tf.config.list_physical_devices('GPU'):
        model.jit_compile = False
    return model


def timestamped_output_directory(now: _datetime.datetime | None = None) -> str:
    """Return the exact relative output-directory form used by the GUI."""

    timestamp = now if now is not None else _datetime.datetime.now()
    return "./Masks-" + timestamp.strftime("%Y%m%d_%H%M%S")
