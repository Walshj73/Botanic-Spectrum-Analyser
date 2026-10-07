"""Inspect serialized segmentation architecture before Keras deserializes it.

This rejects executable or unknown configuration objects. It does not make
TensorFlow execution of an untrusted graph a sandbox; model provenance remains
important. Inspection uses only JSON and HDF5/ZIP metadata, never Keras object
deserialization or imports named by the model file.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import zipfile

import h5py

from bsa.utils.resources import ModelFileError
from bsa.utils.read_paths import FileIdentity, ReadPathError, ResolvedFile, open_regular_file


_MAX_CONFIG_BYTES = 32 * 1024 * 1024
_MAX_CONFIG_NODES = 100_000
_MAX_CONFIG_DEPTH = 100
_SAFE_LAYER_MODULE_PARTS = (
    ".layers.convolutional.",
    ".layers.pooling.",
    ".layers.normalization.",
    ".layers.activations.",
    ".layers.merging.",
    ".layers.reshaping.",
    ".layers.regularization.",
    ".layers.attention.",
    ".layers.rnn.",
)
_SAFE_CORE_LAYERS = {
    "Dense", "EinsumDense", "Embedding", "Identity", "InputLayer", "Masking",
}
_SAFE_PREPROCESSING_LAYERS = {"Normalization", "Rescaling", "Resizing"}
_INERT_SERIALIZED_VALUES = {
    "__keras_tensor__", "__tensor__", "__numpy__", "__bytes__",
    "__slice__", "__ellipsis__",
}
_UNSAFE_CLASSES = {
    "Lambda", "TFSMLayer", "JaxLayer", "FlaxLayer", "TorchModuleWrapper",
}
_EXPLANATION = (
    "Model files can contain executable or unsafe deserialization behavior. "
    "Open only trusted, compatible BSA/Keras segmentation models."
)


class UnsafeModelConfiguration(ModelFileError):
    """A serialized component is unsafe or unsupported by BSA's loader."""

    def __init__(self, reason: str):
        super().__init__(f"Model rejected before loading: {reason}. {_EXPLANATION}")


class InvalidModelStructure(ModelFileError):
    """The selected file is not a readable Keras model archive."""

    def __init__(self):
        super().__init__("Model file is not a readable Keras H5/HDF5 or .keras model archive.")


def _object_names(namespace: object, base_type: type) -> dict[str, type]:
    """Use installed Keras's public built-ins, without resolving file-supplied names."""

    return {
        name: value
        for name, value in vars(namespace).items()
        if inspect.isclass(value) and issubclass(value, base_type)
        and name not in {"Layer", "Initializer", "Regularizer", "Constraint"}
        and value.__module__.startswith("keras.")
    }


def _simple_builtin_names(namespace: object, module_prefix: str) -> set[str]:
    return {
        name for name, value in vars(namespace).items()
        if not name.startswith("_") and name not in {"get", "serialize", "deserialize"}
        and callable(value)
        and getattr(value, "__module__", "").startswith(module_prefix)
    }


def _supported_objects(keras: object) -> tuple[dict[str, set[str | None]], dict[str, set[str]]]:
    layer_objects = _object_names(keras.layers, keras.layers.Layer)
    supported: dict[str, set[str | None]] = {}
    for name, layer in layer_objects.items():
        module = layer.__module__
        if (
            any(part in module for part in _SAFE_LAYER_MODULE_PARTS)
            or name in _SAFE_CORE_LAYERS
            or name in _SAFE_PREPROCESSING_LAYERS
        ) and name not in _UNSAFE_CLASSES:
            supported[name] = {None, "keras.layers", module}

    for namespace_name, base_name in (
        ("initializers", "Initializer"),
        ("regularizers", "Regularizer"),
        ("constraints", "Constraint"),
    ):
        namespace = getattr(keras, namespace_name)
        for name, obj in _object_names(namespace, getattr(namespace, base_name)).items():
            supported.setdefault(name, set()).update(
                {None, f"keras.{namespace_name}", obj.__module__}
            )

    supported.update({
        "Functional": {None, "keras.src.models.functional"},
        "Sequential": {None, "keras", keras.models.Sequential.__module__},
        "DTypePolicy": {None, "keras", keras.DTypePolicy.__module__},
    })
    for name in _INERT_SERIALIZED_VALUES:
        supported[name] = {None}

    identifiers = {
        "activation": _simple_builtin_names(keras.activations, "keras.src.activations"),
        "initializer": _simple_builtin_names(keras.initializers, "keras.src.initializers"),
        "regularizer": _simple_builtin_names(keras.regularizers, "keras.src.regularizers"),
        "constraint": _simple_builtin_names(keras.constraints, "keras.src.constraints"),
    }
    return supported, identifiers


def inspect_model_configuration(config: object, keras: object) -> None:
    """Reject unknown/code-backed objects and suspicious references in a config."""

    if not isinstance(config, dict) or config.get("class_name") not in {"Functional", "Sequential"}:
        raise UnsafeModelConfiguration("unsupported model architecture")
    supported, identifiers = _supported_objects(keras)
    stack: list[tuple[object, int]] = [(config, 0)]
    visited = 0
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > _MAX_CONFIG_NODES or depth > _MAX_CONFIG_DEPTH:
            raise UnsafeModelConfiguration("model configuration is too complex")
        if isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
            continue
        if not isinstance(value, dict):
            continue

        class_name = value.get("class_name")
        if "class_name" in value:
            if not isinstance(class_name, str) or class_name in _UNSAFE_CLASSES:
                raise UnsafeModelConfiguration("unsafe serialized layer or function")
            allowed_modules = supported.get(class_name)
            if allowed_modules is None:
                raise UnsafeModelConfiguration("unknown or custom serialized component")
            if value.get("module") not in allowed_modules:
                raise UnsafeModelConfiguration("arbitrary module reference")
            registered_name = value.get("registered_name")
            if registered_name is not None and registered_name != class_name:
                raise UnsafeModelConfiguration("custom object registration")
            if class_name in {"Functional", "Sequential"}:
                nested = value.get("config")
                if not isinstance(nested, dict) or not isinstance(nested.get("layers"), list):
                    raise UnsafeModelConfiguration("invalid built-in model structure")
        elif "module" in value or "registered_name" in value:
            raise UnsafeModelConfiguration("arbitrary module or custom object reference")

        for key, item in value.items():
            if key == "compile_config":
                # compile=False does not deserialize training-only objects.
                continue
            if key in {"function", "function_type", "python_function"}:
                raise UnsafeModelConfiguration("serialized Python function")
            for suffix in identifiers:
                if key == suffix or key.endswith("_" + suffix):
                    if isinstance(item, str) and item not in identifiers[suffix]:
                        raise UnsafeModelConfiguration("unknown callable reference")
            stack.append((item, depth + 1))


def _read_h5_config(source: object) -> object:
    try:
        source.seek(0)
        with h5py.File(source, "r") as archive:
            if "model_config" not in archive.attrs or "model_weights" not in archive:
                raise InvalidModelStructure()
            if archive.attrs.get_id("model_config").get_storage_size() > _MAX_CONFIG_BYTES:
                raise InvalidModelStructure()
            raw = archive.attrs.get("model_config")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_CONFIG_BYTES:
                raise InvalidModelStructure()
            return json.loads(raw)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        if isinstance(error, InvalidModelStructure):
            raise
        raise InvalidModelStructure() from error


def _read_keras_config(source: object) -> object:
    try:
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            files = [info for info in archive.infolist() if info.filename == "config.json"]
            if len(files) != 1 or files[0].file_size > _MAX_CONFIG_BYTES:
                raise InvalidModelStructure()
            return json.loads(archive.read(files[0]).decode("utf-8"))
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        if isinstance(error, InvalidModelStructure):
            raise
        raise InvalidModelStructure() from error


def inspect_model_file_state(
    model_path: os.PathLike[str] | str,
    keras: object,
) -> ResolvedFile:
    """Validate an on-disk model before any Keras deserialization happens.

    Inspection reads from the opened regular-file descriptor. Its resolved
    target and file identity can be checked immediately before Keras reopens
    the same canonical path for deserialization.
    """

    try:
        with open_regular_file(model_path, "Model file") as (source, resolved):
            if os.fstat(source.fileno()).st_size == 0:
                raise ModelFileError(
                    f"Model file is empty: {resolved.path}. Install a trained model or choose another file."
                )
            suffix = resolved.path.suffix.lower()
            if suffix in {".h5", ".hdf5"}:
                config = _read_h5_config(source)
            elif suffix == ".keras":
                config = _read_keras_config(source)
            else:
                raise InvalidModelStructure()
            inspect_model_configuration(config, keras)
            current = FileIdentity.from_stat(os.fstat(source.fileno()))
            if current != resolved.identity:
                raise ModelFileError(
                    f"Model file changed while it was being inspected: {resolved.path}."
                )
            return resolved
    except ReadPathError as error:
        raise ModelFileError(str(error)) from error


def inspect_model_file(model_path: os.PathLike[str] | str, keras: object) -> Path:
    """Return the resolved target after safely inspecting its configuration."""

    return inspect_model_file_state(model_path, keras).path
