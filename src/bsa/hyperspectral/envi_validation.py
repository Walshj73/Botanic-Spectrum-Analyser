"""Preflight untrusted ENVI headers and binary cubes before Spectral loads them."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

from bsa.hyperspectral import dataset
from bsa.utils.read_paths import (
    FileIdentity, ReadPathError, ResolvedFile, open_regular_file,
)


MAX_HEADER_BYTES = 4 * 1024 * 1024
_UNSIGNED_INTEGER = re.compile(r"[0-9]{1,32}\Z")
_FINITE_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_DTYPES = {
    1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 6: "c8",
    9: "c16", 12: "u2", 13: "u4", 14: "i8", 15: "u8",
}
_INTERLEAVES = {"bil", "BIL", "bip", "BIP", "bsq", "BSQ"}
SUPPORTED_ENVI_DATA_TYPES = frozenset(_DTYPES)
SUPPORTED_ENVI_INTERLEAVES = frozenset(value.lower() for value in _INTERLEAVES)


class EnviInputError(ValueError):
    """An ENVI header or binary is ambiguous, inconsistent, or unsafe to load."""


@dataclass(frozen=True)
class EnviMetadata:
    header_path: Path
    samples: int
    lines: int
    bands: int
    interleave: str
    data_type: int
    byte_order: int
    header_offset: int
    wavelengths: tuple[float, ...] | None
    sensor_type: str
    itemsize: int
    header_file: ResolvedFile | None = None
    binary_file: ResolvedFile | None = None
    wavelength_units: str = "Unknown"

    @property
    def expected_bytes(self) -> int:
        return self.header_offset + self.lines * self.samples * self.bands * self.itemsize


def _read_header_fields(
    path: os.PathLike[str] | str,
) -> tuple[dict[str, tuple[str, bool]], ResolvedFile]:
    """Read a bounded header from one opened, resolved regular-file handle."""

    header = Path(path)
    try:
        with open_regular_file(header, "ENVI header") as (source, resolved):
            if resolved.identity.size > MAX_HEADER_BYTES:
                raise EnviInputError(
                    f"ENVI header exceeds the {MAX_HEADER_BYTES}-byte metadata limit: {header}."
                )
            content = source.read(MAX_HEADER_BYTES + 1)
            if FileIdentity.from_stat(os.fstat(source.fileno())) != resolved.identity:
                raise EnviInputError(f"ENVI header changed while it was being read: {header}.")
    except ReadPathError as error:
        raise EnviInputError(str(error)) from error
    if len(content) > MAX_HEADER_BYTES:
        raise EnviInputError(
            f"ENVI header exceeds the {MAX_HEADER_BYTES}-byte metadata limit: {header}."
        )
    try:
        content = content.decode("utf-8")
    except UnicodeError as error:
        raise EnviInputError(f"Cannot decode ENVI header {header}: {error}.") from error
    lines = content.splitlines()
    if not lines or lines[0].strip() != "ENVI":
        raise EnviInputError(f"Missing ENVI header signature in {header}.")

    fields: dict[str, tuple[str, bool]] = {}
    index = 1
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith(";"):
            continue
        if "=" not in line:
            raise EnviInputError(f"Malformed ENVI header field in {header}: {line[:80]!r}.")
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.casefold()
        if not key:
            raise EnviInputError(f"Empty ENVI header field name in {header}.")
        if key in fields:
            raise EnviInputError(f"Duplicate ENVI header field {key!r} in {header}.")

        braced = value.startswith("{")
        if braced:
            while not value.rstrip().endswith("}"):
                if index >= len(lines):
                    raise EnviInputError(f"Unclosed ENVI {key!r} list in {header}.")
                continuation = lines[index].strip()
                index += 1
                if not continuation.startswith(";"):
                    value += "\n" + continuation
            if value.count("{") != 1 or value.count("}") != 1:
                raise EnviInputError(f"Malformed ENVI {key!r} list in {header}.")
            value = value[1:-1]
        elif "{" in value or "}" in value:
            raise EnviInputError(f"Malformed ENVI {key!r} value in {header}.")
        fields[key] = (value.strip(), braced)
    return fields, resolved


def read_header_fields(path: os.PathLike[str] | str) -> dict[str, tuple[str, bool]]:
    """Read one bounded header, retaining brace structure and rejecting duplicates."""

    return _read_header_fields(path)[0]


def _integer(fields: dict[str, tuple[str, bool]], key: str, *,
             positive: bool = False, default: int | None = None) -> int:
    if key not in fields:
        if default is not None:
            return default
        raise EnviInputError(f"Missing required ENVI field {key!r}.")
    text, braced = fields[key]
    if braced or _UNSIGNED_INTEGER.fullmatch(text) is None:
        raise EnviInputError(f"ENVI field {key!r} must be a nonnegative decimal integer.")
    number = int(text)
    if positive and number == 0:
        raise EnviInputError(f"ENVI field {key!r} must be greater than zero.")
    if number > sys.maxsize:
        raise EnviInputError(f"ENVI field {key!r} exceeds process addressability.")
    return number


def band_count(fields: dict[str, tuple[str, bool]]) -> int | None:
    return _integer(fields, "bands", positive=True) if "bands" in fields else None


def wavelength_values(fields: dict[str, tuple[str, bool]],
                      bands: int | None) -> tuple[float, ...] | None:
    if "wavelength" not in fields:
        return None
    value, braced = fields["wavelength"]
    if not braced or bands is None:
        raise EnviInputError("ENVI wavelengths require a braced list and a valid bands field.")
    if not value or value.count(",") + 1 != bands:
        raise EnviInputError(f"ENVI wavelength count does not match bands ({bands}).")
    values: list[float] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            raise EnviInputError("ENVI wavelength list contains an empty value.")
        if _FINITE_DECIMAL.fullmatch(token) is None:
            raise EnviInputError(f"Invalid ENVI wavelength {token[:40]!r}.")
        try:
            number = float(token)
        except ValueError as error:
            raise EnviInputError(f"Invalid ENVI wavelength {token[:40]!r}.") from error
        if not math.isfinite(number):
            raise EnviInputError("ENVI wavelengths must be finite.")
        values.append(number)
    if len(values) > 1:
        increasing = all(first < second for first, second in zip(values, values[1:]))
        decreasing = all(first > second for first, second in zip(values, values[1:]))
        if not increasing and not decreasing:
            raise EnviInputError("ENVI wavelengths must be strictly ordered without duplicates.")
    return tuple(values)


def parse_envi_header(path: os.PathLike[str] | str) -> EnviMetadata:
    fields, header_file = _read_header_fields(path)
    samples = _integer(fields, "samples", positive=True)
    lines = _integer(fields, "lines", positive=True)
    bands = _integer(fields, "bands", positive=True)
    offset = _integer(fields, "header offset", default=0)
    code = _integer(fields, "data type")
    if code not in _DTYPES:
        raise EnviInputError(f"Unsupported ENVI data type {code}.")
    byte_order = _integer(fields, "byte order")
    if byte_order not in (0, 1):
        raise EnviInputError(f"Invalid ENVI byte order {byte_order}; expected 0 or 1.")
    if "interleave" not in fields or fields["interleave"][1]:
        raise EnviInputError("Missing or malformed ENVI interleave field.")
    interleave = fields["interleave"][0]
    if interleave not in _INTERLEAVES:
        raise EnviInputError(f"Unsupported ENVI interleave {interleave!r}.")
    file_type, file_type_braced = fields.get("file type", ("ENVI Standard", False))
    if file_type_braced or file_type != "ENVI Standard":
        raise EnviInputError(f"Unsupported ENVI file type {file_type!r}.")
    if "reflectance scale factor" in fields:
        if fields["reflectance scale factor"][1] or _FINITE_DECIMAL.fullmatch(fields["reflectance scale factor"][0]) is None:
            raise EnviInputError("Invalid ENVI reflectance scale factor.")
        try:
            scale = float(fields["reflectance scale factor"][0])
        except ValueError as error:
            raise EnviInputError("Invalid ENVI reflectance scale factor.") from error
        if not math.isfinite(scale) or scale <= 0:
            raise EnviInputError("ENVI reflectance scale factor must be finite and positive.")
    wavelengths = wavelength_values(fields, bands)
    itemsize = np.dtype(_DTYPES[code]).itemsize
    payload = lines * samples * bands * itemsize
    if payload > sys.maxsize - offset:
        raise EnviInputError("ENVI cube exceeds process addressability.")
    return EnviMetadata(
        header_file.path, samples, lines, bands, interleave.lower(), code, byte_order,
        offset, wavelengths, fields.get("sensor type", ("", False))[0], itemsize,
        header_file=header_file,
        wavelength_units=fields.get("wavelength units", ("Unknown", False))[0],
    )


def validate_envi_binary(header_path: os.PathLike[str] | str,
                         image_path: os.PathLike[str] | str) -> EnviMetadata:
    """Require exact regular-file size before Spectral can allocate the cube."""

    metadata = parse_envi_header(header_path)
    image = Path(image_path)
    try:
        with open_regular_file(image, "ENVI binary") as (source, binary_file):
            details = os.fstat(source.fileno())
            if FileIdentity.from_stat(details) != binary_file.identity:
                raise EnviInputError(f"ENVI binary changed while it was being checked: {image}.")
            if metadata.header_file is not None:
                require_file_unchanged(metadata.header_file, "ENVI header")
            if metadata.header_offset > details.st_size:
                raise EnviInputError(f"ENVI header offset exceeds binary file size: {image}.")
            if details.st_size != metadata.expected_bytes:
                relation = "truncated" if details.st_size < metadata.expected_bytes else "excess data"
                raise EnviInputError(
                    f"ENVI binary has {relation}: expected {metadata.expected_bytes} bytes, "
                    f"found {details.st_size} bytes in {image}."
                )
    except ReadPathError as error:
        raise EnviInputError(str(error)) from error
    identity = dataset.identify_hyperspectral_file(image.name)
    if identity is not None and metadata.sensor_type.strip().upper() in {"SWIR", "VNIR"}:
        if identity.sensor != metadata.sensor_type.strip().upper():
            raise EnviInputError(
                f"ENVI sensor type {metadata.sensor_type!r} contradicts {image.name!r}."
            )
    return replace(metadata, binary_file=binary_file)


def require_file_unchanged(resolved: ResolvedFile, description: str) -> None:
    """Translate the shared read-path check to the ENVI validation contract."""

    from bsa.utils.read_paths import require_unchanged

    try:
        require_unchanged(resolved, description)
    except ReadPathError as error:
        raise EnviInputError(str(error)) from error


def validate_reference_triplet(dark: EnviMetadata, data: EnviMetadata,
                               white: EnviMetadata) -> None:
    """Reject incompatible reference grids before any cube is loaded."""

    for role, reference in (("Dark", dark), ("White", white)):
        if reference.samples != data.samples or reference.bands != data.bands:
            raise EnviInputError(f"{role}/Data ENVI samples or bands do not match.")
        if reference.wavelengths != data.wavelengths:
            raise EnviInputError(f"{role}/Data ENVI wavelength grids do not match.")
        if reference.data_type != data.data_type or reference.byte_order != data.byte_order:
            raise EnviInputError(f"{role}/Data ENVI data type or byte order does not match.")
        if reference.interleave != data.interleave:
            raise EnviInputError(f"{role}/Data ENVI interleave does not match.")
        known = {value.sensor_type.strip().upper() for value in (reference, data)} & {"SWIR", "VNIR"}
        if len(known) > 1:
            raise EnviInputError(f"{role}/Data ENVI sensor types conflict.")
