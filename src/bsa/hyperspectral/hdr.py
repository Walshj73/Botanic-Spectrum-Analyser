"""Tk-independent ENVI header generation for the BSA HDR Creator."""

from __future__ import annotations

import csv
import io
import os
import math
from pathlib import Path, PureWindowsPath
import re
import tempfile
from typing import Any
from typing import Sequence

from bsa.hyperspectral.envi_validation import (
    MAX_HEADER_BYTES,
    SUPPORTED_ENVI_DATA_TYPES,
    SUPPORTED_ENVI_INTERLEAVES,
    parse_envi_header,
)
from bsa.utils.read_paths import FileIdentity, ReadPathError, open_regular_file


# Keep the Creator's metadata well beyond ordinary scientific datasets while
# bounding the generated wavelength list and header text. The active GUI used
# to prompt for dimensions up to 5,000; these API limits are intentionally much
# higher to support larger acquisitions without permitting absurd allocations.
MAX_CREATOR_DIMENSION = 1_000_000
MAX_CREATOR_BANDS = 100_000
MAX_CREATOR_HEADER_OFFSET = 2_147_483_647
MAX_CREATOR_INTEGER_DIGITS = 10
MAX_WAVELENGTH_TOKEN_LENGTH = 128

_UNSIGNED_INTEGER = re.compile(r"[0-9]+\Z")
_FINITE_DECIMAL = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)


def integer_entry_text(text: str) -> bool:
    """Allow empty and pasted digits while editing; validate the final value on submit."""

    return isinstance(text, str) and len(text) <= MAX_CREATOR_INTEGER_DIGITS and bool(
        re.fullmatch(r"[0-9]*", text)
    )


def decimal_entry_text(text: str) -> bool:
    """Allow normal transitional decimal notation, including an empty field."""

    return isinstance(text, str) and len(text) <= MAX_WAVELENGTH_TOKEN_LENGTH and bool(
        re.fullmatch(r"[0-9.+eE-]*", text)
    )


def byte_order_entry_text(text: str) -> bool:
    return text in ("", "0", "1")


def load_wavelength_grid(path: os.PathLike[str] | str) -> tuple[float, ...]:
    """Load a measured grid, preserving the existing grid-only API."""

    return load_wavelength_calibration(path)[0]


def load_wavelength_calibration(
    path: os.PathLike[str] | str,
) -> tuple[tuple[float, ...], str]:
    """Load wavelengths and any authoritative ENVI wavelength-unit metadata."""

    selected = Path(path)
    suffix = selected.suffix.casefold()
    if suffix == ".hdr":
        parsed = parse_envi_header(selected)
        if parsed.wavelengths is None:
            raise ValueError("Selected ENVI header has no wavelength field.")
        if len(parsed.wavelengths) < 2 or len(parsed.wavelengths) > MAX_CREATOR_BANDS:
            raise ValueError("Imported wavelength count is outside HDR Creator's band range.")
        return parsed.wavelengths, _wavelength_units(parsed.wavelength_units)
    if suffix not in {".csv", ".txt"}:
        raise ValueError("Import an ENVI .hdr, one-column .csv, or one-value-per-line .txt file.")
    try:
        with open_regular_file(selected, "Wavelength grid") as (source, resolved):
            if resolved.identity.size > MAX_HEADER_BYTES:
                raise ValueError("Wavelength list exceeds the 4 MiB input limit.")
            raw = source.read(MAX_HEADER_BYTES + 1)
            if len(raw) > MAX_HEADER_BYTES:
                raise ValueError("Wavelength list exceeds the 4 MiB input limit.")
            if FileIdentity.from_stat(os.fstat(source.fileno())) != resolved.identity:
                raise ValueError("Wavelength list changed while it was being read.")
        text = raw.decode("utf-8-sig")
        if suffix == ".csv":
            rows = list(csv.reader(io.StringIO(text), strict=True))
            values = [row[0].strip() for row in rows if row]
            if any(len(row) != 1 for row in rows):
                raise ValueError("Wavelength CSV must have exactly one column.")
        else:
            values = [line.strip() for line in text.splitlines() if line.strip()]
        if not 2 <= len(values) <= MAX_CREATOR_BANDS:
            raise ValueError("Wavelength list must contain 2 to 100,000 values.")
        return tuple(_validated_wavelengths(values, len(values))), "Unknown"
    except (ReadPathError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Cannot import wavelength grid {selected}: {error}.") from error


def is_integer_character(character: str) -> bool:
    """Preserve the integer-entry validator used by the HDR tab."""

    return len(character) == 1 and character in "0123456789"


def is_decimal_character(character: str) -> bool:
    """Allow numeric notation; semantic validation checks the complete value."""

    return len(character) == 1 and character in "0123456789.+-eE"


def is_byte_order_edit(character: str, action: str, current_value: str) -> bool:
    """Validate the existing one-character byte-order entry behavior."""

    if action == "0":
        return True
    if character in ("0", "1"):
        return len(current_value) == 0
    return False


def has_valid_decimal_points(text: str) -> bool:
    """Return whether text has the existing maximum of one decimal point."""

    return len(text.split(".")) <= 2


def fields_are_complete(folder: str, *values: str) -> bool:
    """Return whether the folder and every existing HDR field contain text."""

    return bool(folder.strip()) and all(bool(value.strip()) for value in values)


def _integer_value(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Parse a bounded integer without rounding floats or accepting bools."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer.")
    if isinstance(value, int):
        number = value
        digits = len(str(abs(value)))
    elif isinstance(value, str):
        text = value.strip()
        if _UNSIGNED_INTEGER.fullmatch(text) is None:
            raise ValueError(f"{field} must be a whole-number integer.")
        digits = len(text)
        if digits > MAX_CREATOR_INTEGER_DIGITS:
            raise ValueError(f"{field} is too large for practical ENVI metadata.")
        number = int(text)
    else:
        raise ValueError(f"{field} must be a whole-number integer.")
    if digits > MAX_CREATOR_INTEGER_DIGITS or not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return number


def _finite_number(value: Any, field: str) -> float:
    """Parse one finite ENVI decimal value without accepting expressions."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number.")
    if isinstance(value, str):
        token = value.strip()
        if not token or len(token) > MAX_WAVELENGTH_TOKEN_LENGTH:
            raise ValueError(f"{field} must be a finite number.")
        if _FINITE_DECIMAL.fullmatch(token) is None:
            raise ValueError(f"{field} must be a finite number.")
    elif not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number.")
    return number


def _validated_wavelengths(values: Sequence[Any], bands: int) -> list[float]:
    try:
        count = len(values)
    except TypeError as error:
        raise ValueError("Wavelengths must be a sequence of numeric values.") from error
    if count != bands:
        raise ValueError(f"Wavelength count must match bands ({bands}).")
    parsed = [_finite_number(value, f"Wavelength {index + 1}")
              for index, value in enumerate(values)]
    if len(parsed) > 1:
        increasing = all(first < second for first, second in zip(parsed, parsed[1:]))
        decreasing = all(first > second for first, second in zip(parsed, parsed[1:]))
        if not increasing and not decreasing:
            raise ValueError("Wavelengths must be strictly increasing or decreasing without duplicates.")
    return parsed


def _wavelength_units(value: str) -> str:
    """Require a single safe ENVI field value without changing its meaning."""

    if (
        not isinstance(value, str) or not value or len(value) > 128
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 or character in "{}" for character in value)
    ):
        raise ValueError("Wavelength units must be a nonempty, single-line ENVI value.")
    return value


def _format_generated_wavelength(value: float) -> str:
    """Keep normal headers at two decimals without expanding huge exponents."""

    formatted = f"{value:.2f}"
    if len(formatted) > MAX_WAVELENGTH_TOKEN_LENGTH:
        return repr(value)
    return formatted


def generate_wavelengths(start: float, end: float, steps: int) -> list[float]:
    """Generate a bounded, finite, strictly ordered two-decimal wavelength list."""

    count = _integer_value(
        steps, "Bands", minimum=2, maximum=MAX_CREATOR_BANDS
    )
    first = _finite_number(start, "Starting wavelength")
    last = _finite_number(end, "Ending wavelength")
    if first == last:
        raise ValueError("Starting and ending wavelengths must be different.")
    wavelength_list = [first]
    step_size = (last - first) / (count - 1)
    for index in range(1, count):
        next_wavelength = round(first + index * step_size, 2)
        wavelength_list.append(next_wavelength)
    _validated_wavelengths(wavelength_list, count)
    return wavelength_list


def build_envi_header(
    samples: int | str,
    lines: int | str,
    bands: int | str,
    data_type: int | str,
    byte_order: int | str,
    wavelength_range: Sequence[Any] | None,
    *,
    wavelengths: Sequence[Any] | None = None,
    header_offset: int | str = 0,
    interleave: str = "BIL",
    wavelength_units: str = "Unknown",
) -> str:
    """Build a semantically valid ENVI header supported by BSA's reader."""

    sample_count = _integer_value(
        samples, "Samples", minimum=1, maximum=MAX_CREATOR_DIMENSION
    )
    line_count = _integer_value(
        lines, "Lines", minimum=1, maximum=MAX_CREATOR_DIMENSION
    )
    band_count = _integer_value(
        bands, "Bands", minimum=2, maximum=MAX_CREATOR_BANDS
    )
    type_code = _integer_value(data_type, "ENVI data type", minimum=0, maximum=99)
    if type_code not in SUPPORTED_ENVI_DATA_TYPES:
        raise ValueError(f"Unsupported ENVI data type {type_code}.")
    order = _integer_value(byte_order, "Byte order", minimum=0, maximum=1)
    offset = _integer_value(
        header_offset, "Header offset", minimum=0,
        maximum=MAX_CREATOR_HEADER_OFFSET,
    )
    if not isinstance(interleave, str) or interleave.casefold() not in SUPPORTED_ENVI_INTERLEAVES:
        raise ValueError("Interleave must be BIL, BIP or BSQ, as supported by BSA.")
    normalized_interleave = interleave.upper()
    units = _wavelength_units(wavelength_units)

    if wavelengths is not None:
        if wavelength_range is not None:
            raise ValueError("Supply either a wavelength range or a wavelength list, not both.")
        parsed_wavelengths = _validated_wavelengths(wavelengths, band_count)
        formatted_wavelengths = ", ".join(repr(value) for value in parsed_wavelengths)
    else:
        if wavelength_range is None:
            raise ValueError("Enter a wavelength range or wavelength list.")
        try:
            range_count = len(wavelength_range)
        except TypeError as error:
            raise ValueError("Wavelength range must contain a start and end value.") from error
        if range_count != 2:
            raise ValueError("Wavelength range must contain a start and end value.")
        parsed_wavelengths = generate_wavelengths(
            wavelength_range[0], wavelength_range[1], band_count
        )
        formatted_wavelengths = ", ".join(
            _format_generated_wavelength(value) for value in parsed_wavelengths
        )

    return f'''ENVI
description = {{ }}
samples = {sample_count}
lines = {line_count}
bands = {band_count}
header offset = {offset}
file type = ENVI Standard
data type = {type_code}
interleave = {normalized_interleave}
sensor type = Unknown
byte order = {order}
wavelength units = {units}
wavelength = {{
{formatted_wavelengths}
}}'''


def hdr_output_path(
    folder: os.PathLike[str] | str,
    file_name: str,
) -> Path:
    """Resolve one HDR leaf beneath the selected output directory."""

    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("Enter an HDR filename.")
    if (
        file_name in {".", ".."}
        or not file_name.strip(".")
        or any(character in file_name for character in ('/', '\\', ':', '<', '>', '"', '|', '?', '*'))
        or PureWindowsPath(file_name).drive
    ):
        raise ValueError("The HDR filename must be one filename, without a path.")
    if any(ord(character) < 32 or ord(character) == 127 for character in file_name):
        raise ValueError("The HDR filename cannot contain control characters.")
    if file_name[-1] in {" ", "."}:
        raise ValueError("The HDR filename cannot end with a space or dot.")

    name = file_name if file_name.lower().endswith(".hdr") else f"{file_name}.hdr"
    windows_stem = name.split(".", 1)[0].rstrip(" ").upper()
    if windows_stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or windows_stem in {
        *(f"COM{number}" for number in (*range(1, 10), "¹", "²", "³")),
        *(f"LPT{number}" for number in (*range(1, 10), "¹", "²", "³")),
    }:
        raise ValueError("The HDR filename is a reserved device name.")

    try:
        directory = Path(folder).resolve(strict=True)
    except RuntimeError as error:
        raise ValueError("The HDR output folder contains a symlink loop.") from error
    if not directory.is_dir():
        raise NotADirectoryError(f"HDR output folder is not a directory: {directory}")
    try:
        maximum = os.pathconf(directory, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        maximum = 255
    if maximum <= 0:
        maximum = 255
    length = (
        len(name.encode("utf-16-le")) // 2 if os.name == "nt"
        else len(os.fsencode(name))
    )
    if length > maximum:
        raise ValueError("The HDR filename is too long for the output folder.")

    output_path = directory / name
    try:
        resolved_output = output_path.resolve(strict=False)
    except RuntimeError as error:
        raise ValueError("The HDR filename points through a symlink loop.") from error
    if not resolved_output.is_relative_to(directory):
        raise ValueError("The HDR filename resolves outside the selected folder.")
    if os.path.lexists(output_path):
        raise FileExistsError(f"An HDR output with this name already exists: {name}")
    return output_path


def write_envi_header(
    samples: int | str,
    lines: int | str,
    bands: int | str,
    data_type: int | str,
    byte_order: int | str,
    wavelength_range: Sequence[Any] | None,
    file_name: str,
    folder: os.PathLike[str] | str,
    *,
    wavelengths: Sequence[Any] | None = None,
    header_offset: int | str = 0,
    interleave: str = "BIL",
    wavelength_units: str = "Unknown",
) -> Path:
    """Generate and write one header, returning its platform-neutral path."""

    output_path = hdr_output_path(folder, file_name)
    header = build_envi_header(
        samples,
        lines,
        bands,
        data_type,
        byte_order,
        wavelength_range,
        wavelengths=wavelengths,
        header_offset=header_offset,
        interleave=interleave,
        wavelength_units=wavelength_units,
    )
    if len(header.encode("utf-8")) > MAX_HEADER_BYTES:
        raise ValueError(
            f"Generated HDR exceeds the {MAX_HEADER_BYTES}-byte ENVI metadata limit."
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=output_path.parent,
            prefix=".bsa-hdr-", suffix=".tmp", delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            output_file.write(header)
            output_file.flush()
            os.fsync(output_file.fileno())
        # Validate the complete staged text through the same production parser
        # used before BSA opens ENVI binaries. Nothing is published if Creator
        # and Reader disagree about its metadata.
        parse_envi_header(temporary_path)
        # Both operations publish a complete file and fail if the destination
        # appeared after validation. POSIX rename replaces existing files, so
        # use a hard link there; Windows rename refuses an existing target.
        if os.name == "nt":
            os.rename(temporary_path, output_path)
            temporary_path = None
        else:
            os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path
