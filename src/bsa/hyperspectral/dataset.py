"""Validated discovery and identity matching for BSA hyperspectral datasets.

PlantScreen BIL triplets encode identity in their common filename prefix. RAW
files use generic calibration names, so BSA accepts a RAW set only when there
is one data cube, one dark reference and one white reference. Mask association
is a separate validation layer: the analyser requires it, while metadata-only
tools may omit it. Ambiguous inputs are rejected before any scientific data is
loaded or calibrated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path, PurePath
import re
from typing import Iterable


_BIL_FILE_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<sensor>VNIR|SWIR|OTHER)-"
    r"(?P<role>DarkCalibration|Data|WhiteCalibration)\.bil$",
    re.IGNORECASE,
)
_RAW_DATA_PATTERN = re.compile(r"^vnir_\d+_\d+_\d+\.raw$", re.IGNORECASE)
_RAW_DARK_PATTERN = re.compile(r"^DARKREF_vnir\.raw$", re.IGNORECASE)
_RAW_WHITE_PATTERN = re.compile(r"^WHITEREF_vnir\.raw$", re.IGNORECASE)
_MASK_SUFFIX_PATTERN = re.compile(r"-PlantMask$", re.IGNORECASE)
_RGB_PREVIEW_SUFFIX_PATTERN = re.compile(r"-HcRgbImage-\d+$", re.IGNORECASE)
_SENSOR_IDENTITY_PATTERN = re.compile(r"^.+-(VNIR|SWIR|OTHER)$", re.IGNORECASE)
_RAW_MASK_IDENTITY_PATTERN = re.compile(r"^vnir_\d+_\d+_\d+$", re.IGNORECASE)


class DatasetMatchingError(ValueError):
    """Raised when selected files cannot be matched without guessing."""

    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Hyperspectral dataset matching failed:\n{details}")


def categorize_file(file_name: str) -> str:
    """Categorize a filename using the application's existing role patterns."""

    dark_bil_pattern = r"-VNIR-DarkCalibration"
    data_bil_pattern = r"-VNIR-Data"
    white_bil_pattern = r"-VNIR-WhiteCalibration"

    dark_swir_pattern = r"-SWIR-DarkCalibration"
    data_swir_pattern = r"-SWIR-Data"
    white_swir_pattern = r"-SWIR-WhiteCalibration"

    dark_other_pattern = r"-OTHER-DarkCalibration"
    data_other_pattern = r"-OTHER-Data"
    white_other_pattern = r"-OTHER-WhiteCalibration"

    dark_raw_pattern = r"DARKREF_vnir"
    data_raw_pattern = r"vnir_\d+_\d+_\d+"
    white_raw_pattern = r"WHITEREF_vnir"

    if (
        re.search(dark_bil_pattern, file_name, re.IGNORECASE)
        or re.search(dark_swir_pattern, file_name, re.IGNORECASE)
        or re.search(dark_other_pattern, file_name, re.IGNORECASE)
        or re.search(dark_raw_pattern, file_name, re.IGNORECASE)
    ):
        return "dark"
    if (
        re.search(data_bil_pattern, file_name, re.IGNORECASE)
        or re.search(data_swir_pattern, file_name, re.IGNORECASE)
        or re.search(data_other_pattern, file_name, re.IGNORECASE)
        or (
            re.search(data_raw_pattern, file_name, re.IGNORECASE)
            and not re.search(white_raw_pattern, file_name, re.IGNORECASE)
        )
    ):
        return "data"
    if (
        re.search(white_bil_pattern, file_name, re.IGNORECASE)
        or re.search(white_swir_pattern, file_name, re.IGNORECASE)
        or re.search(white_other_pattern, file_name, re.IGNORECASE)
        or re.search(white_raw_pattern, file_name, re.IGNORECASE)
    ):
        return "white"
    return "unknown"


@dataclass(frozen=True)
class DatasetSetPaths:
    """One validated dark, data and white association, with an optional mask.

    ``identity_key`` is the case-insensitive, format-qualified identity used by
    dataset discovery.  It is deliberately exposed here so other BSA tools can
    reuse the validated identity without parsing filenames again.
    """

    identity: str
    dark: PurePath
    data: PurePath
    white: PurePath
    mask: PurePath | None
    identity_key: str = ""
    sensor: str = ""
    format_name: str = ""


@dataclass(frozen=True)
class DiscoveredDataset:
    """Discovered filenames and their validated sample associations."""

    image_directory: PurePath
    mask_directory: PurePath | None
    image_files: list[str]
    mask_files: list[str]
    dark_files: list[str]
    data_files: list[str]
    white_files: list[str]
    matched_sets: tuple[DatasetSetPaths, ...]

    @property
    def number_of_files(self) -> int:
        return len(self.image_files)

    @property
    def number_of_sets(self) -> int:
        return len(self.matched_sets)


@dataclass(frozen=True)
class _IdentifiedFile:
    identity: str
    key: str
    role: str
    name: str
    format_name: str
    sensor: str


def _bil_identity(file_name: str) -> _IdentifiedFile | None:
    match = _BIL_FILE_PATTERN.fullmatch(file_name)
    if match is None:
        return None
    sensor = match.group("sensor").upper()
    identity = f"{match.group('prefix')}-{sensor}"
    role = {
        "darkcalibration": "dark",
        "data": "data",
        "whitecalibration": "white",
    }[match.group("role").casefold()]
    return _IdentifiedFile(
        identity=identity,
        key=f"bil:{identity.casefold()}",
        role=role,
        name=file_name,
        format_name="BIL",
        sensor=sensor,
    )


def _raw_identity(file_name: str) -> _IdentifiedFile | None:
    if _RAW_DATA_PATTERN.fullmatch(file_name):
        identity = Path(file_name).stem
        return _IdentifiedFile(
            identity=identity,
            key=f"raw:{identity.casefold()}",
            role="data",
            name=file_name,
            format_name="RAW",
            sensor="VNIR",
        )
    if _RAW_DARK_PATTERN.fullmatch(file_name):
        return _IdentifiedFile(
            "VNIR RAW reference", "raw:reference", "dark", file_name, "RAW", "VNIR"
        )
    if _RAW_WHITE_PATTERN.fullmatch(file_name):
        return _IdentifiedFile(
            "VNIR RAW reference", "raw:reference", "white", file_name, "RAW", "VNIR"
        )
    return None


def identify_hyperspectral_file(file_name: str) -> _IdentifiedFile | None:
    """Return the identity and role encoded by one supported filename."""

    if file_name.casefold().endswith(".bil"):
        return _bil_identity(file_name)
    if file_name.casefold().endswith(".raw"):
        return _raw_identity(file_name)
    return None


def mask_identity(file_name: str) -> tuple[str, str] | None:
    """Return ``(key, identity)`` for a recognisable Mask Creator filename."""

    stem = Path(file_name).stem
    suffix_match = _MASK_SUFFIX_PATTERN.search(stem)
    if suffix_match is None or suffix_match.end() != len(stem):
        return None
    identity = stem[: suffix_match.start()]
    preview_match = _RGB_PREVIEW_SUFFIX_PATTERN.search(identity)
    if preview_match is not None and preview_match.end() == len(identity):
        identity = identity[: preview_match.start()]
        if _SENSOR_IDENTITY_PATTERN.fullmatch(identity):
            return f"bil:{identity.casefold()}", identity
    if _RAW_MASK_IDENTITY_PATTERN.fullmatch(identity):
        return f"raw:{identity.casefold()}", identity
    return None


def _duplicate_issue(identity: str, role: str, names: list[str]) -> str:
    return (
        f"Duplicate {role} files for sample '{identity}': "
        + ", ".join(repr(name) for name in names)
    )


def discover_dataset(
    image_directory: os.PathLike[str] | str,
    mask_directory: os.PathLike[str] | str | None,
    *,
    require_masks: bool = True,
) -> DiscoveredDataset:
    """Discover acquisitions and validate masks when supplied.

    The analyser-facing default remains strict: every acquisition must have
    exactly one identity-matched mask. Metadata-only callers may set
    ``require_masks=False``; supplied masks are still validated, but absent
    masks do not invalidate an otherwise complete acquisition triplet.
    """

    image_names = sorted(
        name
        for name in os.listdir(image_directory)
        if name.casefold().endswith((".bil", ".raw"))
    )
    issues: list[str] = []
    identified_files: list[_IdentifiedFile] = []
    if mask_directory is None:
        mask_names: list[str] = []
        if require_masks:
            issues.append("A segmentation mask folder is required for analysis.")
    else:
        mask_names = sorted(
            name
            for name in os.listdir(mask_directory)
            if name.casefold().endswith((".png", ".jpg"))
        )

    if not image_names:
        issues.append("No supported hyperspectral data files were found in the selected folder.")

    for name in image_names:
        identified = identify_hyperspectral_file(name)
        if identified is None:
            role = categorize_file(name)
            if role == "unknown":
                issues.append(
                    f"Unrecognised hyperspectral file {name!r}; no supported sample identity or role is encoded."
                )
            else:
                issues.append(
                    f"Ambiguous {role} filename {name!r}; it resembles a supported role but does not follow an identity-bearing BIL or exact RAW convention."
                )
            continue
        identified_files.append(identified)

    dark_names = sorted(
        (file.name for file in identified_files if file.role == "dark"),
        key=str.casefold,
    )
    data_names = sorted(
        (file.name for file in identified_files if file.role == "data"),
        key=str.casefold,
    )
    white_names = sorted(
        (file.name for file in identified_files if file.role == "white"),
        key=str.casefold,
    )

    roles_by_key: dict[str, dict[str, list[_IdentifiedFile]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for file in identified_files:
        roles_by_key[file.key][file.role].append(file)

    candidate_sets: dict[str, dict[str, _IdentifiedFile]] = {}
    candidate_identities: dict[str, str] = {}

    for key in sorted(key for key in roles_by_key if key.startswith("bil:")):
        roles = roles_by_key[key]
        identity = next(file.identity for files in roles.values() for file in files)
        candidate_identities[key] = identity
        for role in ("dark", "data", "white"):
            files = roles.get(role, [])
            if not files:
                issues.append(f"Missing {role} file for sample '{identity}'.")
            elif len(files) > 1:
                issues.append(
                    _duplicate_issue(identity, role, [file.name for file in files])
                )
        if all(len(roles.get(role, [])) == 1 for role in ("dark", "data", "white")):
            candidate_sets[key] = {
                role: roles[role][0] for role in ("dark", "data", "white")
            }

    raw_data_files = [
        file
        for file in identified_files
        if file.format_name == "RAW" and file.role == "data"
    ]
    raw_reference_roles = roles_by_key.get("raw:reference", {})
    raw_dark_files = raw_reference_roles.get("dark", [])
    raw_white_files = raw_reference_roles.get("white", [])
    if raw_data_files or raw_dark_files or raw_white_files:
        for raw_data in raw_data_files:
            candidate_identities[raw_data.key] = raw_data.identity
        if len(raw_data_files) > 1:
            issues.append(
                "Ambiguous RAW calibration association: generic files "
                f"{[file.name for file in raw_dark_files]} and "
                f"{[file.name for file in raw_white_files]} do not encode which of "
                f"the data files {[file.name for file in raw_data_files]} they calibrate."
            )
        if len(raw_dark_files) != 1:
            if not raw_dark_files:
                issues.append("Missing DARKREF_vnir.raw for the RAW data set.")
            else:
                issues.append(
                    _duplicate_issue(
                        "VNIR RAW reference",
                        "dark",
                        [file.name for file in raw_dark_files],
                    )
                )
        if len(raw_white_files) != 1:
            if not raw_white_files:
                issues.append("Missing WHITEREF_vnir.raw for the RAW data set.")
            else:
                issues.append(
                    _duplicate_issue(
                        "VNIR RAW reference",
                        "white",
                        [file.name for file in raw_white_files],
                    )
                )
        if len(raw_data_files) == 1:
            raw_data = raw_data_files[0]
            if len(raw_dark_files) == 1 and len(raw_white_files) == 1:
                candidate_sets[raw_data.key] = {
                    "dark": raw_dark_files[0],
                    "data": raw_data,
                    "white": raw_white_files[0],
                }
        elif not raw_data_files:
            issues.append("RAW calibration references were found without a VNIR RAW data file.")

    masks_by_key: dict[str, list[str]] = defaultdict(list)
    for name in mask_names:
        identified_mask = mask_identity(name)
        if identified_mask is None:
            issues.append(
                f"Unmatched mask {name!r}; its filename does not identify a supported BIL or RAW sample produced by Mask Creator."
            )
            continue
        key, _ = identified_mask
        masks_by_key[key].append(name)

    for key, names in masks_by_key.items():
        if key not in candidate_identities:
            issues.append(
                f"Mask file(s) {names!r} identify a sample with no matching data cube."
            )

    for key, identity in candidate_identities.items():
        masks = masks_by_key.get(key, [])
        if not masks and require_masks:
            issues.append(f"Missing segmentation mask for sample '{identity}'.")
        elif len(masks) > 1:
            issues.append(_duplicate_issue(identity, "mask", masks))

    if issues:
        raise DatasetMatchingError(issues)

    image_root = Path(image_directory)
    mask_root = Path(mask_directory) if mask_directory is not None else None
    matched_sets = tuple(
        DatasetSetPaths(
            identity=candidate_identities[key],
            dark=image_root / roles["dark"].name,
            data=image_root / roles["data"].name,
            white=image_root / roles["white"].name,
            mask=(
                mask_root / masks_by_key[key][0]
                if mask_root is not None and masks_by_key.get(key)
                else None
            ),
            identity_key=key,
            sensor=roles["data"].sensor,
            format_name=roles["data"].format_name,
        )
        for key, roles in sorted(
            candidate_sets.items(),
            key=lambda item: item[1]["data"].name.casefold(),
        )
    )
    return DiscoveredDataset(
        image_directory=image_root,
        mask_directory=mask_root,
        image_files=image_names,
        mask_files=mask_names,
        dark_files=dark_names,
        data_files=data_names,
        white_files=white_names,
        matched_sets=matched_sets,
    )


def discover_acquisitions(
    image_directory: os.PathLike[str] | str,
    mask_directory: os.PathLike[str] | str | None = None,
) -> DiscoveredDataset:
    """Discover validated acquisitions with optional mask validation.

    This is the metadata-only Label Creator entry point. It deliberately calls
    the same parser and association implementation as ``discover_dataset``.
    """

    return discover_dataset(
        image_directory,
        mask_directory,
        require_masks=False,
    )


def paths_for_set(dataset: DiscoveredDataset, index: int) -> DatasetSetPaths:
    """Return one prevalidated association in deterministic data-file order."""

    return dataset.matched_sets[index]
