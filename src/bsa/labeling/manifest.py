"""Identity-safe CSV label manifests shared by BSA applications.

This module deliberately consumes :class:`DatasetSetPaths` produced by the
shared discovery implementation in ``hyperspectral_dataset``. It does not
parse acquisition filenames and therefore cannot drift from BSA's validated
BIL/RAW matching rules.

An acquisition ID is a digest of metadata already available from Phase 6A:
the format-qualified identity, sensor, format, acquisition/calibration
filenames, and data-file size from its directory entry. Masks are deliberately
excluded, so IDs remain stable before and after segmentation and when a mask is
renamed or moved. No hyperspectral file is opened, decoded, loaded, or
content-hashed. IDs also remain stable when a validated dataset is moved to a
different root. The dataset ID is derived from the complete, order-independent
set of acquisition IDs. Human-readable provenance remains in every CSV row and
is checked when the manifest is applied.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping

from bsa.exporting.output_safety import PlannedOutput, publish_staged_outputs, validate_output_plan
from bsa.hyperspectral.dataset import DatasetSetPaths, DiscoveredDataset
from bsa.utils.read_paths import FileIdentity, ReadPathError, open_regular_file


MANIFEST_VERSION = "2"
# Metadata budgets: 64 MiB accommodates tens of thousands of acquisitions;
# 32,767 characters is also Excel's maximum exact cell text length.
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_ROWS = 50_000
MAX_MANIFEST_FIELD_CHARS = 32_767
MAX_MANIFEST_LINE_CHARS = 1024 * 1024
LABELLED = "labelled"
UNLABELLED = "unlabelled"
MANIFEST_COLUMNS = (
    "manifest_version",
    "dataset_id",
    "acquisition_id",
    "sample_id",
    "identity_key",
    "sensor",
    "file_format",
    "data_file",
    "dark_file",
    "white_file",
    "data_size_bytes",
    "label_status",
    "class",
)
MAX_MANIFEST_COLUMNS = len(MANIFEST_COLUMNS)

_LEGACY_MANIFEST_COLUMNS_V1 = (
    "manifest_version",
    "dataset_id",
    "acquisition_id",
    "sample_id",
    "identity_key",
    "sensor",
    "file_format",
    "data_file",
    "dark_file",
    "white_file",
    "mask_file",
    "data_size_bytes",
    "label_status",
    "class",
)

_ACQUISITION_ID_PATTERN = re.compile(r"^bsa-acq-v2:[0-9a-f]{64}$")
_DATASET_ID_PATTERN = re.compile(r"^bsa-dataset-v2:[0-9a-f]{64}$")
_FILE_SIZE_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")


class LabelManifestError(ValueError):
    """Raised when a manifest cannot be trusted or applied unambiguously."""

    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Label manifest validation failed:\n{details}")


class _BoundedManifestReader(io.RawIOBase):
    """Limit actual bytes read, including a file grown after fstat."""

    def __init__(self, source: io.RawIOBase):
        self.source = source
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        remaining = MAX_MANIFEST_BYTES - self.bytes_read
        amount = self.source.readinto(memoryview(buffer)[: min(len(buffer), remaining + 1)])
        if amount is None:
            return 0
        self.bytes_read += amount
        if self.bytes_read > MAX_MANIFEST_BYTES:
            raise LabelManifestError(["The label manifest exceeds the 64 MiB file limit."])
        return amount

    def close(self) -> None:
        try:
            self.source.close()
        finally:
            super().close()


def _bounded_lines(source: io.TextIOBase) -> Iterable[str]:
    """Bound each physical CSV line before csv.reader receives it."""

    while True:
        line = source.readline(MAX_MANIFEST_LINE_CHARS + 1)
        if not line:
            return
        if len(line) > MAX_MANIFEST_LINE_CHARS:
            raise LabelManifestError(["A label manifest CSV line exceeds the 1 MiB limit."])
        yield line


@dataclass(frozen=True)
class AcquisitionProvenance:
    """Stable identity and reviewable provenance for one validated set."""

    acquisition_id: str
    sample_id: str
    identity_key: str
    sensor: str
    file_format: str
    data_file: str
    dark_file: str
    white_file: str
    data_size_bytes: str


@dataclass(frozen=True)
class LabelManifestRecord:
    """One labelled or explicitly unlabelled acquisition row."""

    manifest_version: str
    dataset_id: str
    acquisition_id: str
    sample_id: str
    identity_key: str
    sensor: str
    file_format: str
    data_file: str
    dark_file: str
    white_file: str
    data_size_bytes: str
    label_status: str
    class_name: str
    source_line: int = 0


@dataclass(frozen=True)
class LabelManifest:
    """Parsed manifest data; record order has no semantic meaning."""

    dataset_id: str
    records: tuple[LabelManifestRecord, ...]


@dataclass(frozen=True)
class ValidatedLabelAssignments:
    """The analyser-facing result of identity and provenance validation."""

    classes_by_acquisition_id: dict[str, str]
    classes_by_identity_key: dict[str, str]
    unlabelled_acquisition_ids: tuple[str, ...]
    dataset_id: str

    @property
    def labelled_count(self) -> int:
        return len(self.classes_by_acquisition_id)

    @property
    def excluded_unlabelled_count(self) -> int:
        """Number class-based analysis must exclude and report."""

        return len(self.unlabelled_acquisition_ids)

    @property
    def total_count(self) -> int:
        return self.labelled_count + self.excluded_unlabelled_count

    def class_for(self, acquisition: DatasetSetPaths) -> str | None:
        """Return the assigned class, or ``None`` for an explicit non-label."""

        return self.classes_by_identity_key.get(acquisition.identity_key.casefold())


def _sets_from_dataset(
    dataset: DiscoveredDataset | Iterable[DatasetSetPaths],
) -> tuple[DatasetSetPaths, ...]:
    if isinstance(dataset, DiscoveredDataset):
        return dataset.matched_sets
    return tuple(dataset)


def _prefixed_digest(prefix: str, values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{prefix}:{digest.hexdigest()}"


def acquisition_provenance(acquisition: DatasetSetPaths) -> AcquisitionProvenance:
    """Build a stable identity from an already validated acquisition."""

    issues = []
    if not acquisition.identity:
        issues.append("A validated acquisition has no sample identity.")
    if not acquisition.identity_key:
        issues.append(
            f"Acquisition {acquisition.identity!r} has no Phase 6A identity key."
        )
    if acquisition.sensor not in {"VNIR", "SWIR", "OTHER"}:
        issues.append(
            f"Acquisition {acquisition.identity!r} has invalid sensor {acquisition.sensor!r}."
        )
    if acquisition.format_name not in {"BIL", "RAW"}:
        issues.append(
            f"Acquisition {acquisition.identity!r} has invalid format {acquisition.format_name!r}."
        )
    if issues:
        raise LabelManifestError(issues)

    data_file = Path(acquisition.data)
    dark_file = Path(acquisition.dark)
    white_file = Path(acquisition.white)
    try:
        data_size_bytes = data_file.stat().st_size
    except OSError as error:
        raise LabelManifestError(
            [
                f"Cannot read directory metadata for data file "
                f"{str(acquisition.data)!r}: {error}."
            ]
        ) from error

    acquisition_id = _prefixed_digest(
        "bsa-acq-v2",
        (
            acquisition.identity_key.casefold(),
            acquisition.format_name.upper(),
            acquisition.sensor.upper(),
            data_file.name,
            dark_file.name,
            white_file.name,
            str(data_size_bytes),
        ),
    )
    return AcquisitionProvenance(
        acquisition_id=acquisition_id,
        sample_id=acquisition.identity,
        identity_key=acquisition.identity_key.casefold(),
        sensor=acquisition.sensor.upper(),
        file_format=acquisition.format_name.upper(),
        data_file=data_file.name,
        dark_file=dark_file.name,
        white_file=white_file.name,
        data_size_bytes=str(data_size_bytes),
    )


def dataset_provenance(
    dataset: DiscoveredDataset | Iterable[DatasetSetPaths],
) -> tuple[str, tuple[AcquisitionProvenance, ...]]:
    """Return an order-independent dataset ID and its acquisition records."""

    records = tuple(acquisition_provenance(item) for item in _sets_from_dataset(dataset))
    issues: list[str] = []
    if not records:
        issues.append("The selected dataset contains no validated acquisitions.")
    by_key: dict[str, list[AcquisitionProvenance]] = {}
    by_id: dict[str, list[AcquisitionProvenance]] = {}
    for record in records:
        by_key.setdefault(record.identity_key.casefold(), []).append(record)
        by_id.setdefault(record.acquisition_id, []).append(record)
    for key, matches in by_key.items():
        if len(matches) > 1:
            issues.append(
                f"Ambiguous acquisition identity {key!r} occurs {len(matches)} times in the selected dataset."
            )
    for acquisition_id, matches in by_id.items():
        if len(matches) > 1:
            issues.append(
                f"Duplicate acquisition ID {acquisition_id!r} occurs {len(matches)} times in the selected dataset."
            )
    if issues:
        raise LabelManifestError(issues)
    dataset_id = _prefixed_digest(
        "bsa-dataset-v2", sorted(record.acquisition_id for record in records)
    )
    return dataset_id, records


def _record_from_provenance(
    dataset_id: str,
    provenance: AcquisitionProvenance,
    class_name: str | None,
) -> LabelManifestRecord:
    return LabelManifestRecord(
        manifest_version=MANIFEST_VERSION,
        dataset_id=dataset_id,
        acquisition_id=provenance.acquisition_id,
        sample_id=provenance.sample_id,
        identity_key=provenance.identity_key,
        sensor=provenance.sensor,
        file_format=provenance.file_format,
        data_file=provenance.data_file,
        dark_file=provenance.dark_file,
        white_file=provenance.white_file,
        data_size_bytes=provenance.data_size_bytes,
        label_status=UNLABELLED if class_name is None else LABELLED,
        class_name="" if class_name is None else class_name,
    )


def create_label_manifest(
    dataset: DiscoveredDataset | Iterable[DatasetSetPaths],
    assignments: Mapping[str, str | None] | None = None,
) -> LabelManifest:
    """Create one row per acquisition from acquisition-ID keyed assignments.

    ``None`` means explicitly unlabelled.  Omitted acquisition IDs also become
    explicit unlabelled rows; no class is ever inferred from a filename.
    """

    assignments = {} if assignments is None else dict(assignments)
    dataset_id, provenance = dataset_provenance(dataset)
    known_ids = {record.acquisition_id for record in provenance}
    issues = [
        f"Assignment refers to unknown acquisition ID {key!r}."
        for key in assignments
        if key not in known_ids
    ]
    for key, value in assignments.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            issues.append(f"Assignment for acquisition {key!r} has an empty class value.")
    if issues:
        raise LabelManifestError(issues)
    manifest = LabelManifest(
        dataset_id=dataset_id,
        records=tuple(
            _record_from_provenance(
                dataset_id, record, assignments.get(record.acquisition_id)
            )
            for record in provenance
        ),
    )
    _validate_manifest_structure(manifest)
    return manifest


def _validate_manifest_structure(manifest: LabelManifest) -> None:
    issues: list[str] = []
    if not _DATASET_ID_PATTERN.fullmatch(manifest.dataset_id):
        issues.append(f"Invalid dataset ID {manifest.dataset_id!r}.")
    if not manifest.records:
        issues.append("The label manifest contains no acquisition records.")
    if len(manifest.records) > MAX_MANIFEST_ROWS:
        issues.append(f"The label manifest exceeds {MAX_MANIFEST_ROWS:,} rows.")

    records_by_id: dict[str, list[LabelManifestRecord]] = {}
    records_by_identity: dict[str, list[LabelManifestRecord]] = {}
    for index, record in enumerate(manifest.records, start=2):
        line = record.source_line or index
        for field_name in MANIFEST_COLUMNS:
            value = record.class_name if field_name == "class" else getattr(record, field_name)
            if "\x00" in value:
                issues.append(f"Line {line}: NUL characters are unsupported.")
                break
            if len(value) > MAX_MANIFEST_FIELD_CHARS:
                issues.append(f"Line {line}: {field_name} exceeds the character limit.")
                break
        if record.manifest_version != MANIFEST_VERSION:
            issues.append(
                f"Line {line}: unsupported manifest version {record.manifest_version!r}."
            )
        if record.dataset_id != manifest.dataset_id:
            issues.append(f"Line {line}: inconsistent dataset ID {record.dataset_id!r}.")
        if not _ACQUISITION_ID_PATTERN.fullmatch(record.acquisition_id):
            issues.append(f"Line {line}: invalid acquisition ID {record.acquisition_id!r}.")
        if not record.sample_id.strip():
            issues.append(f"Line {line}: missing sample ID.")
        if not record.identity_key.strip():
            issues.append(f"Line {line}: missing identity key.")
        if record.sensor not in {"VNIR", "SWIR", "OTHER"}:
            issues.append(f"Line {line}: invalid sensor {record.sensor!r}.")
        if record.file_format not in {"BIL", "RAW"}:
            issues.append(f"Line {line}: invalid file format {record.file_format!r}.")
        elif record.sample_id.strip() and record.identity_key.strip():
            expected_key = f"{record.file_format.casefold()}:{record.sample_id.casefold()}"
            if record.identity_key.casefold() != expected_key:
                issues.append(
                    f"Line {line}: identity key {record.identity_key!r} does not match "
                    f"sample ID {record.sample_id!r} and format {record.file_format!r}."
                )
        if record.file_format == "RAW" and record.sensor != "VNIR":
            issues.append(f"Line {line}: RAW acquisitions must use the VNIR sensor.")
        for field_name in ("data_file", "dark_file", "white_file"):
            if not getattr(record, field_name).strip():
                issues.append(f"Line {line}: missing {field_name.replace('_', ' ')}.")
        if not _FILE_SIZE_PATTERN.fullmatch(record.data_size_bytes):
            issues.append(f"Line {line}: invalid data file size.")
        if record.label_status == LABELLED:
            if not record.class_name.strip():
                issues.append(f"Line {line}: labelled sample has an empty class value.")
        elif record.label_status == UNLABELLED:
            if record.class_name:
                issues.append(
                    f"Line {line}: explicitly unlabelled sample must have an empty class value."
                )
        else:
            issues.append(f"Line {line}: invalid label status {record.label_status!r}.")
        records_by_id.setdefault(record.acquisition_id, []).append(record)
        records_by_identity.setdefault(record.identity_key.casefold(), []).append(record)

    for acquisition_id, records in records_by_id.items():
        if len(records) > 1:
            assignments = {(item.label_status, item.class_name) for item in records}
            issues.append(f"Duplicate acquisition ID {acquisition_id!r} in the manifest.")
            if len(assignments) > 1:
                issues.append(
                    f"Conflicting class assignments for acquisition ID {acquisition_id!r}."
                )
    for identity, records in records_by_identity.items():
        if len(records) > 1:
            issues.append(f"Duplicate sample identity {identity!r} in the manifest.")
    if issues:
        raise LabelManifestError(issues)


def save_label_manifest(
    path: os.PathLike[str] | str,
    manifest: LabelManifest,
) -> None:
    """Validate and publish a UTF-8 CSV manifest without replacement."""

    _validate_manifest_structure(manifest)
    destination = Path(path)
    planned = PlannedOutput(manifest.dataset_id, "canonical label manifest", destination)
    validate_output_plan((planned,))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_name = output.name
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(MANIFEST_COLUMNS)
            for record in manifest.records:
                writer.writerow(
                    (
                        record.manifest_version,
                        record.dataset_id,
                        record.acquisition_id,
                        record.sample_id,
                        record.identity_key,
                        record.sensor,
                        record.file_format,
                        record.data_file,
                        record.dark_file,
                        record.white_file,
                        record.data_size_bytes,
                        record.label_status,
                        record.class_name,
                    )
                )
        if os.stat(temporary_name).st_size > MAX_MANIFEST_BYTES:
            raise LabelManifestError(["The label manifest exceeds the 64 MiB file limit."])
        publish_staged_outputs(((planned, Path(temporary_name)),))
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_manifest_review_xlsx(
    path: os.PathLike[str] | str,
    manifest: LabelManifest,
) -> None:
    """Write a human-review workbook; it is never an analysis input."""

    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.utils.exceptions import IllegalCharacterError

    _validate_manifest_structure(manifest)
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("Label review")

    def literal_row(values: Iterable[str]) -> list[WriteOnlyCell]:
        cells = []
        for value in values:
            cell = WriteOnlyCell(sheet, value=value)
            cell.data_type = "s"
            cells.append(cell)
        return cells

    destination = Path(path)
    planned = PlannedOutput(manifest.dataset_id, "spreadsheet review", destination)
    validate_output_plan((planned,))
    temporary_name: str | None = None
    try:
        sheet.append(literal_row(MANIFEST_COLUMNS))
        for record in manifest.records:
            sheet.append(literal_row(
                record.class_name if name == "class" else getattr(record, name)
                for name in MANIFEST_COLUMNS
            ))
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".xlsx",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        workbook.save(temporary_name)
        publish_staged_outputs(((planned, Path(temporary_name)),))
    except IllegalCharacterError as error:
        raise LabelManifestError([
            "A manifest text value contains a character unsupported by XLSX review export."
        ]) from error
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_label_manifest(
    path: os.PathLike[str] | str,
    *,
    _expected_dataset_id: str | None = None,
    _known_acquisition_ids: set[str] | None = None,
) -> LabelManifest:
    """Stream and validate a bounded UTF-8 version-2 CSV manifest."""

    try:
        # Resolve and reject special files before opening. The parser consumes
        # this exact opened target, so later symlink changes cannot redirect it.
        with open_regular_file(path, "The label manifest") as (source, resolved):
            if resolved.identity.size == 0:
                raise LabelManifestError(["The label manifest CSV is empty."])
            if resolved.identity.size > MAX_MANIFEST_BYTES:
                raise LabelManifestError(["The label manifest exceeds the 64 MiB file limit."])
            with io.TextIOWrapper(
                io.BufferedReader(_BoundedManifestReader(source)),
                encoding="utf-8-sig",
                newline="",
            ) as text_source:
                previous_limit = csv.field_size_limit()
                csv.field_size_limit(MAX_MANIFEST_FIELD_CHARS)
                try:
                    reader = csv.reader(_bounded_lines(text_source), strict=True)
                    header_row = next(reader, None)
                    if header_row is None:
                        raise LabelManifestError(["The label manifest CSV is empty."])
                    header = tuple(header_row)
                    if header == _LEGACY_MANIFEST_COLUMNS_V1:
                        raise LabelManifestError([
                            "Invalid label manifest header: version 1 manifests are incompatible because their acquisition "
                            "and dataset IDs depend on mask filenames; recreate the manifest with version 2."
                        ])
                    if len(header) > MAX_MANIFEST_COLUMNS:
                        raise LabelManifestError(["Invalid label manifest header: excessive columns."])
                    if header != MANIFEST_COLUMNS:
                        if len(header) == 1:
                            detail = "one-column positional label files are not identity-validated manifests"
                        elif len(set(header)) != len(header):
                            detail = "duplicate column names"
                        else:
                            detail = f"expected columns {list(MANIFEST_COLUMNS)!r}"
                        raise LabelManifestError([f"Invalid label manifest header: {detail}."])

                    records: list[LabelManifestRecord] = []
                    seen_ids: set[str] = set()
                    seen_identities: set[str] = set()
                    dataset_id: str | None = None
                    for row in reader:
                        line = reader.line_num
                        if len(records) >= MAX_MANIFEST_ROWS:
                            raise LabelManifestError([f"Line {line}: manifest exceeds {MAX_MANIFEST_ROWS:,} rows."])
                        if len(row) > MAX_MANIFEST_COLUMNS:
                            raise LabelManifestError([f"Line {line}: excessive field count."])
                        if len(row) != len(MANIFEST_COLUMNS):
                            raise LabelManifestError([
                                f"Line {line}: expected {len(MANIFEST_COLUMNS)} fields, found {len(row)}."
                            ])
                        if any(len(field) > MAX_MANIFEST_FIELD_CHARS for field in row):
                            raise LabelManifestError([f"Line {line}: field exceeds the character limit."])
                        if any("\x00" in field for field in row):
                            raise LabelManifestError([f"Line {line}: NUL characters are unsupported."])
                        values = dict(zip(MANIFEST_COLUMNS, row))
                        if dataset_id is None:
                            dataset_id = values["dataset_id"]
                        record = LabelManifestRecord(
                            manifest_version=values["manifest_version"],
                            dataset_id=values["dataset_id"],
                            acquisition_id=values["acquisition_id"],
                            sample_id=values["sample_id"],
                            identity_key=values["identity_key"],
                            sensor=values["sensor"],
                            file_format=values["file_format"],
                            data_file=values["data_file"],
                            dark_file=values["dark_file"],
                            white_file=values["white_file"],
                            data_size_bytes=values["data_size_bytes"],
                            label_status=values["label_status"],
                            class_name=values["class"],
                            source_line=line,
                        )
                        if record.acquisition_id in seen_ids:
                            raise LabelManifestError([f"Duplicate acquisition ID {record.acquisition_id!r} in the manifest."])
                        identity = record.identity_key.casefold()
                        if identity in seen_identities:
                            raise LabelManifestError([f"Duplicate sample identity {identity!r} in the manifest."])
                        _validate_manifest_structure(LabelManifest(dataset_id, (record,)))
                        if _expected_dataset_id is not None and dataset_id != _expected_dataset_id:
                            raise LabelManifestError([
                                f"Manifest dataset ID {dataset_id!r} does not match the selected dataset {_expected_dataset_id!r}."
                            ])
                        if _known_acquisition_ids is not None and record.acquisition_id not in _known_acquisition_ids:
                            raise LabelManifestError([
                                f"Unknown or unmatched acquisition {record.acquisition_id!r} ({record.sample_id!r}) is not in the selected dataset."
                            ])
                        seen_ids.add(record.acquisition_id)
                        seen_identities.add(identity)
                        records.append(record)
                    manifest = LabelManifest(dataset_id=dataset_id or "", records=tuple(records))
                    _validate_manifest_structure(manifest)
                    if FileIdentity.from_stat(os.fstat(source.fileno())) != resolved.identity:
                        raise LabelManifestError(["The label manifest changed while it was being parsed."])
                    return manifest
                finally:
                    csv.field_size_limit(previous_limit)
    except ReadPathError as error:
        raise LabelManifestError([str(error)]) from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise LabelManifestError([f"Cannot read label manifest CSV: {error}."]) from error


def validate_label_manifest(
    manifest: LabelManifest,
    dataset: DiscoveredDataset | Iterable[DatasetSetPaths],
) -> ValidatedLabelAssignments:
    """Match rows to the selected dataset by identity, never by row position."""

    _validate_manifest_structure(manifest)
    expected_dataset_id, expected_records = dataset_provenance(dataset)
    expected_by_id = {record.acquisition_id: record for record in expected_records}
    actual_by_id = {record.acquisition_id: record for record in manifest.records}
    issues: list[str] = []
    if manifest.dataset_id != expected_dataset_id:
        issues.append(
            f"Manifest dataset ID {manifest.dataset_id!r} does not match the selected dataset {expected_dataset_id!r}."
        )

    provenance_fields = (
        "sample_id",
        "identity_key",
        "sensor",
        "file_format",
        "data_file",
        "dark_file",
        "white_file",
        "data_size_bytes",
    )
    for acquisition_id, record in actual_by_id.items():
        expected = expected_by_id.get(acquisition_id)
        if expected is None:
            issues.append(
                f"Unknown or unmatched acquisition {acquisition_id!r} ({record.sample_id!r}) is not in the selected dataset."
            )
            continue
        for field_name in provenance_fields:
            actual_value = getattr(record, field_name)
            expected_value = getattr(expected, field_name)
            if actual_value != expected_value:
                issues.append(
                    f"Acquisition {acquisition_id!r} has mismatched {field_name.replace('_', ' ')}: "
                    f"manifest {actual_value!r}, dataset {expected_value!r}."
                )
    for acquisition_id, expected in expected_by_id.items():
        if acquisition_id not in actual_by_id:
            issues.append(
                f"Selected-dataset sample {expected.sample_id!r} ({acquisition_id}) is missing from the manifest."
            )
    if issues:
        raise LabelManifestError(issues)

    labelled: dict[str, str] = {}
    by_identity: dict[str, str] = {}
    unlabelled: list[str] = []
    for acquisition_id, record in actual_by_id.items():
        if record.label_status == LABELLED:
            labelled[acquisition_id] = record.class_name
            by_identity[record.identity_key.casefold()] = record.class_name
        else:
            unlabelled.append(acquisition_id)
    return ValidatedLabelAssignments(
        classes_by_acquisition_id=labelled,
        classes_by_identity_key=by_identity,
        unlabelled_acquisition_ids=tuple(sorted(unlabelled)),
        dataset_id=expected_dataset_id,
    )


def load_and_validate_label_manifest(
    path: os.PathLike[str] | str,
    dataset: DiscoveredDataset | Iterable[DatasetSetPaths],
) -> ValidatedLabelAssignments:
    """Convenience integration contract shared by both BSA applications."""

    expected_dataset_id, expected_records = dataset_provenance(dataset)
    return validate_label_manifest(
        load_label_manifest(
            path,
            _expected_dataset_id=expected_dataset_id,
            _known_acquisition_ids={record.acquisition_id for record in expected_records},
        ),
        dataset,
    )
