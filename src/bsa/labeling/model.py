"""GUI-independent assignment model for BSA Label Creator.

The model receives canonical acquisition records prepared from Phase 6A/6B.1.
It never parses filenames, scans directories, or opens acquisition files.
Filtering and sorting return views over stable acquisition IDs, so displayed row
positions can never determine which samples receive a class.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
from typing import Iterable

from bsa.hyperspectral.dataset import DiscoveredDataset
from bsa.labeling.manifest import AcquisitionProvenance, ValidatedLabelAssignments


ALL_CLASSES = "__bsa_all_classes__"
UNLABELLED_ONLY = "__bsa_unlabelled_only__"


class LabelCreatorModelError(ValueError):
    """Raised when an assignment operation would be invalid or ambiguous."""


@dataclass(frozen=True)
class AcquisitionItem:
    """Cached metadata needed by the assignment table."""

    acquisition_id: str
    sample_id: str
    identity_key: str
    sensor: str
    file_format: str
    acquisition_filename: str
    source_folder: str


@dataclass(frozen=True)
class AssignmentProposal:
    """Reviewable assignment change that has not yet been applied."""

    acquisition_ids: tuple[str, ...]
    target_class: str | None
    change_count: int
    overwrite_count: int
    description: str

    @property
    def match_count(self) -> int:
        return len(self.acquisition_ids)


@dataclass(frozen=True)
class AssignmentSummary:
    total: int
    labelled: int
    unlabelled: int
    per_class: dict[str, int]


@dataclass(frozen=True)
class _UndoOperation:
    classes_before: tuple[str, ...]
    assignments_before: dict[str, str | None]
    description: str


def acquisition_items_from_dataset(
    dataset: DiscoveredDataset,
    provenance: Iterable[AcquisitionProvenance],
) -> tuple[AcquisitionItem, ...]:
    """Join cached manifest provenance to validated Phase 6A paths by identity."""

    provenance_by_key = {
        record.identity_key.casefold(): record for record in provenance
    }
    if len(provenance_by_key) != dataset.number_of_sets:
        raise LabelCreatorModelError(
            "Validated acquisition provenance contains duplicate identities."
        )

    items: list[AcquisitionItem] = []
    for matched in dataset.matched_sets:
        key = matched.identity_key.casefold()
        record = provenance_by_key.get(key)
        if record is None:
            raise LabelCreatorModelError(
                f"No canonical acquisition ID exists for sample {matched.identity!r}."
            )
        items.append(
            AcquisitionItem(
                acquisition_id=record.acquisition_id,
                sample_id=record.sample_id,
                identity_key=record.identity_key,
                sensor=record.sensor,
                file_format=record.file_format,
                acquisition_filename=record.data_file,
                source_folder=str(Path(matched.data).parent),
            )
        )
    return tuple(items)


class LabelAssignmentModel:
    """Identity-keyed class management, table views, proposals, and undo."""

    def __init__(self, acquisitions: Iterable[AcquisitionItem], undo_limit: int = 100):
        items = tuple(acquisitions)
        by_id: dict[str, AcquisitionItem] = {}
        by_identity: dict[str, AcquisitionItem] = {}
        issues: list[str] = []
        for item in items:
            if not item.acquisition_id:
                issues.append(f"Sample {item.sample_id!r} has no acquisition ID.")
            elif item.acquisition_id in by_id:
                issues.append(f"Duplicate acquisition ID {item.acquisition_id!r}.")
            else:
                by_id[item.acquisition_id] = item
            identity = item.identity_key.casefold()
            if not identity:
                issues.append(f"Sample {item.sample_id!r} has no canonical identity.")
            elif identity in by_identity:
                issues.append(f"Ambiguous acquisition identity {identity!r}.")
            else:
                by_identity[identity] = item
        if issues:
            raise LabelCreatorModelError("\n".join(issues))

        self._items = items
        self._by_id = by_id
        self._assignments: dict[str, str | None] = {
            item.acquisition_id: None for item in items
        }
        self._classes: list[str] = []
        self._undo: list[_UndoOperation] = []
        self._undo_limit = max(1, undo_limit)
        self._revision = 0

    @property
    def acquisitions(self) -> tuple[AcquisitionItem, ...]:
        return self._items

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self._classes)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def undo_description(self) -> str | None:
        return self._undo[-1].description if self._undo else None

    def class_for_id(self, acquisition_id: str) -> str | None:
        try:
            return self._assignments[acquisition_id]
        except KeyError as error:
            raise LabelCreatorModelError(
                f"Unknown acquisition ID {acquisition_id!r}."
            ) from error

    def _class_name(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise LabelCreatorModelError("Class names must contain visible text.")
        return name.strip()

    def _push_undo(
        self,
        classes_before: tuple[str, ...],
        assignments_before: dict[str, str | None],
        description: str,
    ) -> None:
        self._undo.append(
            _UndoOperation(classes_before, assignments_before, description)
        )
        if len(self._undo) > self._undo_limit:
            del self._undo[0]
        self._revision += 1

    def create_class(self, name: str) -> str:
        name = self._class_name(name)
        if name in self._classes:
            raise LabelCreatorModelError(f"Class {name!r} already exists.")
        before = self.classes
        self._classes.append(name)
        self._push_undo(before, {}, f"create class {name!r}")
        return name

    def rename_class(self, old_name: str, new_name: str) -> str:
        if old_name not in self._classes:
            raise LabelCreatorModelError(f"Unknown class {old_name!r}.")
        new_name = self._class_name(new_name)
        if new_name != old_name and new_name in self._classes:
            raise LabelCreatorModelError(f"Class {new_name!r} already exists.")
        if new_name == old_name:
            return new_name
        changed = {
            acquisition_id: assigned
            for acquisition_id, assigned in self._assignments.items()
            if assigned == old_name
        }
        before = self.classes
        index = self._classes.index(old_name)
        self._classes[index] = new_name
        for acquisition_id in changed:
            self._assignments[acquisition_id] = new_name
        self._push_undo(before, changed, f"rename class {old_name!r}")
        return new_name

    def remove_class(self, name: str) -> int:
        if name not in self._classes:
            raise LabelCreatorModelError(f"Unknown class {name!r}.")
        changed = {
            acquisition_id: assigned
            for acquisition_id, assigned in self._assignments.items()
            if assigned == name
        }
        before = self.classes
        self._classes.remove(name)
        for acquisition_id in changed:
            self._assignments[acquisition_id] = None
        self._push_undo(before, changed, f"remove class {name!r}")
        return len(changed)

    def _require_target_class(self, class_name: str | None) -> None:
        if class_name is not None and class_name not in self._classes:
            raise LabelCreatorModelError(f"Unknown class {class_name!r}.")

    def propose_assignment(
        self,
        acquisition_ids: Iterable[str],
        class_name: str | None,
        description: str = "assign selected acquisitions",
    ) -> AssignmentProposal:
        self._require_target_class(class_name)
        unique_ids = tuple(dict.fromkeys(acquisition_ids))
        unknown = [value for value in unique_ids if value not in self._by_id]
        if unknown:
            raise LabelCreatorModelError(
                f"Unknown acquisition ID(s): {', '.join(repr(value) for value in unknown)}."
            )
        change_count = sum(
            self._assignments[value] != class_name for value in unique_ids
        )
        overwrite_count = sum(
            self._assignments[value] is not None
            and self._assignments[value] != class_name
            for value in unique_ids
        )
        return AssignmentProposal(
            acquisition_ids=unique_ids,
            target_class=class_name,
            change_count=change_count,
            overwrite_count=overwrite_count,
            description=description,
        )

    def propose_filename_pattern(
        self, pattern: str, class_name: str
    ) -> AssignmentProposal:
        if not isinstance(pattern, str) or not pattern.strip():
            raise LabelCreatorModelError("A filename pattern is required.")
        pattern = pattern.strip().casefold()
        matches = (
            item.acquisition_id
            for item in self._items
            if fnmatch.fnmatchcase(item.acquisition_filename.casefold(), pattern)
        )
        return self.propose_assignment(
            matches,
            class_name,
            description=f"assign filename pattern {pattern!r}",
        )

    @staticmethod
    def _folder_key(folder: os.PathLike[str] | str) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(folder)))

    def propose_folder(
        self, folder: os.PathLike[str] | str, class_name: str
    ) -> AssignmentProposal:
        if not os.fspath(folder):
            raise LabelCreatorModelError("A source folder is required.")
        key = self._folder_key(folder)
        matches = (
            item.acquisition_id
            for item in self._items
            if self._folder_key(item.source_folder) == key
        )
        return self.propose_assignment(
            matches,
            class_name,
            description=f"assign source folder {os.fspath(folder)!r}",
        )

    def apply(self, proposal: AssignmentProposal) -> int:
        self._require_target_class(proposal.target_class)
        current = self.propose_assignment(
            proposal.acquisition_ids,
            proposal.target_class,
            proposal.description,
        )
        changed = {
            acquisition_id: self._assignments[acquisition_id]
            for acquisition_id in current.acquisition_ids
            if self._assignments[acquisition_id] != current.target_class
        }
        if not changed:
            return 0
        for acquisition_id in changed:
            self._assignments[acquisition_id] = current.target_class
        self._push_undo(self.classes, changed, current.description)
        return len(changed)

    def undo(self) -> str:
        if not self._undo:
            raise LabelCreatorModelError("There is no assignment change to undo.")
        operation = self._undo.pop()
        self._classes = list(operation.classes_before)
        for acquisition_id, class_name in operation.assignments_before.items():
            self._assignments[acquisition_id] = class_name
        self._revision += 1
        return operation.description

    def acquisition_for_id(self, acquisition_id: str) -> AcquisitionItem:
        try:
            return self._by_id[acquisition_id]
        except KeyError as error:
            raise LabelCreatorModelError(
                f"Unknown acquisition ID {acquisition_id!r}."
            ) from error

    def assignment_mapping(self) -> dict[str, str | None]:
        """Return a manifest-ready acquisition-ID mapping."""

        return dict(self._assignments)

    def apply_validated_assignments(
        self, validated: ValidatedLabelAssignments
    ) -> None:
        labelled_ids = set(validated.classes_by_acquisition_id)
        unlabelled_ids = set(validated.unlabelled_acquisition_ids)
        expected_ids = set(self._by_id)
        if labelled_ids | unlabelled_ids != expected_ids:
            raise LabelCreatorModelError(
                "Validated assignments do not cover the current dataset exactly."
            )
        before_assignments = dict(self._assignments)
        before_classes = self.classes
        imported_classes: list[str] = []
        for item in self._items:
            class_name = validated.classes_by_acquisition_id.get(item.acquisition_id)
            self._assignments[item.acquisition_id] = class_name
            if class_name is not None and class_name not in imported_classes:
                imported_classes.append(class_name)
        self._classes = imported_classes
        self._push_undo(before_classes, before_assignments, "import label manifest")

    def summary(self) -> AssignmentSummary:
        per_class = {name: 0 for name in self._classes}
        for class_name in self._assignments.values():
            if class_name is not None:
                per_class[class_name] = per_class.get(class_name, 0) + 1
        labelled = sum(per_class.values())
        return AssignmentSummary(
            total=len(self._items),
            labelled=labelled,
            unlabelled=len(self._items) - labelled,
            per_class=per_class,
        )

    def folders(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_folder for item in self._items}, key=str.casefold))

    def view(
        self,
        search: str = "",
        class_filter: str = ALL_CLASSES,
        sort_column: str = "sample_id",
        descending: bool = False,
    ) -> tuple[AcquisitionItem, ...]:
        if class_filter not in {ALL_CLASSES, UNLABELLED_ONLY} and class_filter not in self._classes:
            raise LabelCreatorModelError(f"Unknown class filter {class_filter!r}.")
        search_key = search.casefold().strip()
        matches: list[AcquisitionItem] = []
        for item in self._items:
            assigned = self._assignments[item.acquisition_id]
            if class_filter == UNLABELLED_ONLY and assigned is not None:
                continue
            if class_filter not in {ALL_CLASSES, UNLABELLED_ONLY} and assigned != class_filter:
                continue
            if search_key and not any(
                search_key in value.casefold()
                for value in (
                    item.sample_id,
                    item.sensor,
                    item.acquisition_filename,
                    assigned or "Unlabelled",
                )
            ):
                continue
            matches.append(item)

        sort_values = {
            "sample_id": lambda item: item.sample_id.casefold(),
            "sensor": lambda item: item.sensor.casefold(),
            "filename": lambda item: item.acquisition_filename.casefold(),
            "class": lambda item: (self._assignments[item.acquisition_id] or "").casefold(),
        }
        try:
            key_function = sort_values[sort_column]
        except KeyError as error:
            raise LabelCreatorModelError(
                f"Unknown table sort column {sort_column!r}."
            ) from error
        return tuple(
            sorted(
                matches,
                key=lambda item: (key_function(item), item.acquisition_id),
                reverse=descending,
            )
        )
