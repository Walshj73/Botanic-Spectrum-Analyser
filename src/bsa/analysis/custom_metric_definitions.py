"""Ordered, validated custom metrics for one selected spectral definition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from bsa.analysis.custom_indices import validate_custom_index_name
from bsa.utils.expressions import referenced_variables


class CustomMetricDefinitionError(ValueError):
    """A queued metric needs correction before analysis."""


@dataclass(frozen=True)
class CustomMetricDefinition:
    name: str
    formula: str
    var1: int | None = None
    var2: int | None = None
    var3: int | None = None

    def band_mapping(self) -> dict[str, int]:
        return {
            variable: index
            for variable, index in (
                ("Var1", self.var1), ("Var2", self.var2), ("Var3", self.var3)
            )
            if index is not None
        }


class CustomMetricQueue:
    """Keep metric order and discard old mappings when the HDR grid changes."""

    def __init__(self) -> None:
        self._metrics: list[CustomMetricDefinition] = []
        self.band_count: int | None = None
        self.wavelengths: tuple[float, ...] = ()
        self._grid_key: tuple[object, ...] | None = None

    @property
    def metrics(self) -> tuple[CustomMetricDefinition, ...]:
        return tuple(self._metrics)

    def set_grid(
        self,
        header_path: str | None,
        band_count: int | None,
        wavelengths: tuple[float, ...] = (),
    ) -> bool:
        path = str(Path(header_path).resolve()) if header_path else None
        key = (path, band_count, tuple(wavelengths))
        changed = key != self._grid_key and (
            self._grid_key is not None or bool(self._metrics)
        )
        if changed:
            self.clear()
        self._grid_key = key
        self.band_count = band_count
        self.wavelengths = tuple(wavelengths)
        return changed

    def clear(self) -> None:
        self._metrics.clear()

    def remove(self, index: int) -> CustomMetricDefinition:
        return self._metrics.pop(index)

    def add_or_update(
        self,
        name: str,
        formula: str,
        selections: dict[str, int | str | None],
        selected_index: int | None = None,
    ) -> CustomMetricDefinition:
        validate_custom_index_name(name)
        if not formula or not formula.strip():
            raise CustomMetricDefinitionError("A custom metric formula is required.")
        used = referenced_variables(formula)
        if not used:
            raise CustomMetricDefinitionError("The formula must reference at least one band variable.")
        for index, existing in enumerate(self._metrics):
            if index != selected_index and existing.name.strip().casefold() == name.strip().casefold():
                raise CustomMetricDefinitionError(f"Custom metric name {name!r} is already queued.")

        resolved: dict[str, int | None] = {}
        for variable in ("Var1", "Var2", "Var3"):
            raw = selections.get(variable)
            if raw is None or str(raw).strip() == "":
                if variable in used:
                    raise CustomMetricDefinitionError(f"{variable} must be selected for this formula.")
                resolved[variable] = None
                continue
            if isinstance(raw, bool) or re.fullmatch(r"[+-]?\d+", str(raw).strip()) is None:
                raise CustomMetricDefinitionError(
                    f"{variable} must be a zero-based integer band index."
                )
            number = int(str(raw).strip())
            if number < 0 or self.band_count is not None and number >= self.band_count:
                maximum = self.band_count - 1 if self.band_count is not None else "the final band"
                raise CustomMetricDefinitionError(
                    f"{variable} band index {number} is outside 0..{maximum}."
                )
            resolved[variable] = number

        metric = CustomMetricDefinition(
            name=name,
            formula=formula,
            var1=resolved["Var1"],
            var2=resolved["Var2"],
            var3=resolved["Var3"],
        )
        if selected_index is None:
            self._metrics.append(metric)
        else:
            self._metrics[selected_index] = metric
        return metric
