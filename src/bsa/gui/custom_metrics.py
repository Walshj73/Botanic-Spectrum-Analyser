"""Compact Tk editor for the analyser's ordered custom-metric queue."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from bsa.analysis.custom_metric_definitions import (
    CustomMetricDefinition,
    CustomMetricQueue,
)
from bsa.hyperspectral.io import extract_band_count, extract_wavelengths


def band_display(index: int, wavelengths: tuple[float, ...]) -> str:
    if index < len(wavelengths):
        return f"{index} — {wavelengths[index]:.2f} nm"
    return str(index)


class CustomMetricEditor:
    """All widget actions remain on Tk's main thread."""

    def __init__(self, parent: tk.Misc, root: tk.Misc):
        self.root = root
        self.queue = CustomMetricQueue()
        self.selected_index: int | None = None
        self.frame = ttk.LabelFrame(parent, text="Custom metrics (optional)", padding=(14, 10))
        self.frame.pack(fill="x", padx=16, pady=6)

        ttk.Label(self.frame, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Label(self.frame, text="Formula (Var1, Var2, Var3)").grid(row=0, column=1, columnspan=3, sticky="w")
        self.name_entry = ttk.Entry(self.frame, width=19)
        self.formula_entry = ttk.Entry(self.frame, width=55)
        self.name_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.formula_entry.grid(row=1, column=1, columnspan=3, sticky="ew")

        self.selectors: dict[str, ttk.Combobox] = {}
        for column, variable in enumerate(("Var1", "Var2", "Var3"), start=1):
            ttk.Label(self.frame, text=f"{variable} band (0-based)").grid(
                row=2, column=column, sticky="w", pady=(4, 0)
            )
            selector = ttk.Combobox(self.frame, width=19, state="normal")
            selector.grid(row=3, column=column, sticky="ew", padx=(0, 4))
            self.selectors[variable] = selector

        buttons = ttk.Frame(self.frame)
        buttons.grid(row=4, column=0, columnspan=4, sticky="w", pady=(5, 3))
        ttk.Button(buttons, text="Add / Update Metric", command=self.add_or_update).pack(side="left")
        ttk.Button(buttons, text="Remove Metric", command=self.remove).pack(side="left", padx=5)
        ttk.Button(buttons, text="Clear Metrics", command=self.clear).pack(side="left")

        columns = ("Name", "Formula", "Var1", "Var2", "Var3")
        self.table = ttk.Treeview(self.frame, columns=columns, show="headings", height=3, selectmode="browse")
        for name, width in (("Name", 120), ("Formula", 290), ("Var1", 105), ("Var2", 105), ("Var3", 105)):
            self.table.heading(name, text=name)
            self.table.column(name, width=width, minwidth=70, stretch=name == "Formula")
        self.table.grid(row=5, column=0, columnspan=4, sticky="ew")
        self.table.bind("<<TreeviewSelect>>", self._selection_changed)
        self.status = ttk.Label(self.frame, text="Add metrics before starting analysis.")
        self.status.grid(row=6, column=0, columnspan=4, sticky="w", pady=(3, 0))
        self.frame.columnconfigure(0, weight=1)
        for column in (1, 2, 3):
            self.frame.columnconfigure(column, weight=2)

    @property
    def metrics(self) -> tuple[CustomMetricDefinition, ...]:
        return self.queue.metrics

    def _show_error(self, error: Exception) -> None:
        messagebox.showerror("Custom metric validation failed", str(error), parent=self.root)

    def _index_from_selector(self, variable: str) -> int | str | None:
        value = self.selectors[variable].get().strip()
        if not value:
            return None
        if " — " in value:
            prefix, _, _wavelength = value.partition(" — ")
            index = int(prefix)
            if value != band_display(index, self.queue.wavelengths):
                raise ValueError(f"{variable} must be selected from the current spectral grid.")
            return index
        return value

    def _set_selector(self, variable: str, index: int | None) -> None:
        selector = self.selectors[variable]
        selector.set("" if index is None else band_display(index, self.queue.wavelengths))

    def _clear_editor(self) -> None:
        self.selected_index = None
        self.name_entry.delete(0, tk.END)
        self.formula_entry.delete(0, tk.END)
        for variable in self.selectors:
            self._set_selector(variable, None)
        self.table.selection_remove(self.table.selection())

    def _refresh_table(self) -> None:
        self.table.delete(*self.table.get_children())
        for index, metric in enumerate(self.queue.metrics):
            self.table.insert("", "end", iid=str(index), values=(
                metric.name, metric.formula,
                *(band_display(value, self.queue.wavelengths) if value is not None else ""
                  for value in (metric.var1, metric.var2, metric.var3)),
            ))
        self.status.configure(text=f"{len(self.queue.metrics)} custom metric(s) queued for analysis.")

    def _selection_changed(self, _event=None) -> None:
        selected = self.table.selection()
        if not selected:
            return
        self.selected_index = int(selected[0])
        metric = self.queue.metrics[self.selected_index]
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, metric.name)
        self.formula_entry.delete(0, tk.END)
        self.formula_entry.insert(0, metric.formula)
        for variable, value in (("Var1", metric.var1), ("Var2", metric.var2), ("Var3", metric.var3)):
            self._set_selector(variable, value)

    def add_or_update(self) -> None:
        try:
            selections = {name: self._index_from_selector(name) for name in self.selectors}
            self.queue.add_or_update(
                self.name_entry.get(), self.formula_entry.get(), selections,
                self.selected_index,
            )
        except ValueError as error:
            self._show_error(error)
            return
        self._clear_editor()
        self._refresh_table()

    def remove(self) -> None:
        if self.selected_index is None:
            self._show_error(ValueError("Select a queued metric to remove."))
            return
        self.queue.remove(self.selected_index)
        self._clear_editor()
        self._refresh_table()

    def clear(self) -> None:
        self.queue.clear()
        self._clear_editor()
        self._refresh_table()

    def set_busy(self, busy: bool) -> None:
        """Keep an active analysis's metric snapshot fixed while allowing tab changes."""

        for widget in (self.name_entry, self.formula_entry):
            widget.configure(state="disabled" if busy else "normal")
        for widget in self.selectors.values():
            widget.configure(state="disabled" if busy else "readonly" if widget["values"] else "normal")
        for child in self.frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for button in child.winfo_children():
                    button.configure(state="disabled" if busy else "normal")
        self.table.configure(selectmode="none" if busy else "browse")

    def other_header_changed(self) -> None:
        self.clear()
        self.status.configure(text="HDR selection changed; previous metric mappings were cleared.")

    def data_header_changed(self, path: str | None) -> None:
        count: int | None = None
        wavelengths: tuple[float, ...] = ()
        if path:
            try:
                count = extract_band_count(path)
                parsed = tuple(extract_wavelengths(path))
                if parsed and (count is None or len(parsed) == count):
                    wavelengths = parsed
                    count = len(parsed)
            except (OSError, UnicodeError, ValueError):
                pass  # Metadata display is optional; ENVI loading validates later.
        changed = self.queue.set_grid(path, count, wavelengths)
        options = tuple(band_display(index, wavelengths) for index in range(count or 0))
        for selector in self.selectors.values():
            selector.configure(values=options, state="readonly" if options else "normal")
        if changed:
            self._clear_editor()
            self._refresh_table()
            self.status.configure(text="Spectral grid changed; previous metric mappings were cleared.")
        elif count is not None:
            self.status.configure(text=f"{count} bands available; {len(self.metrics)} metric(s) queued.")
