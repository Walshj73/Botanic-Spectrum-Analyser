"""Researcher-facing HDR Creator controls and wavelength calibration preview."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from bsa.hyperspectral import hdr


IMPORT_MODE = "Import wavelength grid"
UNIFORM_MODE = "Uniform wavelength range"


class HDRCreatorPanel(ttk.Frame):
    """Create ENVI headers from explicit per-band calibration or uniform spacing."""

    def __init__(self, parent: tk.Misc, root: tk.Tk,
                 open_folder: Callable[[str], object]) -> None:
        super().__init__(parent, padding=8)
        self.root = root
        self.open_folder = open_folder
        self.output_folder = ""
        self.imported_grid: tuple[float, ...] | None = None
        self.imported_source = ""
        self.imported_units = "Unknown"
        self.mode_var = tk.StringVar(master=root, value=IMPORT_MODE)
        self.variables = {name: tk.StringVar(master=root) for name in (
            "samples", "lines", "bands", "data_type", "byte_order",
            "first", "last", "filename",
        )}
        self.entries: dict[str, tk.Entry] = {}
        self.grid_columnconfigure(1, weight=1)

        self.status = ttk.Label(self, text="Choose a measured wavelength grid, then enter the BIL metadata.")
        self.status.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))

        integer_check = (root.register(hdr.integer_entry_text), "%P")
        decimal_check = (root.register(hdr.decimal_entry_text), "%P")
        order_check = (root.register(hdr.byte_order_entry_text), "%P")

        def entry(row: int, name: str, label: str, check: tuple[str, str] | None = None) -> tk.Entry:
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
            options = {"validate": "key", "validatecommand": check} if check else {}
            control = tk.Entry(self, textvariable=self.variables[name], **options)
            control.grid(row=row, column=1, sticky="ew", pady=2)
            self.entries[name] = control
            return control

        entry(1, "samples", "Samples (columns):", integer_check)
        entry(2, "lines", "Lines (rows):", integer_check)
        entry(3, "bands", "Spectral bands:", integer_check)
        entry(4, "data_type", "ENVI data type code (12 = uint16):", integer_check)
        entry(5, "byte_order", "Byte order (0 = little, 1 = big):", order_check)

        ttk.Label(self, text="Wavelength mode:").grid(row=6, column=0, sticky="w")
        self.mode_combo = ttk.Combobox(
            self, textvariable=self.mode_var, values=(IMPORT_MODE, UNIFORM_MODE), state="readonly"
        )
        self.mode_combo.grid(row=6, column=1, sticky="ew", pady=2)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh())
        ttk.Label(self, text="Import is recommended for measured camera calibration.",
                  wraplength=210).grid(row=6, column=2, sticky="w", padx=8)

        entry(7, "first", "Uniform first wavelength (nm):", decimal_check)
        entry(8, "last", "Uniform last wavelength (nm):", decimal_check)
        ttk.Label(self, text="Uniform mode assumes equal spacing between endpoints.",
                  wraplength=340).grid(row=8, column=2, sticky="w", padx=8)
        entry(9, "filename", "Output filename:")

        ttk.Label(self, text="Folder to save:").grid(row=10, column=0, sticky="w")
        self.folder_label = ttk.Label(self, text="No folder selected", wraplength=340)
        self.folder_label.grid(row=10, column=1, sticky="w")
        self.browse_button = ttk.Button(self, text="Browse...", command=self.browse_output)
        self.browse_button.grid(row=10, column=2, sticky="w", padx=8)

        ttk.Label(self, text="Camera wavelength grid:").grid(row=11, column=0, sticky="w")
        self.source_label = ttk.Label(self, text="No grid imported", wraplength=340)
        self.source_label.grid(row=11, column=1, sticky="w")
        self.import_button = ttk.Button(self, text="Import HDR / CSV / TXT...", command=self.browse_grid)
        self.import_button.grid(row=11, column=2, sticky="w", padx=8)

        preview_frame = ttk.LabelFrame(self, text="Band index and wavelength preview")
        preview_frame.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=4)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_summary = ttk.Label(preview_frame, text="No grid selected")
        self.preview_summary.grid(row=0, column=0, sticky="w")
        self.preview = ttk.Treeview(
            preview_frame, columns=("band", "wavelength"), show="headings", height=4
        )
        self.preview.heading("band", text="Band")
        self.preview.heading("wavelength", text="Wavelength")
        self.preview.column("band", width=70, anchor="e", stretch=False)
        self.preview.column("wavelength", width=150, anchor="e")
        self.preview.grid(row=1, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=scrollbar.set)

        self.start_button = ttk.Button(self, text="Start HDR creator", command=self.submit)
        self.start_button.grid(row=13, column=1, sticky="w", pady=4)

        for variable in self.variables.values():
            variable.trace_add("write", lambda *_args: self._refresh())
        self._refresh()

    def browse_output(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="Select HDR output folder")
        if selected:
            self.output_folder = selected
            self.folder_label.configure(text=selected)
            self._refresh()

    def browse_grid(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root, title="Select camera wavelength calibration",
            filetypes=(("ENVI headers", "*.hdr"), ("Wavelength CSV", "*.csv"),
                       ("Wavelength text", "*.txt"), ("All files", "*.*")),
        )
        if selected:
            self.import_grid(selected)

    def import_grid(self, path: str) -> bool:
        """Import only wavelength values; retain all other Creator metadata fields."""

        try:
            values, units = hdr.load_wavelength_calibration(path)
        except (ValueError, OSError) as error:
            self.imported_grid = None
            self.imported_source = ""
            self.imported_units = "Unknown"
            self._refresh()
            messagebox.showerror("HDR Creator", str(error), parent=self.root)
            return False
        self.imported_grid = values
        self.imported_source = str(path)
        self.imported_units = units
        self.mode_var.set(IMPORT_MODE)
        self._refresh()
        if self.variables["bands"].get().strip() and len(values) != self._band_count():
            messagebox.showerror(
                "HDR Creator",
                f"Imported {len(values)} wavelengths but Bands is {self.variables['bands'].get()}. "
                "Change Bands or import a matching grid.", parent=self.root,
            )
            return False
        return True

    def _band_count(self) -> int | None:
        value = self.variables["bands"].get()
        return int(value) if len(value) <= hdr.MAX_CREATOR_INTEGER_DIGITS and value.isascii() and value.isdecimal() else None

    def _set_preview(self, values: tuple[float, ...], source: str, units: str) -> None:
        self.preview.delete(*self.preview.get_children())
        count = len(values)
        self.preview_summary.configure(
            text=f"{source}: {count} bands; first {repr(values[0])}; "
                 f"last {repr(values[-1])}; units: {units}"
        )
        self.preview.heading("wavelength", text=f"Wavelength ({units})")
        # Real camera grids fit in full; keep huge valid grids responsive.
        if count <= 2_000:
            indices = range(count)
        else:
            indices = (*range(1_000), *range(count - 1_000, count))
        for index in indices:
            self.preview.insert("", "end", values=(index, repr(values[index])))
        if count > 2_000:
            self.preview_summary.configure(
                text=self.preview_summary.cget("text") + "; preview shows first/last 1,000"
            )

    def _refresh(self) -> None:
        imported = self.mode_var.get() == IMPORT_MODE
        for name in ("first", "last"):
            self.entries[name].configure(state="disabled" if imported else "normal")
        self.import_button.configure(state="normal" if imported else "disabled")
        count = self._band_count()
        if imported:
            grid = self.imported_grid
            self.source_label.configure(text=(Path(self.imported_source).name if grid else "No grid imported"))
            if grid:
                self._set_preview(grid, f"Imported {Path(self.imported_source).name}", self.imported_units)
            else:
                self.preview.delete(*self.preview.get_children())
                self.preview_summary.configure(text="Import a complete camera wavelength grid")
            grid_valid = grid is not None and count == len(grid)
            if grid is None:
                reason = "Import a camera wavelength grid to create this header."
            elif count != len(grid):
                reason = f"Imported grid has {len(grid)} wavelengths; Bands must match."
            else:
                reason = "Imported calibration is ready and retained for the next header."
        else:
            self.source_label.configure(text="Uniform spacing from entered endpoints")
            grid_valid = bool(self.variables["first"].get().strip()
                              and self.variables["last"].get().strip())
            reason = "Uniform mode assumes equal wavelength spacing; it is not camera calibration."
            self.preview.delete(*self.preview.get_children())
            try:
                if count is not None and 2 <= count <= hdr.MAX_CREATOR_BANDS and grid_valid:
                    first = self.variables["first"].get()
                    last = self.variables["last"].get()
                    if count <= 2_000:
                        grid = tuple(hdr.generate_wavelengths(first, last, count))
                        self._set_preview(grid, "Uniform spacing", "nm")
                    else:
                        self.preview_summary.configure(
                            text=f"Uniform spacing: {count} bands; first {first} nm; last {last} nm"
                        )
                else:
                    self.preview_summary.configure(text="Enter a valid uniform range for preview")
            except ValueError:
                self.preview_summary.configure(text="Uniform endpoints need correction")
        required = ("samples", "lines", "bands", "data_type", "byte_order", "filename")
        complete = bool(self.output_folder) and all(
            self.variables[name].get().strip() for name in required
        )
        self.start_button.configure(state="normal" if complete and grid_valid else "disabled")
        self.status.configure(text=reason if complete else "Complete all required fields and choose an output folder. " + reason)

    def submit(self) -> bool:
        imported = self.mode_var.get() == IMPORT_MODE
        if imported and (self.imported_grid is None or self._band_count() != len(self.imported_grid)):
            messagebox.showerror("HDR Creator", "Import a grid whose count matches Bands.", parent=self.root)
            return False
        try:
            output = hdr.write_envi_header(
                self.variables["samples"].get(), self.variables["lines"].get(),
                self.variables["bands"].get(), self.variables["data_type"].get(),
                self.variables["byte_order"].get(),
                None if imported else (self.variables["first"].get(), self.variables["last"].get()),
                self.variables["filename"].get(), self.output_folder,
                wavelengths=self.imported_grid if imported else None,
                wavelength_units=self.imported_units if imported else "Unknown",
            )
        except (ValueError, OSError) as error:
            self.status.configure(text="HDR Creator failed; correct the fields and try again.")
            messagebox.showerror("HDR Creator", str(error), parent=self.root)
            return False
        self.status.configure(text=f"HDR ready: {output}")
        self.open_folder(self.output_folder)
        return True
