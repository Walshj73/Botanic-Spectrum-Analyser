"""Standalone Tkinter application for identity-safe experimental labelling."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable

from bsa.hyperspectral import dataset as hyperspectral_dataset
from bsa.labeling import manifest as label_manifest
from bsa.labeling.model import (
    ALL_CLASSES,
    UNLABELLED_ONLY,
    AssignmentProposal,
    LabelAssignmentModel,
    LabelCreatorModelError,
    acquisition_items_from_dataset,
)
from bsa.utils.resources import resource_path


UI_FONT_SIZE = 10
UI_SMALL_FONT_SIZE = 9
BODY_FONT_NAME = "BsaLabelCreatorBodyFont"
HEADING_FONT_NAME = "BsaLabelCreatorHeadingFont"
SMALL_FONT_NAME = "BsaLabelCreatorSmallFont"
STYLE_PREFIX = "BsaLabelCreator"
PATTERN_HELP_TEXT = (
    "Use * to match any number of characters and ? to match one character.\n"
    "Example: *GOLDENPROMISE* selects filenames containing GOLDENPROMISE. "
    "Patterns only select acquisitions for class assignment; they do not "
    "rename files and do not infer classes."
)


class LabelCreatorTypographyError(RuntimeError):
    """Raised when Tk cannot render a proportional desktop font."""


def font_is_proportional(font: tkfont.Font) -> bool:
    """Return whether rendered narrow and wide glyphs have different widths."""

    return font.measure("iiiiiiiiii") != font.measure("WWWWWWWWWW")


def resolve_ui_font_family(root: tk.Misc, platform: str | None = None) -> str:
    """Resolve a preferred family and verify what Tk actually renders."""

    platform = sys.platform if platform is None else platform
    if platform == "win32":
        preferred = ("Segoe UI", "Arial", "Noto Sans", "DejaVu Sans")
    elif platform == "darwin":
        preferred = (
            "SF Pro Text",
            "Helvetica Neue",
            "Helvetica",
            "Noto Sans",
            "DejaVu Sans",
        )
    else:
        preferred = ("Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial")

    installed = {family.casefold(): family for family in tkfont.families(root)}
    ordered_candidates = [
        installed.get(requested.casefold(), requested) for requested in preferred
    ]
    for requested in ordered_candidates:
        probe = tkfont.Font(root=root, family=requested, size=UI_FONT_SIZE)
        actual_family = str(probe.actual("family"))
        if font_is_proportional(probe):
            return actual_family

    available = ", ".join(sorted(installed.values(), key=str.casefold)) or "none"
    raise LabelCreatorTypographyError(
        "This Python/Tk runtime cannot render a proportional sans-serif font. "
        f"Tk reports these font families: {available}. On Linux, use a Tk build "
        "with Xft/fontconfig support (for Arch Linux: install the 'tk' package "
        "and launch Label Creator with the system Python)."
    )


def _configure_named_font(
    root: tk.Misc,
    name: str,
    *,
    family: str,
    size: int,
    weight: str = "normal",
) -> tkfont.Font:
    try:
        font = tkfont.nametofont(name, root=root)
    except tk.TclError:
        font = tkfont.Font(root=root, name=name)
    font.configure(family=family, size=size, weight=weight)
    return font


def configure_label_creator_typography(
    root: tk.Misc,
    style: ttk.Style | None = None,
) -> str:
    """Create verified fonts and Label Creator-specific ttk styles."""

    style = ttk.Style(root) if style is None else style
    family = resolve_ui_font_family(root)
    body_font = _configure_named_font(
        root, BODY_FONT_NAME, family=family, size=UI_FONT_SIZE
    )
    heading_font = _configure_named_font(
        root,
        HEADING_FONT_NAME,
        family=family,
        size=UI_FONT_SIZE,
        weight="bold",
    )
    small_font = _configure_named_font(
        root, SMALL_FONT_NAME, family=family, size=UI_SMALL_FONT_SIZE
    )

    # Tk dialogs and any classic child controls use these standard named fonts.
    font_settings = {
        "TkDefaultFont": (UI_FONT_SIZE, "normal"),
        "TkTextFont": (UI_FONT_SIZE, "normal"),
        "TkMenuFont": (UI_FONT_SIZE, "normal"),
        "TkHeadingFont": (UI_FONT_SIZE, "bold"),
        "TkCaptionFont": (UI_FONT_SIZE, "bold"),
        "TkSmallCaptionFont": (UI_SMALL_FONT_SIZE, "normal"),
        "TkIconFont": (UI_FONT_SIZE, "normal"),
        "TkTooltipFont": (UI_SMALL_FONT_SIZE, "normal"),
    }
    for font_name, (size, weight) in font_settings.items():
        try:
            named_font = tkfont.nametofont(font_name, root=root)
        except tk.TclError:
            continue
        named_font.configure(family=family, size=size, weight=weight)

    root.option_add("*Menu.font", BODY_FONT_NAME)
    root.option_add("*Listbox.font", BODY_FONT_NAME)
    root.option_add("*TCombobox*Listbox.font", BODY_FONT_NAME)

    for widget_class in (
        "TButton",
        "TLabel",
        "TEntry",
        "TCombobox",
        "TCheckbutton",
        "TRadiobutton",
        "TMenubutton",
        "TLabelframe.Label",
    ):
        style.configure(f"{STYLE_PREFIX}.{widget_class}", font=BODY_FONT_NAME)
    style.configure(f"{STYLE_PREFIX}.Treeview", font=BODY_FONT_NAME, rowheight=26)
    style.configure(
        f"{STYLE_PREFIX}.Treeview.Heading", font=HEADING_FONT_NAME
    )

    # Keep strong references for the lifetime of the Tk interpreter.
    root._bsa_label_creator_fonts = (body_font, heading_font, small_font)
    return family


def apply_label_creator_typography(widget: tk.Misc) -> None:
    """Assign Label Creator styles to constructed widgets explicitly."""

    style_by_type: tuple[tuple[type[tk.Widget], str], ...] = (
        (ttk.Button, f"{STYLE_PREFIX}.TButton"),
        (ttk.Label, f"{STYLE_PREFIX}.TLabel"),
        (ttk.Entry, f"{STYLE_PREFIX}.TEntry"),
        (ttk.Combobox, f"{STYLE_PREFIX}.TCombobox"),
        (ttk.Checkbutton, f"{STYLE_PREFIX}.TCheckbutton"),
        (ttk.Radiobutton, f"{STYLE_PREFIX}.TRadiobutton"),
        (ttk.Menubutton, f"{STYLE_PREFIX}.TMenubutton"),
        (ttk.LabelFrame, f"{STYLE_PREFIX}.TLabelframe"),
        (ttk.Treeview, f"{STYLE_PREFIX}.Treeview"),
    )
    if isinstance(widget, (tk.Menu, tk.Listbox)):
        widget.configure(font=BODY_FONT_NAME)
    else:
        for widget_type, style_name in style_by_type:
            if isinstance(widget, widget_type):
                widget.configure(style=style_name)
                break
    for child in widget.winfo_children():
        apply_label_creator_typography(child)


def configure_label_creator_icon(
    window: tk.Misc,
    platform: str | None = None,
) -> str | None:
    """Set the BSA icon with native-format preference and graceful fallback."""

    platform = sys.platform if platform is None else platform
    ico_path = resource_path("BSA_logo.ico")
    png_path = resource_path("BSA_logo.png")

    def use_ico() -> bool:
        if not ico_path.is_file():
            return False
        try:
            window.iconbitmap(str(ico_path))
        except (OSError, tk.TclError):
            return False
        return True

    def use_png() -> bool:
        if not png_path.is_file():
            return False
        try:
            image = tk.PhotoImage(master=window, file=str(png_path))
            window.iconphoto(True, image)
        except (OSError, tk.TclError):
            return False
        window._bsa_label_creator_icon = image
        return True

    attempts = ((use_ico, "ico"), (use_png, "png"))
    if platform != "win32":
        attempts = ((use_png, "png"), (use_ico, "ico"))
    for loader, format_name in attempts:
        if loader():
            return format_name
    return None


@dataclass(frozen=True)
class LoadedLabelDataset:
    dataset: hyperspectral_dataset.DiscoveredDataset
    dataset_id: str
    model: LabelAssignmentModel
    image_directory: str
    mask_directory: str | None


def load_label_dataset(
    image_directory: str,
    mask_directory: str | None = None,
) -> LoadedLabelDataset:
    """Build Label Creator state from validated metadata-only discovery."""

    dataset = hyperspectral_dataset.discover_acquisitions(
        image_directory, mask_directory
    )
    dataset_id, provenance = label_manifest.dataset_provenance(dataset)
    model = LabelAssignmentModel(acquisition_items_from_dataset(dataset, provenance))
    return LoadedLabelDataset(
        dataset=dataset,
        dataset_id=dataset_id,
        model=model,
        image_directory=image_directory,
        mask_directory=mask_directory,
    )


class LabelCreatorApplication:
    """Standalone GUI; scientific cube contents are never accessed."""

    FILTER_ALL_TEXT = "All acquisitions"
    FILTER_UNLABELLED_TEXT = "Unlabelled only"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BSA Label Creator")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)
        self.window_icon_format = configure_label_creator_icon(self.root)
        self.style = ttk.Style(self.root)
        self.ui_font_family = configure_label_creator_typography(
            self.root, self.style
        )

        self.loaded: LoadedLabelDataset | None = None
        self._saved_revision = 0
        self._sort_column = "sample_id"
        self._sort_descending = False
        self._filter_keys: list[str] = [ALL_CLASSES, UNLABELLED_ONLY]
        self._filter_job: str | None = None
        self._busy = False
        self._closing = False
        self._background_results: queue.Queue[tuple[bool, object, object]] = queue.Queue()
        self._busy_widgets: list[tk.Widget] = []

        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a validated hyperspectral dataset.")
        self.conflict_var = tk.StringVar(value="No unresolved conflicts.")
        self.total_var = tk.StringVar(value="Total: 0")
        self.labelled_var = tk.StringVar(value="Labelled: 0")
        self.unlabelled_var = tk.StringVar(value="Unlabelled: 0")
        self.pattern_var = tk.StringVar()

        self._build_menu()
        self._build_interface()
        apply_label_creator_typography(self.root)
        self.search_var.trace_add("write", self._schedule_table_refresh)
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-Z>", lambda _event: self.undo())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll_background_results)

    @property
    def model(self) -> LabelAssignmentModel | None:
        return self.loaded.model if self.loaded is not None else None

    @property
    def has_unsaved_changes(self) -> bool:
        return self.model is not None and self.model.revision != self._saved_revision

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Load Dataset…", command=self.choose_dataset)
        file_menu.add_command(label="Import Label Manifest…", command=self.import_manifest)
        file_menu.add_command(label="Export Label Manifest…", command=self.export_manifest)
        file_menu.add_command(
            label="Export Spreadsheet Review (.xlsx)…",
            command=self.export_spreadsheet_review,
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        menu.add_cascade(label="Edit", menu=edit_menu)
        self.root.configure(menu=menu)

    def _build_interface(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x")
        for text, command in (
            ("Load Dataset", self.choose_dataset),
            ("Import Manifest", self.import_manifest),
            ("Export Manifest", self.export_manifest),
        ):
            button = ttk.Button(toolbar, text=text, command=command)
            button.pack(side="left", padx=(0, 6))
            self._busy_widgets.append(button)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=(8, 0))
        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=160)
        self.progress.pack(side="right")
        self.progress.pack_forget()

        summary = ttk.Frame(outer, padding=(0, 10, 0, 8))
        summary.pack(fill="x")
        ttk.Label(summary, textvariable=self.total_var).pack(side="left", padx=(0, 18))
        ttk.Label(summary, textvariable=self.labelled_var).pack(side="left", padx=(0, 18))
        ttk.Label(summary, textvariable=self.unlabelled_var).pack(side="left", padx=(0, 18))
        ttk.Label(summary, textvariable=self.conflict_var).pack(side="right")

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        class_panel = ttk.Frame(body, padding=(0, 0, 10, 0))
        table_panel = ttk.Frame(body)
        body.add(class_panel, weight=1)
        body.add(table_panel, weight=5)

        ttk.Label(class_panel, text="Experimental classes").pack(anchor="w")
        class_list_frame = ttk.Frame(class_panel)
        class_list_frame.pack(fill="both", expand=False, pady=(4, 6))
        self.class_list = tk.Listbox(class_list_frame, exportselection=False, height=9)
        class_scroll = ttk.Scrollbar(
            class_list_frame, orient="vertical", command=self.class_list.yview
        )
        self.class_list.configure(yscrollcommand=class_scroll.set)
        self.class_list.pack(side="left", fill="both", expand=True)
        class_scroll.pack(side="right", fill="y")
        self.class_list.bind(
            "<<ListboxSelect>>", self._use_selected_list_class_for_assignment
        )
        class_buttons = ttk.Frame(class_panel)
        class_buttons.pack(fill="x")
        ttk.Button(class_buttons, text="Add", command=self.add_class).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(class_buttons, text="Rename", command=self.rename_class).pack(
            side="left", fill="x", expand=True, padx=3
        )
        ttk.Button(class_buttons, text="Remove", command=self.remove_class).pack(
            side="left", fill="x", expand=True
        )

        ttk.Label(class_panel, text="Counts by class").pack(anchor="w", pady=(14, 2))
        self.class_counts = ttk.Treeview(
            class_panel, columns=("class", "count"), show="headings", height=9
        )
        self.class_counts.heading("class", text="Class")
        self.class_counts.heading("count", text="Count")
        self.class_counts.column("class", width=150, stretch=True)
        self.class_counts.column("count", width=55, stretch=False, anchor="e")
        self.class_counts.pack(fill="both", expand=True)

        search_bar = ttk.Frame(table_panel)
        search_bar.pack(fill="x", pady=(0, 6))
        ttk.Label(search_bar, text="Search:").pack(side="left")
        ttk.Entry(search_bar, textvariable=self.search_var).pack(
            side="left", fill="x", expand=True, padx=(5, 12)
        )
        ttk.Label(search_bar, text="Show:").pack(side="left")
        self.filter_combo = ttk.Combobox(search_bar, state="readonly", width=24)
        self.filter_combo.pack(side="left", padx=(5, 0))
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_table())

        tree_frame = ttk.Frame(table_panel)
        tree_frame.pack(fill="both", expand=True)
        columns = ("sample_id", "sensor", "filename", "class")
        self.table = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
        )
        headings = {
            "sample_id": "Sample identity",
            "sensor": "Sensor",
            "filename": "Acquisition filename",
            "class": "Experimental class",
        }
        widths = {"sample_id": 260, "sensor": 80, "filename": 300, "class": 220}
        for column in columns:
            self.table.heading(
                column,
                text=headings[column],
                command=lambda selected=column: self.sort_by(selected),
            )
            self.table.column(column, width=widths[column], stretch=column != "sensor")
        vertical = ttk.Scrollbar(tree_frame, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(
            tree_frame, orient="horizontal", command=self.table.xview
        )
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        assignment = ttk.LabelFrame(table_panel, text="Assign selected class", padding=8)
        assignment.pack(fill="x", pady=(8, 0))
        self.assignment_class = ttk.Combobox(assignment, state="readonly", width=25)
        self.assignment_class.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ttk.Button(
            assignment, text="Assign selected rows", command=self.assign_selected
        ).grid(row=0, column=1, padx=3)
        ttk.Button(
            assignment, text="Return selected to unlabelled", command=self.unassign_selected
        ).grid(row=0, column=2, padx=3)

        ttk.Label(assignment, text="Filename pattern (* and ? wildcards):").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(assignment, textvariable=self.pattern_var).grid(
            row=2, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(
            assignment, text="Review pattern assignment", command=self.assign_pattern
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=3)
        self.pattern_help = ttk.Label(
            assignment,
            text=PATTERN_HELP_TEXT,
            justify="left",
            wraplength=680,
        )
        self.pattern_help.grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

        ttk.Label(assignment, text="Source folder:").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self.folder_combo = ttk.Combobox(assignment, state="readonly")
        self.folder_combo.grid(row=5, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            assignment, text="Review folder assignment", command=self.assign_folder
        ).grid(row=5, column=1, columnspan=2, sticky="w", padx=3)
        assignment.columnconfigure(0, weight=1)

        ttk.Label(
            outer,
            text="Labels.csv is for BSA import. Review untrusted labels in the spreadsheet review export, not directly in a spreadsheet.",
            anchor="w",
        ).pack(fill="x", pady=(8, 0))
        status = ttk.Label(outer, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", pady=(8, 0))
        self._update_class_controls()

    def _require_model(self) -> LabelAssignmentModel | None:
        if self.model is None:
            messagebox.showinfo(
                "BSA Label Creator", "Load a validated dataset first.", parent=self.root
            )
            return None
        return self.model

    def _selected_class_name(self) -> str | None:
        model = self._require_model()
        if model is None:
            return None
        name = self.assignment_class.get()
        if name not in model.classes:
            messagebox.showinfo(
                "Select a class",
                "Create and select an experimental class first.",
                parent=self.root,
            )
            return None
        return name

    def _selected_table_ids(self) -> tuple[str, ...]:
        return tuple(self.table.selection())

    def _selected_list_class(self) -> str | None:
        selected = self.class_list.curselection()
        if not selected:
            messagebox.showinfo(
                "Select a class", "Select an experimental class first.", parent=self.root
            )
            return None
        return self.class_list.get(selected[0])

    def _use_selected_list_class_for_assignment(self, _event=None) -> None:
        selected = self.class_list.curselection()
        if selected:
            self.assignment_class.set(self.class_list.get(selected[0]))

    def _model_action(self, action: Callable[[], object]) -> object | None:
        try:
            result = action()
        except LabelCreatorModelError as error:
            messagebox.showerror("Label assignment error", str(error), parent=self.root)
            return None
        self.refresh_all()
        return result

    def add_class(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        if model is None:
            return
        name = simpledialog.askstring(
            "Create experimental class", "Class name:", parent=self.root
        )
        if name is None:
            return
        created = self._model_action(lambda: model.create_class(name))
        if isinstance(created, str):
            self.assignment_class.set(created)

    def rename_class(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        old_name = self._selected_list_class() if model is not None else None
        if model is None or old_name is None:
            return
        new_name = simpledialog.askstring(
            "Rename experimental class",
            "New class name:",
            initialvalue=old_name,
            parent=self.root,
        )
        if new_name is None:
            return
        renamed = self._model_action(lambda: model.rename_class(old_name, new_name))
        if isinstance(renamed, str):
            self.assignment_class.set(renamed)

    def remove_class(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        name = self._selected_list_class() if model is not None else None
        if model is None or name is None:
            return
        assigned = model.summary().per_class.get(name, 0)
        if not messagebox.askyesno(
            "Remove experimental class",
            f"Remove {name!r}?\n\n{assigned} acquisition(s) will return to the "
            "explicitly unlabelled state.",
            parent=self.root,
        ):
            return
        self._model_action(lambda: model.remove_class(name))

    def _review_and_apply(self, proposal: AssignmentProposal) -> None:
        model = self._require_model()
        if model is None:
            return
        if proposal.match_count == 0:
            messagebox.showinfo(
                "No matching acquisitions",
                "The proposed operation matches no acquisitions.",
                parent=self.root,
            )
            return
        target = proposal.target_class if proposal.target_class is not None else "Unlabelled"
        preview = [
            model.acquisition_for_id(value).sample_id
            for value in proposal.acquisition_ids[:12]
        ]
        preview_text = "\n".join(f"  • {value}" for value in preview)
        if proposal.match_count > len(preview):
            preview_text += f"\n  … and {proposal.match_count - len(preview)} more"
        overwrite = ""
        if proposal.overwrite_count:
            overwrite = (
                f"\n\nWarning: {proposal.overwrite_count} existing class "
                "assignment(s) will be overwritten."
            )
        prompt = (
            f"Target class: {target}\n"
            f"Matched acquisitions: {proposal.match_count}\n"
            f"Assignments changed: {proposal.change_count}"
            f"{overwrite}\n\nReview:\n{preview_text}\n\nApply this operation?"
        )
        if not messagebox.askyesno(
            "Review assignment changes", prompt, parent=self.root
        ):
            return
        changed = self._model_action(lambda: model.apply(proposal))
        if isinstance(changed, int):
            self.status_var.set(f"Updated {changed} acquisition assignment(s).")

    def assign_selected(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        class_name = self._selected_class_name() if model is not None else None
        if model is None or class_name is None:
            return
        selected = self._selected_table_ids()
        if not selected:
            messagebox.showinfo(
                "Select acquisitions", "Select one or more table rows.", parent=self.root
            )
            return
        try:
            proposal = model.propose_assignment(selected, class_name)
        except LabelCreatorModelError as error:
            messagebox.showerror("Label assignment error", str(error), parent=self.root)
            return
        self._review_and_apply(proposal)

    def unassign_selected(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        if model is None:
            return
        selected = self._selected_table_ids()
        if not selected:
            messagebox.showinfo(
                "Select acquisitions", "Select one or more table rows.", parent=self.root
            )
            return
        self._review_and_apply(
            model.propose_assignment(
                selected, None, description="return selected acquisitions to unlabelled"
            )
        )

    def assign_pattern(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        class_name = self._selected_class_name() if model is not None else None
        if model is None or class_name is None:
            return
        try:
            proposal = model.propose_filename_pattern(self.pattern_var.get(), class_name)
        except LabelCreatorModelError as error:
            messagebox.showerror("Pattern assignment error", str(error), parent=self.root)
            return
        self._review_and_apply(proposal)

    def assign_folder(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        class_name = self._selected_class_name() if model is not None else None
        if model is None or class_name is None:
            return
        try:
            proposal = model.propose_folder(self.folder_combo.get(), class_name)
        except LabelCreatorModelError as error:
            messagebox.showerror("Folder assignment error", str(error), parent=self.root)
            return
        self._review_and_apply(proposal)

    def undo(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        if model is None:
            return
        try:
            description = model.undo()
        except LabelCreatorModelError as error:
            self.status_var.set(str(error))
            return
        self.refresh_all()
        self.status_var.set(f"Undid: {description}.")

    def _schedule_table_refresh(self, *_args) -> None:
        if self._filter_job is not None:
            self.root.after_cancel(self._filter_job)
        self._filter_job = self.root.after(150, self.refresh_table)

    def _selected_filter_key(self) -> str:
        index = self.filter_combo.current()
        if index < 0 or index >= len(self._filter_keys):
            return ALL_CLASSES
        return self._filter_keys[index]

    def sort_by(self, column: str) -> None:
        if self._sort_column == column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            self._sort_descending = False
        self.refresh_table()

    def refresh_table(self) -> None:
        self._filter_job = None
        model = self.model
        selected = set(self._selected_table_ids())
        self.table.delete(*self.table.get_children())
        if model is None:
            return
        try:
            rows = model.view(
                search=self.search_var.get(),
                class_filter=self._selected_filter_key(),
                sort_column=self._sort_column,
                descending=self._sort_descending,
            )
        except LabelCreatorModelError:
            rows = model.view(search=self.search_var.get())
        for item in rows:
            class_name = model.class_for_id(item.acquisition_id) or "Unlabelled"
            self.table.insert(
                "",
                "end",
                iid=item.acquisition_id,
                values=(
                    item.sample_id,
                    item.sensor,
                    item.acquisition_filename,
                    class_name,
                ),
            )
        retained = tuple(value for value in selected if self.table.exists(value))
        if retained:
            self.table.selection_set(retained)
        self.status_var.set(f"Showing {len(rows)} of {len(model.acquisitions)} acquisitions.")

    def _update_class_controls(self) -> None:
        model = self.model
        previous_assignment = self.assignment_class.get()
        previous_filter = self._selected_filter_key()
        classes = model.classes if model is not None else ()
        self.class_list.delete(0, "end")
        for name in classes:
            self.class_list.insert("end", name)
        self.assignment_class.configure(values=classes)
        if previous_assignment in classes:
            self.assignment_class.set(previous_assignment)
        elif classes:
            self.assignment_class.set(classes[0])
        else:
            self.assignment_class.set("")
        self._filter_keys = [ALL_CLASSES, UNLABELLED_ONLY, *classes]
        self.filter_combo.configure(
            values=(self.FILTER_ALL_TEXT, self.FILTER_UNLABELLED_TEXT, *classes)
        )
        try:
            self.filter_combo.current(self._filter_keys.index(previous_filter))
        except ValueError:
            self.filter_combo.current(0)
        folders = model.folders() if model is not None else ()
        self.folder_combo.configure(values=folders)
        if self.folder_combo.get() not in folders:
            self.folder_combo.set(folders[0] if folders else "")

    def refresh_summary(self) -> None:
        model = self.model
        summary = model.summary() if model is not None else None
        self.total_var.set(f"Total: {summary.total if summary else 0}")
        self.labelled_var.set(f"Labelled: {summary.labelled if summary else 0}")
        self.unlabelled_var.set(f"Unlabelled: {summary.unlabelled if summary else 0}")
        self.class_counts.delete(*self.class_counts.get_children())
        if summary is not None:
            for index, (name, count) in enumerate(summary.per_class.items()):
                self.class_counts.insert("", "end", iid=str(index), values=(name, count))

    def refresh_all(self) -> None:
        self._update_class_controls()
        self.refresh_summary()
        self.refresh_table()

    def _confirm_discard(self) -> bool:
        if not self.has_unsaved_changes:
            return True
        return messagebox.askyesno(
            "Unsaved label assignments",
            "Discard label assignments that have not been exported?",
            parent=self.root,
        )

    def choose_dataset(self) -> None:
        if self._busy or not self._confirm_discard():
            return
        image_directory = filedialog.askdirectory(
            title="Select BIL/RAW acquisition folder", parent=self.root
        )
        if not image_directory:
            return
        mask_directory: str | None = None
        if messagebox.askyesno(
            "Optional mask validation",
            "Segmentation masks are not required for experimental class "
            "labelling and can be generated later with BSA's Segmentor.\n\n"
            "Do you want to validate an existing mask folder now?",
            parent=self.root,
            default=messagebox.NO,
        ):
            selected_mask_directory = filedialog.askdirectory(
                title="Optional: select an existing mask folder",
                parent=self.root,
            )
            if selected_mask_directory:
                mask_directory = selected_mask_directory

        def discover() -> LoadedLabelDataset:
            return load_label_dataset(image_directory, mask_directory)

        self._start_background(
            "Discovering and validating acquisition metadata…",
            discover,
            self._dataset_loaded,
        )

    def _dataset_loaded(self, loaded: object) -> None:
        if not isinstance(loaded, LoadedLabelDataset):
            raise TypeError("Unexpected dataset result.")
        self.loaded = loaded
        self._saved_revision = loaded.model.revision
        self.conflict_var.set("No unresolved identity or assignment conflicts.")
        self.search_var.set("")
        self.refresh_all()
        count = len(loaded.model.acquisitions)
        noun = "acquisition" if count == 1 else "acquisitions"
        self.status_var.set(
            f"Ready — {count} {noun} loaded."
        )

    def import_manifest(self) -> None:
        if self._busy or not self._confirm_discard():
            return
        model = self._require_model()
        if model is None or self.loaded is None:
            return
        path = filedialog.askopenfilename(
            title="Import BSA label manifest",
            filetypes=(("CSV label manifest", "*.csv"), ("All files", "*.*")),
            parent=self.root,
        )
        if not path:
            return
        dataset = self.loaded.dataset

        def load_manifest() -> label_manifest.ValidatedLabelAssignments:
            return label_manifest.load_and_validate_label_manifest(path, dataset)

        def imported(result: object) -> None:
            if not isinstance(result, label_manifest.ValidatedLabelAssignments):
                raise TypeError("Unexpected manifest result.")
            model.apply_validated_assignments(result)
            self._saved_revision = model.revision
            self.conflict_var.set("No unresolved identity or assignment conflicts.")
            self.refresh_all()
            self.status_var.set(
                f"Imported {result.labelled_count} labelled and "
                f"{result.excluded_unlabelled_count} unlabelled acquisition(s)."
            )

        self._start_background("Validating label manifest…", load_manifest, imported)

    def export_manifest(self) -> None:
        if self._busy:
            return
        model = self._require_model()
        if model is None or self.loaded is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export BSA label manifest",
            defaultextension=".csv",
            filetypes=(("CSV label manifest", "*.csv"),),
            parent=self.root,
        )
        if not path:
            return
        dataset = self.loaded.dataset
        assignments = model.assignment_mapping()
        revision = model.revision

        def export() -> tuple[label_manifest.ValidatedLabelAssignments, int]:
            manifest = label_manifest.create_label_manifest(dataset, assignments)
            label_manifest.validate_label_manifest(manifest, dataset)
            label_manifest.save_label_manifest(path, manifest)
            validated = label_manifest.load_and_validate_label_manifest(path, dataset)
            return validated, revision

        def exported(result: object) -> None:
            validated, exported_revision = result
            if model.revision == exported_revision:
                self._saved_revision = exported_revision
            self.conflict_var.set("No unresolved identity or assignment conflicts.")
            self.refresh_summary()
            self.status_var.set(
                f"Exported {validated.total_count} acquisition(s) to {path}."
            )

        self._start_background("Validating and exporting manifest…", export, exported)

    def export_spreadsheet_review(self) -> None:
        """Export current assignments for inspection, without changing Labels.csv."""

        if self._busy:
            return
        model = self._require_model()
        if model is None or self.loaded is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export spreadsheet review",
            defaultextension=".xlsx",
            filetypes=(("Spreadsheet review", "*.xlsx"),),
            parent=self.root,
        )
        if not path:
            return
        dataset = self.loaded.dataset
        assignments = model.assignment_mapping()

        def export() -> int:
            manifest = label_manifest.create_label_manifest(dataset, assignments)
            label_manifest.validate_label_manifest(manifest, dataset)
            label_manifest.save_manifest_review_xlsx(path, manifest)
            return len(manifest.records)

        def exported(count: object) -> None:
            self.status_var.set(f"Exported {count} acquisition(s) for spreadsheet review to {path}.")

        self._start_background("Exporting spreadsheet review…", export, exported)

    def _start_background(
        self,
        status: str,
        operation: Callable[[], object],
        on_success: Callable[[object], None],
    ) -> None:
        if self._busy:
            return
        self._busy = True
        for widget in self._busy_widgets:
            widget.configure(state="disabled")
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        self.progress.pack(side="right")
        self.progress.start(12)
        self.status_var.set(status)

        def worker() -> None:
            try:
                result = operation()
            except Exception as error:
                self._background_results.put((False, on_success, error))
            else:
                self._background_results.put((True, on_success, result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_background_progress(self, *, completed: bool) -> None:
        """Leave a deterministic terminal value, then hide the idle indicator."""

        self.progress.stop()
        self.progress.configure(
            mode="determinate",
            maximum=100,
            value=100 if completed else 0,
        )
        self.progress.pack_forget()

    def _poll_background_results(self) -> None:
        try:
            while True:
                success, callback, payload = self._background_results.get_nowait()
                self._busy = False
                for widget in self._busy_widgets:
                    widget.configure(state="normal")
                if success:
                    try:
                        callback(payload)
                    except Exception as error:
                        self._finish_background_progress(completed=False)
                        self._show_background_error(error)
                    else:
                        self._finish_background_progress(completed=True)
                else:
                    self._finish_background_progress(completed=False)
                    self._show_background_error(payload)
        except queue.Empty:
            pass
        if not self._closing:
            self.root.after(50, self._poll_background_results)

    def _show_background_error(self, error: object) -> None:
        if isinstance(error, hyperspectral_dataset.DatasetMatchingError):
            title = "Dataset matching failed"
        elif isinstance(error, label_manifest.LabelManifestError):
            title = "Label manifest validation failed"
        else:
            title = "BSA Label Creator error"
        message = str(error)
        self.conflict_var.set("Unresolved validation error — see message.")
        first_line = message.splitlines()[0] if message else title
        self.status_var.set(f"Error — {first_line}")
        messagebox.showerror(title, message, parent=self.root)

    def close(self) -> None:
        if self._busy:
            messagebox.showinfo(
                "Operation in progress",
                "Wait for the current dataset or manifest operation to finish.",
                parent=self.root,
            )
            return
        if not self._confirm_discard():
            return
        self._closing = True
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        LabelCreatorApplication(root)
    except LabelCreatorTypographyError as error:
        messagebox.showerror("BSA Label Creator font error", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
