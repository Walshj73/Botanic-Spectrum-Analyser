"""Run a bounded GUI smoke check from the frozen executable and external data.

This module is used only with the private ``--packaging-smoke`` launcher flag.
It is bundled so the check exercises exactly the shipped Python runtime.
"""

import os
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk
from unittest import mock

from PIL import Image

from bsa.gui import main as app
from bsa.hyperspectral import dataset
from bsa.hyperspectral.hdr import build_envi_header
from bsa.labeling import manifest
from bsa.utils.resources import resource_path


def descendants(parent):
    for child in parent.winfo_children():
        yield child
        yield from descendants(child)


def control(parent, caption, kinds=(tk.Button, ttk.Button)):
    return next(child for child in descendants(parent)
                if isinstance(child, kinds) and child.cget("text") == caption)


def wait_for(root, predicate, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError(f"Timed out waiting for {description}")


def run_smoke(example_root, output_root):
    assert getattr(sys, "frozen", False), "smoke must run from the packaged executable"
    assert Path(sys._MEIPASS).is_dir()
    source = Path(example_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert source.is_dir() and output.is_dir()
    assert Path(app.__file__).resolve().is_relative_to(Path(sys._MEIPASS).resolve())
    assert resource_path("Models", "BARLEY SWIR.h5").is_file()
    assert resource_path("BSA_logo.png").is_file()

    identities = ("68-15-PS_Tray_1008-SWIR", "68-15-PS_Tray_1012-SWIR")
    bils, masks, analysis_output, saves, rgb = (
        output / name for name in ("bils", "masks", "analysis", "images", "rgb")
    )
    for folder in (bils, masks, analysis_output, saves, rgb):
        folder.mkdir()
    for identity in identities:
        for role in ("DarkCalibration", "Data", "WhiteCalibration"):
            name = f"{identity}-{role}.bil"
            (bils / name).symlink_to((source / "SWIR BILS" / name).resolve())
        name = f"{identity}-HcRgbImage-0-PlantMask.png"
        (masks / name).symlink_to((source / "SWIR Masks" / name).resolve())
    Image.new("RGB", (64, 64), (32, 150, 74)).save(rgb / "smoke.png")

    headers = {}
    for role, lines in (("dark", 5), ("data", 420), ("white", 5)):
        path = output / f"{role}.hdr"
        path.write_text(build_envi_header(510, lines, 318, 12, 0, (900, 1700)))
        headers[role] = str(path)
    discovered = dataset.discover_dataset(bils, masks)
    _, provenance = manifest.dataset_provenance(discovered)
    assignments = {record.acquisition_id: label for record, label in
                   zip(provenance, ("Controlled", "Waterlogged"))}
    labels_path = output / "Labels.csv"
    manifest.save_label_manifest(labels_path, manifest.create_label_manifest(discovered, assignments))

    prior_cwd = Path.cwd()
    os.chdir(output)
    root = None
    try:
        with mock.patch.object(tk.Tk, "mainloop", return_value=None), \
             mock.patch.object(app.messagebox, "showerror") as errors, \
             mock.patch.object(app.messagebox, "showinfo"), \
             mock.patch.object(app, "open_in_file_manager"):
            app.main()
            root = app.application_window
            root.update()
            assert root.title() == "Botanic Spectrum Analyser"
            assert bool(root.attributes("-zoomed")), "startup is not maximized"
            assert root.resizable() == (1, 1)
            root.attributes("-zoomed", False)
            root.update()
            root.geometry("1100x800")
            root.update()
            root.attributes("-zoomed", True)
            root.update()
            notebook = next(child for child in root.winfo_children()
                            if isinstance(child, ttk.Notebook))
            assert len(notebook.tabs()) == 5
            assert control(root, "Import HDR / CSV / TXT...")
            assert control(root, "Add / Update Metric")
            assert "PlantScreen VNIR" in tuple(
                str(child.cget("text")) for child in descendants(root)
                if isinstance(child, (tk.Radiobutton, ttk.Radiobutton))
            )
            assert "PlantScreen SWIR" in tuple(
                str(child.cget("text")) for child in descendants(root)
                if isinstance(child, (tk.Radiobutton, ttk.Radiobutton))
            )
            print("Frozen runtime, maximized resizable GUI, five tabs, HDR and Upload controls: OK", flush=True)

            notebook.select(0)
            model = next(child for child in descendants(root)
                         if isinstance(child, ttk.Combobox)
                         and "BARLEY SWIR" in child.cget("values"))
            model.set("BARLEY SWIR")
            model.event_generate("<<ComboboxSelected>>")
            with mock.patch.object(app.filedialog, "askdirectory",
                                   side_effect=[str(rgb), str(output)]):
                control(root, "Load RGB Slices").invoke()
            segment_button = control(root, "Run Segmentation")
            assert segment_button.cget("state") == "normal"
            segment_button.invoke()
            wait_for(root, lambda: segment_button.cget("state") == "normal", 240,
                     "CPU segmentation")
            assert not errors.called, errors.call_args_list
            mask_runs = list(output.glob("Masks-*/smoke*Mask.png"))
            assert len(mask_runs) == 1, list(output.glob("Masks-*/*"))
            print("Packaged TensorFlow H5 loading and CPU Mask Creator: OK", flush=True)

            notebook.select(2)
            with mock.patch.object(app.filedialog, "askopenfilename",
                                   side_effect=[headers["dark"], headers["data"], headers["white"]]), \
                 mock.patch.object(app.filedialog, "askdirectory",
                                   side_effect=[str(bils), str(masks)]):
                for caption in ("DARK", "DATA", "WHITE", "Load BIL/RAW Files",
                                "Load Segmentation Masks"):
                    control(root, caption).invoke()
            app.sensorselect.set(1)
            with mock.patch.object(app.filedialog, "askopenfilename",
                                   return_value=str(labels_path)):
                control(root, "Yes", (tk.Radiobutton, ttk.Radiobutton)).invoke()
            notebook.select(3)
            with mock.patch.object(app.filedialog, "askdirectory",
                                   return_value=str(analysis_output)):
                control(root, "Start Analyser").invoke()
            wait_for(root, lambda: root._bsa_analysis_controller.active is None,
                     240, "SWIR analysis")
            root.update()
            assert not errors.called, errors.call_args_list
            assert len(app.results_df) == 2
            assert set(app.results_df["Label"]) == {"Controlled", "Waterlogged"}
            assert "NDWI Mean" in app.results_df
            assert list(analysis_output.glob("*.csv")), "automatic analysis CSV missing"
            print("Packaged ENVI/BIL, masks, labels, SWIR analysis and output: OK", flush=True)

            notebook.select(4)
            colour_panel = control(root, "Class colours", (ttk.LabelFrame,))
            assert colour_panel.winfo_manager()
            canvas = next(child for child in colour_panel.winfo_children()
                          if isinstance(child, tk.Canvas))
            rows = next(child for child in canvas.winfo_children()
                        if isinstance(child, ttk.Frame))
            pickers = {
                next(item.cget("text") for item in row.winfo_children()
                     if isinstance(item, ttk.Label)):
                next(item for item in row.winfo_children()
                     if isinstance(item, tk.Button))
                for row in rows.winfo_children()
            }
            assert set(pickers) == {"Controlled", "Waterlogged"}
            with mock.patch.object(app.colorchooser, "askcolor",
                                   return_value=((0, 255, 0), "#00ff00")):
                pickers["Controlled"].invoke()
            assert pickers["Controlled"].cget("fg") == "#00ff00"
            with mock.patch.object(app.filedialog, "askdirectory",
                                   return_value=str(saves)):
                control(root, "Save Image").invoke()
            assert list(saves.glob("*.png")) and list(saves.glob("*.jpg"))
            print("Packaged spectra plot, class colour editor and PNG/JPEG Save: OK", flush=True)
    finally:
        if root is not None and root.winfo_exists():
            root._bsa_close_program()
        os.chdir(prior_cwd)
