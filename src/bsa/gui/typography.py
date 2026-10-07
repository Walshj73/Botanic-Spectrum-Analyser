"""Verified desktop typography for the main BSA Tk interface."""

from __future__ import annotations

import sys
import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


BODY_SIZE = 10
SMALL_SIZE = 9


def font_is_proportional(font: tkfont.Font) -> bool:
    """Check rendered glyph widths, since Tk can substitute a fixed font."""

    return font.measure("iiiiiiiiii") != font.measure("WWWWWWWWWW")


def preferred_families(platform: str | None = None) -> tuple[str, ...]:
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return ("Segoe UI", "Arial", "Noto Sans", "DejaVu Sans")
    if platform == "darwin":
        return ("SF Pro Text", "Helvetica", "Helvetica Neue", "Noto Sans", "DejaVu Sans")
    return ("Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial")


def resolve_font_family(
    root: tk.Misc, platform: str | None = None
) -> tuple[str | None, str, str]:
    """Return the rendered family, requested family and concise diagnostic."""

    preferred = preferred_families(platform)
    installed = {family.casefold(): family for family in tkfont.families(root)}
    attempts: list[tuple[str, str, int, int]] = []
    for requested in preferred:
        candidate = installed.get(requested.casefold(), requested)
        probe = tkfont.Font(root=root, family=candidate, size=BODY_SIZE)
        actual = str(probe.actual("family"))
        narrow = probe.measure("iiiiiiiiii")
        wide = probe.measure("WWWWWWWWWW")
        attempts.append((requested, actual, narrow, wide))
        if actual.casefold() != "fixed" and narrow != wide:
            return actual, requested, (
                f"BSA Tk font: requested '{requested}', resolved '{actual}' "
                f"(proportional; i={narrow}px, W={wide}px)."
            )

    for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try:
            probe = tkfont.nametofont(font_name, root=root)
        except tk.TclError:
            continue
        actual = str(probe.actual("family"))
        if actual.casefold() != "fixed" and font_is_proportional(probe):
            return actual, font_name, (
                f"BSA Tk font: requested system '{font_name}', resolved '{actual}' "
                "(proportional)."
            )

    requested, actual, narrow, wide = attempts[0]
    return None, requested, (
        f"BSA Tk font: requested '{requested}', resolved '{actual}' "
        f"(i={narrow}px, W={wide}px); this Tk runtime cannot render a "
        "proportional font. Leaving its default fonts in place."
    )


def configure_typography(root: tk.Tk) -> str | None:
    """Apply fonts only after Tk demonstrates proportional rendering."""

    family, requested, diagnostic = resolve_font_family(root)
    if os.environ.get("BSA_DEBUG") == "1":
        print(diagnostic, file=sys.stderr)
    root._bsa_font_requested = requested
    root._bsa_font_family = family
    if family is None:
        return None

    settings = {
        "TkDefaultFont": (BODY_SIZE, "normal"),
        "TkTextFont": (BODY_SIZE, "normal"),
        "TkMenuFont": (BODY_SIZE, "normal"),
        "TkHeadingFont": (BODY_SIZE, "bold"),
        "TkCaptionFont": (BODY_SIZE, "bold"),
        "TkSmallCaptionFont": (SMALL_SIZE, "normal"),
        "TkIconFont": (BODY_SIZE, "normal"),
        "TkTooltipFont": (SMALL_SIZE, "normal"),
    }
    for name, (size, weight) in settings.items():
        try:
            named = tkfont.nametofont(name, root=root)
        except tk.TclError:
            continue
        named.configure(family=family, size=size, weight=weight)

    root.option_add("*font", "TkDefaultFont")
    root.option_add("*Text.font", "TkTextFont")
    root.option_add("*Menu.font", "TkMenuFont")
    root.option_add("*Listbox.font", "TkDefaultFont")
    root.option_add("*TCombobox*Listbox.font", "TkDefaultFont")

    style = ttk.Style(root)
    for widget_style in (
        "TButton", "TLabel", "TEntry", "TCombobox", "TCheckbutton",
        "TRadiobutton", "TMenubutton", "TLabelframe.Label", "TNotebook.Tab",
        "Treeview",
    ):
        style.configure(widget_style, font="TkDefaultFont")
    style.configure("Treeview.Heading", font="TkHeadingFont")

    # Change table row height only if the selected font would be clipped.
    body = tkfont.nametofont("TkDefaultFont", root=root)
    minimum_rowheight = body.metrics("linespace") + 4
    current_rowheight = style.lookup("Treeview", "rowheight")
    if not current_rowheight or int(current_rowheight) < minimum_rowheight:
        style.configure("Treeview", rowheight=minimum_rowheight)
    return family
