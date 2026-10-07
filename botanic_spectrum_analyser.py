"""Compatibility launcher for the Botanic Spectrum Analyser GUI."""

from pathlib import Path
import sys


SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from bsa.gui.main import run


if __name__ == "__main__":
    run()
