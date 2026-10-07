"""Compatibility launcher for the standalone BSA Label Creator."""

from pathlib import Path
import sys


SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from bsa.labeling.app import main


if __name__ == "__main__":
    main()
