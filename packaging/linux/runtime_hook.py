"""Use a compatible fontconfig configuration with the bundled Linux library."""

import os
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    os.environ.setdefault(
        "FONTCONFIG_FILE", str(Path(sys._MEIPASS) / "fonts.conf")
    )
    os.environ["PATH"] = os.pathsep.join((sys._MEIPASS, os.environ.get("PATH", "")))
