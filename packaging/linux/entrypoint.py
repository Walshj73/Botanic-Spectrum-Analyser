"""Clean executable entry point for the Linux standalone release."""

import sys


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--packaging-smoke":
        from smoke import run_smoke

        run_smoke(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ["--label-creator"]:
        from bsa.labeling.app import main as run_label_creator

        sys.argv.pop(1)
        run_label_creator()
    else:
        from bsa.gui.main import run

        run()
