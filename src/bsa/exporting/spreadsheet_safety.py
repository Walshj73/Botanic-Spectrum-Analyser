"""Literal text handling for spreadsheet-facing analysis exports.

CSV has no cell types. Dangerous text gains one leading tab in a quoted field;
XLSX retains the exact text and marks its cell as a string. Numeric values are
never converted to text by these helpers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


_FORMULA_START = "=+-@＝＋－＠"


def dangerous_spreadsheet_text(value: str) -> bool:
    """Recognize formula triggers after whitespace/control-character prefixes."""

    index = 0
    while index < len(value) and (value[index].isspace() or value[index] in "\ufeff\x00"):
        index += 1
    return index < len(value) and value[index] in _FORMULA_START


def csv_safe_cell(value: Any) -> Any:
    """Neutralize only dangerous strings; leave numbers and safe text intact."""

    if isinstance(value, str) and dangerous_spreadsheet_text(value):
        return "\t" + value
    return value


def write_dataframe_csv(frame: Any, path: str | Path, *, header: bool = True,
                        mode: str = "w") -> None:
    """Write a DataFrame with each text cell escaped and quoted independently."""

    safe = frame.map(csv_safe_cell)
    safe.columns = [csv_safe_cell(column) for column in frame.columns]
    safe.to_csv(path, index=False, header=header, mode=mode,
                quoting=csv.QUOTE_NONNUMERIC)


def write_dataframe_xlsx(frame: Any, path: str | Path, *, append: bool = False) -> None:
    """Write or append a DataFrame, forcing source strings to literal XLSX cells."""

    import pandas as pd

    writer_options = {"engine": "openpyxl"}
    if append:
        writer_options.update(mode="a", if_sheet_exists="overlay")
    with pd.ExcelWriter(path, **writer_options) as writer:
        startrow = writer.sheets["Sheet1"].max_row if append else 0
        frame.to_excel(writer, index=False, startrow=startrow, header=not append)
        sheet = writer.sheets["Sheet1"]
        if not append:
            for column_index, heading in enumerate(frame.columns, start=1):
                if isinstance(heading, str):
                    sheet.cell(1, column_index).data_type = "s"
        for row_index, row in enumerate(frame.itertuples(index=False, name=None),
                                        start=startrow + (1 if append else 2)):
            for column_index, value in enumerate(row, start=1):
                if isinstance(value, str):
                    sheet.cell(row_index, column_index).data_type = "s"
