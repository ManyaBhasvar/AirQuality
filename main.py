"""Entry point of the Air Quality Analysis System.

Run it with::

    python main.py

The script only prepares the local storage and starts the Tkinter interface;
all logic lives in :mod:`data_manager`, :mod:`analysis`, :mod:`visualization`
and :mod:`gui`.
"""

from __future__ import annotations

import sys

import data_manager
from gui import run


def main() -> int:
    """Prepare the CSV storage and open the main window."""
    try:
        data_manager.ensure_data_file()
    except data_manager.DataError as error:
        print(f"Could not prepare the data file:\n{error}", file=sys.stderr)
        return 1

    try:
        run()
    except Exception as error:  # e.g. no display / Tkinter not available
        print(f"The application could not start: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
