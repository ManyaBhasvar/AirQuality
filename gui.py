"""Tkinter user interface for the Air Quality Analysis System.

The GUI only *displays* results: all file access lives in :mod:`data_manager`,
all calculations in :mod:`analysis` and all charts in :mod:`visualization`.
Only standard Tkinter / ttk widgets are used for the interface itself.

Screens
-------
* :class:`AirQualityApp`            - main window with the action buttons
* :class:`AddObservationWindow`     - form for a new observation
* :class:`ViewRecordsWindow`        - scrollable table of all records
* :class:`SearchFilterWindow`       - filter by location / date / AQI / time
* :class:`StatisticsWindow`         - descriptive statistics
* :class:`LocationComparisonWindow` - compare locations
* :class:`PeriodComparisonWindow`   - compare two date ranges
* :class:`PoorAirQualityWindow`     - highest-AQI observations
* :class:`UnusualValuesWindow`      - z-score anomaly detection
* :class:`ReportsWindow`            - embedded Matplotlib charts + PNG export
"""

from __future__ import annotations

import datetime as dt
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import analysis
import data_manager as dm
import visualization as viz

APP_TITLE = "Air Quality Analysis System"
APP_SUBTITLE = "Offline desktop analysis with Tkinter, Pandas, NumPy and Matplotlib"

# Fonts are filled in by :func:`init_fonts` once a Tk root window exists.
TITLE_FONT: tuple = ("Helvetica", 15, "bold")
SUBTITLE_FONT: tuple = ("Helvetica", 9)
MONO_FONT: tuple = ("Courier", 9)

PAD = 8

#: Columns shown in the "all records" table, with their display widths.
RECORD_COLUMNS = [
    (dm.OBS_ID, 85),
    (dm.DATE, 95),
    (dm.TIME, 60),
    (dm.LOCATION, 140),
    (dm.PM25, 70),
    (dm.PM10, 70),
    (dm.CO, 60),
    (dm.NO2, 60),
    (dm.SO2, 60),
    (dm.O3, 60),
    (dm.AQI, 60),
]

#: The form fields of the "add observation" screen: (column, hint, unit).
FORM_FIELDS = [
    (dm.DATE, "YYYY-MM-DD", ""),
    (dm.TIME, "HH:MM (24-hour)", ""),
    (dm.LOCATION, "e.g. Campus Library", ""),
    (dm.PM25, "0 - 1000", "µg/m³"),
    (dm.PM10, "0 - 2000", "µg/m³"),
    (dm.CO, "0 - 100", "mg/m³"),
    (dm.NO2, "0 - 2000", "µg/m³"),
    (dm.SO2, "0 - 2000", "µg/m³"),
    (dm.O3, "0 - 2000", "µg/m³"),
    (dm.AQI, "0 - 500", "index"),
]


def init_fonts() -> None:
    """Read the default Tk fonts and set up the module font tuples.

    Must be called after a Tk root window exists (Tk needs an interpreter
    before named fonts can be queried).
    """
    global TITLE_FONT, SUBTITLE_FONT, MONO_FONT
    default_family = tkfont.nametofont("TkDefaultFont").actual("family")
    mono_family = tkfont.nametofont("TkFixedFont").actual("family")
    TITLE_FONT = (default_family, 15, "bold")
    SUBTITLE_FONT = (default_family, 9)
    MONO_FONT = (mono_family, 9)


# ---------------------------------------------------------------------------
# Error handling helpers - the user never sees a raw traceback
# ---------------------------------------------------------------------------
def handle_error(parent: tk.Misc, error: Exception) -> None:
    """Show a friendly dialog that matches the kind of error that occurred."""
    if isinstance(error, dm.DataError):
        messagebox.showerror("Data file error", str(error), parent=parent)
    elif isinstance(error, analysis.NoDataError):
        messagebox.showinfo("No data available", str(error), parent=parent)
    elif isinstance(error, ValueError):
        messagebox.showwarning("Invalid input", str(error), parent=parent)
    else:
        messagebox.showerror(
            "Unexpected error",
            f"{type(error).__name__}: {error}\n\n"
            "The action was cancelled and the data file was not modified.",
            parent=parent,
        )


def show_info(parent: tk.Misc, message: str, title: str = "Information") -> None:
    """Show an informational message box."""
    messagebox.showinfo(title, message, parent=parent)


# ---------------------------------------------------------------------------
# Small widget factories
# ---------------------------------------------------------------------------
def build_table(parent: tk.Misc, columns: list[tuple[str, int]],
                height: int = 14) -> tuple[ttk.Frame, ttk.Treeview]:
    """Create a Treeview with headings, vertical and horizontal scrollbars."""
    container = ttk.Frame(parent)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    names = [name for name, _ in columns]
    tree = ttk.Treeview(container, columns=names, show="headings", height=height)
    for name, width in columns:
        tree.heading(name, text=name)
        tree.column(name, width=width, anchor="center", stretch=False)

    vertical = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    horizontal = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    return container, tree


def fill_table(tree: ttk.Treeview, rows: list[tuple]) -> None:
    """Replace the contents of a Treeview with the given rows."""
    tree.delete(*tree.get_children())
    for row in rows:
        tree.insert("", "end", values=row)


def build_text_box(parent: tk.Misc, height: int = 12) -> tuple[ttk.Frame, tk.Text]:
    """Create a read-only text box (monospace) with scrollbars."""
    container = ttk.Frame(parent)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    text = tk.Text(container, height=height, wrap="none", font=MONO_FONT,
                   background="#fbfbfb", relief="solid", borderwidth=1)
    vertical = ttk.Scrollbar(container, orient="vertical", command=text.yview)
    horizontal = ttk.Scrollbar(container, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

    text.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    return container, text


def set_text(widget: tk.Text, content: str) -> None:
    """Replace the content of a read-only Text widget."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


def labelled_entry(parent: tk.Misc, row: int, label: str, hint: str = "",
                   width: int = 18, value: str = "",
                   column_offset: int = 0) -> ttk.Entry:
    """Add a 'label + entry (+ hint)' group to a grid and return the entry."""
    ttk.Label(parent, text=label).grid(row=row, column=column_offset,
                                       sticky="w", padx=(0, PAD), pady=3)
    entry = ttk.Entry(parent, width=width)
    entry.insert(0, value)
    entry.grid(row=row, column=column_offset + 1, sticky="w", pady=3)
    if hint:
        ttk.Label(parent, text=hint, font=SUBTITLE_FONT).grid(
            row=row, column=column_offset + 2, sticky="w", padx=(PAD, PAD * 2), pady=3)
    return entry


def parse_optional_float(text: str, label: str) -> float | None:
    """Convert an optional entry value to a float, with a readable error."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number (got {text!r}).") from exc


# ---------------------------------------------------------------------------
# Base window
# ---------------------------------------------------------------------------
class BaseWindow(tk.Toplevel):
    """Shared layout for the secondary screens: header, body and footer."""

    def __init__(self, app: "AirQualityApp", title: str,
                 geometry: str = "920x620") -> None:
        super().__init__(app)
        self.app = app
        self.title(f"{title} - {APP_TITLE}")
        self.geometry(geometry)
        self.minsize(600, 420)
        self.transient(app)

        header = ttk.Frame(self, padding=(PAD, PAD, PAD, 0))
        header.pack(fill="x")
        ttk.Label(header, text=title, font=TITLE_FONT).pack(anchor="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=PAD, pady=(4, 0))

        self.body = ttk.Frame(self, padding=PAD)
        self.body.pack(fill="both", expand=True)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=PAD)
        footer = ttk.Frame(self, padding=PAD)
        footer.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(footer, textvariable=self.status_var,
                  font=SUBTITLE_FONT).pack(side="left")
        self.buttons = ttk.Frame(footer)
        self.buttons.pack(side="right")

        self.bind("<Escape>", lambda _event: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)

    # -- helpers used by the subclasses -------------------------------------
    def add_footer_button(self, text: str, command, default: bool = False) -> ttk.Button:
        """Add a button to the bottom-right button row."""
        button = ttk.Button(self.buttons, text=text, command=command)
        button.pack(side="right", padx=(6, 0))
        if default:
            # Marks the main action of the screen (Enter-host border in ttk).
            button.configure(default="active")
        return button

    def add_back_button(self) -> ttk.Button:
        """Add a 'Back' button that closes the window."""
        return self.add_footer_button("Back", self.close)

    def set_status(self, text: str) -> None:
        """Write a line of text into the status bar."""
        self.status_var.set(text)

    def current_data(self) -> pd.DataFrame:
        """Always analyse the latest saved data."""
        return self.app.get_data()

    def close(self) -> None:
        """Close the window and unregister it from the main application."""
        self.app.forget_window(self)
        self.destroy()

    def refresh(self) -> None:
        """Reload data and redraw; overridden where a window shows a table."""
        self.app.refresh()
        self.set_status("Data reloaded from the local CSV file.")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class AirQualityApp(tk.Tk):
    """Main application window with the action buttons and the status bar."""

    def __init__(self) -> None:
        super().__init__()
        init_fonts()
        self.title(APP_TITLE)
        self.geometry("600x580")
        self.minsize(560, 540)

        self._data: pd.DataFrame | None = None
        self._windows: dict[str, BaseWindow] = {}
        self.status_records = tk.StringVar(value="Records: 0")

        self._build_menu()
        self._build_layout()
        self.refresh()
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    # -- data access --------------------------------------------------------
    def refresh(self) -> None:
        """Reload the CSV file and update the status bar."""
        try:
            self._data = dm.load_data()
        except Exception as error:  # report the problem instead of crashing
            handle_error(self, error)
            self._data = dm.empty_frame()
        self._update_status()

    def get_data(self) -> pd.DataFrame:
        """Return the cached observations, loading them when necessary."""
        if self._data is None:
            self.refresh()
        return self._data

    def _update_status(self) -> None:
        data = self.get_data()
        if data.empty:
            self.status_records.set("Records: 0  |  No observations stored yet")
        else:
            locations = dm.unique_locations(data)
            self.status_records.set(
                f"Records: {len(data)}  |  Locations: {len(locations)}  |  "
                f"File: {dm.DATA_FILE.relative_to(dm.PROJECT_ROOT)}"
            )

    # -- window management --------------------------------------------------
    def open_window(self, key: str, factory) -> None:
        """Open a window, or bring the already-open one to the front."""
        window = self._windows.get(key)
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            window.focus_set()
            return
        try:
            self._windows[key] = factory(self)
        except Exception as error:
            handle_error(self, error)

    def forget_window(self, window: BaseWindow) -> None:
        """Remove a closing window from the registry."""
        for key, value in list(self._windows.items()):
            if value is window:
                del self._windows[key]

    def close_app(self) -> None:
        """Close every open screen and exit the application."""
        for window in list(self._windows.values()):
            if window.winfo_exists():
                window.destroy()
        self.destroy()

    # -- layout -------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Refresh Data", command=self._refresh_action)
        file_menu.add_command(label="Load Demo (Sample) Data", command=self._load_demo_data)
        file_menu.add_separator()
        file_menu.add_command(label="Clear All Data...", command=self._clear_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close_app)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="View Records",
                              command=lambda: self.open_window("records", ViewRecordsWindow))
        view_menu.add_command(label="Search / Filter",
                              command=lambda: self.open_window("filter", SearchFilterWindow))
        view_menu.add_command(label="Visual Reports",
                              command=lambda: self.open_window("reports", ReportsWindow))
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Data Storage", command=self._show_storage_info)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menubar)

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=PAD)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="AIR QUALITY ANALYSIS SYSTEM",
                  font=TITLE_FONT).pack(pady=(4, 0))
        ttk.Label(container, text=APP_SUBTITLE, font=SUBTITLE_FONT).pack(pady=(0, PAD))
        ttk.Separator(container, orient="horizontal").pack(fill="x")

        actions = ttk.LabelFrame(container, text="Actions", padding=PAD)
        actions.pack(fill="both", expand=True, pady=PAD)
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        buttons = [
            ("Add Observation", lambda: self.open_window("add", AddObservationWindow)),
            ("View Records", lambda: self.open_window("records", ViewRecordsWindow)),
            ("Search / Filter", lambda: self.open_window("filter", SearchFilterWindow)),
            ("Statistics", lambda: self.open_window("stats", StatisticsWindow)),
            ("Compare Locations",
             lambda: self.open_window("locations", LocationComparisonWindow)),
            ("Compare Periods", lambda: self.open_window("periods", PeriodComparisonWindow)),
            ("Poor Air Quality", lambda: self.open_window("poor", PoorAirQualityWindow)),
            ("Unusual Values", lambda: self.open_window("unusual", UnusualValuesWindow)),
            ("Visual Reports", lambda: self.open_window("reports", ReportsWindow)),
            ("Load Demo Data", self._load_demo_data),
        ]
        for index, (label, command) in enumerate(buttons):
            ttk.Button(actions, text=label, command=command, width=22).grid(
                row=index // 2, column=index % 2, padx=6, pady=5, sticky="ew")

        status_bar = ttk.LabelFrame(container, text="Status", padding=(PAD, 4, PAD, 6))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status_records,
                  font=SUBTITLE_FONT).pack(anchor="w")
        ttk.Label(status_bar, font=SUBTITLE_FONT,
                  text="Data is stored locally in a CSV file - no internet connection is "
                       "used, nothing is uploaded.").pack(anchor="w")

    # -- main-window actions ------------------------------------------------
    def _refresh_action(self) -> None:
        self.refresh()
        show_info(self, "Data reloaded from the local CSV file.")

    def _load_demo_data(self) -> None:
        try:
            existing = self.get_data()
            question = (
                "Append the bundled DEMO (sample) observations to your data file?\n\n"
                "The demo rows are sample data for demonstration only and can be removed\n"
                "later with 'File > Clear All Data...'."
            )
            if not existing.empty:
                question += f"\n\nYour file already contains {len(existing)} observation(s)."
            if not messagebox.askyesno("Load demo data", question, parent=self):
                return
            imported = dm.import_sample_data()
            self.refresh()
            show_info(self, f"{imported} demo observation(s) were added.\n\n"
                            "These are sample data - add your own observations with "
                            "'Add Observation'.")
        except Exception as error:
            handle_error(self, error)

    def _clear_data(self) -> None:
        try:
            existing = self.get_data()
            if not messagebox.askyesno(
                "Clear all data",
                f"This deletes all {len(existing)} stored observation(s) from\n"
                f"{dm.DATA_FILE.name}.\n\nThis cannot be undone. Continue?",
                parent=self, icon="warning",
            ):
                return
            dm.replace_all(dm.empty_frame())
            self.refresh()
            show_info(self, "All observations were removed; the CSV header was kept.")
        except Exception as error:
            handle_error(self, error)

    def _show_storage_info(self) -> None:
        data = self.get_data()
        show_info(
            self,
            "CSV STORAGE\n"
            + "=" * 44
            + f"\nData file   : {dm.DATA_FILE}"
            + f"\nSample file : {dm.SAMPLE_FILE}"
            + f"\nColumns     : {', '.join(dm.COLUMNS)}"
            + f"\nObservations: {len(data)}"
            + "\n\nEverything is stored on this computer only.",
            title="Data storage",
        )

    def _show_about(self) -> None:
        show_info(
            self,
            f"{APP_TITLE}\n\n"
            "Academic project built with:\n"
            "- Python standard library + Tkinter (interface)\n"
            "- Pandas (CSV storage, filtering, grouping)\n"
            "- NumPy (statistics, z-score anomaly detection)\n"
            "- Matplotlib (visual reports)\n\n"
            "Runs fully offline: no database, no API, no internet access.",
            title="About",
        )


# ---------------------------------------------------------------------------
# Add observation
# ---------------------------------------------------------------------------
class AddObservationWindow(BaseWindow):
    """Form for recording a new air-quality observation."""

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Add Observation", geometry="640x560")
        self.entries: dict[str, ttk.Entry] = {}
        self._build_form()

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Clear", self.clear_form)
        self.add_footer_button("Save Observation", self.save, default=True)

        for entry in self.entries.values():
            entry.bind("<Return>", lambda _event: self.save())

    def _build_form(self) -> None:
        form = ttk.LabelFrame(self.body, text="Observation details", padding=PAD)
        form.pack(fill="x")

        now = dt.datetime.now()
        defaults = {
            dm.DATE: now.strftime(dm.DATE_FORMAT),
            dm.TIME: now.strftime(dm.TIME_FORMAT),
        }
        for row, (field, hint, unit) in enumerate(FORM_FIELDS):
            text = f"{hint} {unit}".strip()
            self.entries[field] = labelled_entry(
                form, row, field, text, width=26, value=defaults.get(field, ""))

        ttk.Label(self.body, font=SUBTITLE_FONT, justify="left",
                  text=("Numeric fields must be numbers and cannot be negative.\n"
                        "A unique Observation ID is generated automatically when saved.\n"
                        "Press Enter in any field to save the observation.")
                  ).pack(anchor="w", pady=(PAD, 0))

    def clear_form(self) -> None:
        now = dt.datetime.now()
        for field, entry in self.entries.items():
            entry.delete(0, "end")
            if field == dm.DATE:
                entry.insert(0, now.strftime(dm.DATE_FORMAT))
            elif field == dm.TIME:
                entry.insert(0, now.strftime(dm.TIME_FORMAT))
        self.set_status("Form cleared - ready for a new observation.")
        self.entries[dm.LOCATION].focus_set()

    def save(self) -> None:
        raw = {field: entry.get() for field, entry in self.entries.items()}
        try:
            record, errors = dm.validate_observation(raw)
            if errors:
                self.set_status(f"{len(errors)} invalid value(s) - observation not saved.")
                messagebox.showwarning(
                    "Invalid input",
                    "Please correct the following:\n\n"
                    + "\n".join(f"- {item}" for item in errors),
                    parent=self,
                )
                return
            observation_id = dm.save_observation(record)
            self.app.refresh()
            self.set_status(f"Saved as {observation_id}.")
            show_info(self,
                      f"Observation {observation_id} was saved to the CSV file.\n\n"
                      f"Total records now: {len(self.app.get_data())}",
                      title="Observation saved")
            self.clear_form()
        except Exception as error:
            handle_error(self, error)


# ---------------------------------------------------------------------------
# View records
# ---------------------------------------------------------------------------
class ViewRecordsWindow(BaseWindow):
    """Scrollable table with every stored observation."""

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "View Records", geometry="1020x620")
        self.container, self.tree = build_table(self.body, RECORD_COLUMNS, height=18)
        self.container.pack(fill="both", expand=True)

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Refresh", self.refresh)
        self.load_records()

    def load_records(self) -> None:
        try:
            data = self.current_data()
            fill_table(self.tree, analysis.dataframe_to_rows(
                data, [name for name, _ in RECORD_COLUMNS]) if not data.empty else [])
            if data.empty:
                self.set_status("No observations stored yet - use 'Add Observation' first.")
            else:
                self.set_status(f"Showing all {len(data)} observation(s) "
                                "from the local CSV file.")
        except Exception as error:
            handle_error(self, error)

    def refresh(self) -> None:
        self.app.refresh()
        self.load_records()


# ---------------------------------------------------------------------------
# Search and filter
# ---------------------------------------------------------------------------
class SearchFilterWindow(BaseWindow):
    """Filter observations by location, date range, AQI range and time range."""

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Search / Filter Observations", geometry="1020x700")
        self._build_filters()

        self.container, self.tree = build_table(self.body, RECORD_COLUMNS, height=12)
        self.container.pack(fill="both", expand=True, pady=(PAD, 0))

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Clear Filter", self.clear_filter)
        self.add_footer_button("Apply Filter", self.apply_filter, default=True)
        self.apply_filter()  # show every record until a filter is applied

    def _build_filters(self) -> None:
        frame = ttk.LabelFrame(self.body, text="Filter criteria (leave blank to ignore)",
                               padding=PAD)
        frame.pack(fill="x")

        ttk.Label(frame, text="Location:").grid(row=0, column=0, sticky="w", pady=3)
        self.location_box = ttk.Combobox(frame, values=["<any>"], width=20,
                                         state="readonly")
        self.location_box.set("<any>")
        self.location_box.grid(row=0, column=1, sticky="w", pady=3)
        ttk.Label(frame, text="select a stored location", font=SUBTITLE_FONT).grid(
            row=0, column=2, sticky="w", padx=(PAD, PAD * 2), pady=3)

        self.start_date = labelled_entry(frame, 1, "Start Date:", "YYYY-MM-DD", 14)
        self.end_date = labelled_entry(frame, 1, "End Date:", "YYYY-MM-DD", 14,
                                       column_offset=3)
        self.min_aqi = labelled_entry(frame, 2, "Minimum AQI:", "0 - 500", 14)
        self.max_aqi = labelled_entry(frame, 2, "Maximum AQI:", "0 - 500", 14,
                                      column_offset=3)
        self.start_time = labelled_entry(frame, 3, "Time From:", "HH:MM", 14)
        self.end_time = labelled_entry(frame, 3, "Time To:", "HH:MM", 14,
                                       column_offset=3)

        self.update_location_choices()

    def update_location_choices(self) -> None:
        """Refresh the location drop-down from the stored data."""
        self.location_box.configure(
            values=["<any>"] + dm.unique_locations(self.current_data()))

    def _read_filters(self) -> dict:
        location = self.location_box.get().strip()
        return {
            "location": None if location in ("", "<any>") else location,
            "start_date": self.start_date.get().strip() or None,
            "end_date": self.end_date.get().strip() or None,
            "min_aqi": parse_optional_float(self.min_aqi.get(), "Minimum AQI"),
            "max_aqi": parse_optional_float(self.max_aqi.get(), "Maximum AQI"),
            "start_time": self.start_time.get().strip() or None,
            "end_time": self.end_time.get().strip() or None,
        }

    def apply_filter(self) -> None:
        try:
            data = self.current_data()
            result = analysis.filter_data(data, **self._read_filters())
            fill_table(self.tree, analysis.dataframe_to_rows(
                result, [name for name, _ in RECORD_COLUMNS]))
            if data.empty:
                self.set_status("No observations stored yet - nothing to filter.")
            elif result.empty:
                self.set_status("No records matched the filter "
                                f"(out of {len(data)} observations).")
            else:
                self.set_status(f"Showing {len(result)} of {len(data)} observation(s).")
        except Exception as error:
            handle_error(self, error)

    def clear_filter(self) -> None:
        self.location_box.set("<any>")
        for entry in (self.start_date, self.end_date, self.min_aqi, self.max_aqi,
                      self.start_time, self.end_time):
            entry.delete(0, "end")
        fill_table(self.tree, [])
        self.set_status("Filters cleared - click 'Apply Filter' to show all records.")

    def refresh(self) -> None:
        self.app.refresh()
        self.update_location_choices()
        self.apply_filter()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class StatisticsWindow(BaseWindow):
    """Descriptive statistics for AQI, PM2.5 and PM10."""

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Statistical Analysis", geometry="740x700")
        self.container, self.text = build_text_box(self.body, height=26)
        self.container.pack(fill="both", expand=True)

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Refresh", self.refresh)
        self.load_statistics()

    def load_statistics(self) -> None:
        try:
            stats = analysis.calculate_statistics(self.current_data())
            set_text(self.text, analysis.format_statistics_text(stats))
            if stats["total_observations"] == 0:
                self.set_status("No observations available for statistics.")
            else:
                self.set_status(f"Statistics based on {stats['total_observations']} "
                                f"observation(s) from {stats['locations']} location(s).")
        except Exception as error:
            handle_error(self, error)

    def refresh(self) -> None:
        self.app.refresh()
        self.load_statistics()


# ---------------------------------------------------------------------------
# Location comparison
# ---------------------------------------------------------------------------
class LocationComparisonWindow(BaseWindow):
    """Average/min/max AQI and average PM values per location."""

    COLUMNS = [
        (dm.LOCATION, 150),
        ("Observations", 100),
        ("Avg AQI", 90),
        ("Min AQI", 90),
        ("Max AQI", 90),
        ("Avg PM2.5", 100),
        ("Avg PM10", 100),
    ]

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Compare Locations", geometry="900x680")

        self.container, self.tree = build_table(self.body, self.COLUMNS, height=10)
        self.container.pack(fill="both", expand=True)

        text_container, self.text = build_text_box(self.body, height=13)
        text_container.pack(fill="both", expand=True, pady=(PAD, 0))

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Refresh", self.refresh)
        self.load_comparison()

    def load_comparison(self) -> None:
        try:
            comparison = analysis.calculate_location_comparison(self.current_data())
            fill_table(self.tree, analysis.dataframe_to_rows(
                comparison, [name for name, _ in self.COLUMNS]))
            set_text(self.text, analysis.format_location_comparison_text(comparison))
            if comparison.empty:
                self.set_status("No observations available for location comparison.")
            else:
                best, worst = analysis.best_and_worst_location(comparison)
                self.set_status(f"Best average AQI: {best}  |  Worst average AQI: {worst}")
        except Exception as error:
            handle_error(self, error)

    def refresh(self) -> None:
        self.app.refresh()
        self.load_comparison()


# ---------------------------------------------------------------------------
# Period comparison
# ---------------------------------------------------------------------------
class PeriodComparisonWindow(BaseWindow):
    """Compare two inclusive date ranges (day / week / month windows)."""

    COLUMNS = [("Metric", 180), ("Period A", 150), ("Period B", 150),
               ("Difference (B - A)", 150)]

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Compare Time Periods", geometry="900x660")

        start_a, end_a, start_b, end_b = self._default_ranges()
        ranges = ttk.Frame(self.body)
        ranges.pack(fill="x")
        ranges.columnconfigure(0, weight=1)
        ranges.columnconfigure(1, weight=1)

        period_a = ttk.LabelFrame(ranges, text="Period A (earlier range)", padding=PAD)
        period_a.grid(row=0, column=0, sticky="nsew", padx=(0, PAD))
        self.a_start = labelled_entry(period_a, 0, "Start Date:", "YYYY-MM-DD", 14, start_a)
        self.a_end = labelled_entry(period_a, 1, "End Date:", "YYYY-MM-DD", 14, end_a)

        period_b = ttk.LabelFrame(ranges, text="Period B (later range)", padding=PAD)
        period_b.grid(row=0, column=1, sticky="nsew")
        self.b_start = labelled_entry(period_b, 0, "Start Date:", "YYYY-MM-DD", 14, start_b)
        self.b_end = labelled_entry(period_b, 1, "End Date:", "YYYY-MM-DD", 14, end_b)

        self.container, self.tree = build_table(self.body, self.COLUMNS, height=8)
        self.container.pack(fill="both", expand=True, pady=(PAD, 0))

        text_container, self.text = build_text_box(self.body, height=13)
        text_container.pack(fill="both", expand=True, pady=(PAD, 0))

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Compare Periods", self.compare, default=True)
        self.compare()

    def _default_ranges(self) -> tuple[str, str, str, str]:
        """Use the first two weeks covered by the data as a useful default."""
        today = dt.date.today()
        data = self.current_data()
        if data.empty:
            first = today - dt.timedelta(days=13)
        else:
            dates = pd.to_datetime(data[dm.DATE], format=dm.DATE_FORMAT,
                                   errors="coerce").dropna()
            first = dates.min().date() if not dates.empty else today - dt.timedelta(days=13)
        return (
            first.strftime(dm.DATE_FORMAT),
            (first + dt.timedelta(days=6)).strftime(dm.DATE_FORMAT),
            (first + dt.timedelta(days=7)).strftime(dm.DATE_FORMAT),
            (first + dt.timedelta(days=13)).strftime(dm.DATE_FORMAT),
        )

    def compare(self) -> None:
        data = self.current_data()
        if data.empty:
            fill_table(self.tree, [])
            set_text(self.text, analysis.NO_DATA_MESSAGE)
            self.set_status("No observations available for period comparison.")
            return

        try:
            comparison = analysis.calculate_period_comparison(
                self.current_data(),
                (self.a_start.get().strip(), self.a_end.get().strip()),
                (self.b_start.get().strip(), self.b_end.get().strip()),
            )
            table = analysis.period_comparison_table(comparison)
            fill_table(self.tree, analysis.dataframe_to_rows(table, list(table.columns)))
            set_text(self.text, analysis.format_period_comparison_text(comparison))
            self.set_status("Periods compared using average AQI as the primary metric.")
        except Exception as error:
            handle_error(self, error)


# ---------------------------------------------------------------------------
# Poor air quality periods
# ---------------------------------------------------------------------------
class PoorAirQualityWindow(BaseWindow):
    """Highest-AQI observations (comparatively poor air-quality periods)."""

    COLUMNS = [(dm.DATE, 110), (dm.TIME, 80), (dm.LOCATION, 190), (dm.AQI, 80)]

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Poor Air Quality Periods", geometry="880x640")

        controls = ttk.Frame(self.body)
        controls.pack(fill="x")
        ttk.Label(controls, text="Show this many highest-AQI observations:").pack(side="left")
        self.top_n = ttk.Spinbox(controls, from_=1, to=100, width=5)
        self.top_n.set(str(analysis.DEFAULT_TOP_N))
        self.top_n.pack(side="left", padx=PAD)
        ttk.Label(controls, text="(sorted by AQI, highest first)",
                  font=SUBTITLE_FONT).pack(side="left")

        self.container, self.tree = build_table(self.body, self.COLUMNS, height=13)
        self.container.pack(fill="both", expand=True, pady=(PAD, 0))

        text_container, self.text = build_text_box(self.body, height=11)
        text_container.pack(fill="both", expand=True, pady=(PAD, 0))

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Refresh", self.refresh)
        self.add_footer_button("Show", self.load_poor_periods, default=True)
        self.load_poor_periods()

    def load_poor_periods(self) -> None:
        text_value = self.top_n.get().strip()
        try:
            top_n = int(text_value) if text_value else analysis.DEFAULT_TOP_N
        except ValueError:
            handle_error(self, ValueError("The number of observations to show must be a "
                                          "whole number between 1 and 100."))
            return
        if top_n < 1:
            handle_error(self, ValueError("The number of observations to show must be "
                                          "at least 1."))
            return

        try:
            poor = analysis.get_poor_air_quality_periods(self.current_data(), top_n=top_n)
            fill_table(self.tree, analysis.dataframe_to_rows(
                poor, [name for name, _ in self.COLUMNS]))
            set_text(self.text, analysis.format_poor_air_quality_text(poor))
            if poor.empty:
                self.set_status("No observations available to identify poor air quality.")
            else:
                worst = poor.iloc[0]
                self.set_status(f"Highest recorded AQI: {worst[dm.AQI]:.0f} "
                                f"({worst[dm.DATE]} {worst[dm.TIME]}, {worst[dm.LOCATION]})")
        except Exception as error:
            handle_error(self, error)

    def refresh(self) -> None:
        self.app.refresh()
        self.load_poor_periods()


# ---------------------------------------------------------------------------
# Unusual values
# ---------------------------------------------------------------------------
class UnusualValuesWindow(BaseWindow):
    """Z-score based detection of unusually high or low AQI observations."""

    COLUMNS = [(dm.DATE, 110), (dm.TIME, 80), (dm.LOCATION, 170), (dm.AQI, 80),
               ("Z-Score", 90)]

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Detect Unusual Values", geometry="880x640")

        controls = ttk.Frame(self.body)
        controls.pack(fill="x")
        ttk.Label(controls, text="Z-score threshold  |z| >=").pack(side="left")
        self.threshold = ttk.Spinbox(controls, from_=0.5, to=5.0, increment=0.1, width=6)
        self.threshold.set(f"{analysis.DEFAULT_Z_THRESHOLD:g}")
        self.threshold.pack(side="left", padx=PAD)
        ttk.Label(controls, text="z = (AQI - mean) / standard deviation",
                  font=SUBTITLE_FONT).pack(side="left")

        self.container, self.tree = build_table(self.body, self.COLUMNS, height=12)
        self.container.pack(fill="both", expand=True, pady=(PAD, 0))

        text_container, self.text = build_text_box(self.body, height=11)
        text_container.pack(fill="both", expand=True, pady=(PAD, 0))

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Refresh", self.refresh)
        self.add_footer_button("Detect", self.detect, default=True)
        self.detect()

    def detect(self) -> None:
        try:
            threshold = parse_optional_float(self.threshold.get(), "Z-score threshold")
            if threshold is None:
                threshold = analysis.DEFAULT_Z_THRESHOLD
            if threshold <= 0:
                raise ValueError("The z-score threshold must be greater than zero.")

            data = self.current_data()
            anomalies = analysis.detect_unusual_values(data, z_threshold=threshold)
            fill_table(self.tree, analysis.dataframe_to_rows(
                anomalies, [name for name, _ in self.COLUMNS]))

            mean, std = analysis.aqi_z_scores(data)
            if mean is None:
                summary = analysis.NO_DATA_MESSAGE
            elif std is None or std == 0:
                summary = ("Fewer than two valid AQI values (or no variation in the data) - "
                           "z-scores cannot be calculated.")
            else:
                summary = (f"AQI mean = {mean:.2f}\n"
                           f"AQI standard deviation = {std:.2f}\n\n"
                           + analysis.format_unusual_values_text(anomalies, threshold))
            set_text(self.text, summary)

            if data.empty:
                self.set_status("No observations available for anomaly detection.")
            elif anomalies.empty:
                self.set_status("No unusual AQI observations detected.")
            else:
                self.set_status(f"{len(anomalies)} unusual observation(s) detected.")
        except Exception as error:
            handle_error(self, error)

    def refresh(self) -> None:
        self.app.refresh()
        self.detect()


# ---------------------------------------------------------------------------
# Visual reports
# ---------------------------------------------------------------------------
class ReportsWindow(BaseWindow):
    """Embedded Matplotlib charts plus PNG export into the reports/ folder."""

    def __init__(self, app: AirQualityApp) -> None:
        super().__init__(app, "Visual Reports", geometry="1060x760")

        self.summary = tk.StringVar(value="")
        ttk.Label(self.body, textvariable=self.summary, font=SUBTITLE_FONT,
                  justify="left").pack(anchor="w", pady=(0, PAD))

        self.notebook = ttk.Notebook(self.body)
        self.notebook.pack(fill="both", expand=True)
        self._canvases: list = []

        self.add_footer_button("Back", self.close)
        self.add_footer_button("Save PNG Reports", self.save_reports)
        self.add_footer_button("Refresh", self.refresh)
        self._draw_charts()

    def _draw_charts(self) -> None:
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self._canvases = []
        try:
            data = self.current_data()
            if data.empty:
                self.summary.set(analysis.NO_DATA_MESSAGE.replace("\n", " "))
                self.set_status("No observations available for visual reports.")
                return

            charts = [
                ("AQI Trend", viz.create_aqi_trend_figure),
                ("Average AQI by Location", viz.create_location_comparison_figure),
                ("PM2.5 vs PM10", viz.create_pm_scatter_figure),
            ]
            for title, factory in charts:
                figure = factory(data)
                frame = ttk.Frame(self.notebook)
                canvas = FigureCanvasTkAgg(figure, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self.notebook.add(frame, text=title)
                self._canvases.append(canvas)

            self.summary.set(viz.report_summary_text(data))
            self.set_status("Charts drawn locally with Matplotlib and embedded in Tkinter.")
        except Exception as error:
            handle_error(self, error)

    def save_reports(self) -> None:
        try:
            saved = viz.save_all_reports(self.current_data())
            self.set_status(f"{len(saved)} chart(s) saved to the reports/ folder.")
            show_info(
                self,
                "The charts were saved as PNG files in the local reports/ folder:\n\n"
                + "\n".join(f"- {path.name}" for path in saved),
                title="Reports saved",
            )
        except Exception as error:
            handle_error(self, error)

    def refresh(self) -> None:
        self.app.refresh()
        self._draw_charts()


def run() -> None:
    """Create the main window and start the Tkinter event loop."""
    app = AirQualityApp()
    app.mainloop()
