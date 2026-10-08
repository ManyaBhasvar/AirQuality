# Air Quality Analysis System

An offline desktop application (Python + Tkinter) that records air-quality
observations in a local CSV file and provides statistical summaries,
comparisons, anomaly detection and Matplotlib visual reports.

The application runs entirely on one computer: no database, no web server,
no API, no cloud service and no internet connection are used.

---

## 1. Problem Statement

An institution records air-quality observations at different times and wants to
understand the variations in air quality and identify the periods with
comparatively poor air quality.

## 2. Objective

Develop an application that stores air-quality observations and provides
analytical summaries: averages, minimum/maximum values, ranges, comparisons
between locations and time periods, unusual-value detection,
poor-air-quality identification and meaningful visual reports.

---

## 3. Features

| # | Feature | Where |
|---|---------|-------|
| 1 | Record air-quality observations (validated form) | Add Observation |
| 2 | Store observations persistently in a local CSV file | `data/air_quality_data.csv` |
| 3 | View all stored records in a scrollable table | View Records |
| 4 | Search / filter by location, date range, AQI range, time range | Search / Filter |
| 5 | Calculate averages (AQI, PM2.5, PM10) | Statistics |
| 6 | Calculate minimum, maximum, range and standard deviation | Statistics |
| 7 | Compare locations (count, avg/min/max AQI, avg PM2.5, avg PM10, best/worst) | Compare Locations |
| 8 | Compare two time periods (date ranges) with the difference | Compare Periods |
| 9 | Identify poor air-quality periods (highest AQI first) | Poor Air Quality |
| 10 | Detect unusual values with an explainable z-score rule | Unusual Values |
| 11 | Visual report 1: AQI Trend Over Time (line chart) | Visual Reports |
| 12 | Visual report 2: Average AQI by Location (bar chart) | Visual Reports |
| 13 | Extra report: PM2.5 vs PM10 (scatter plot + least-squares fit) | Visual Reports |
| 14 | Save the charts as PNG files in `reports/` | Visual Reports |
| 15 | Simple Tkinter graphical user interface (10 buttons, menu bar) | Main window |
| 16 | Input validation with clear, non-technical error messages | Add Observation, filters |
| 17 | Graceful error handling (missing / corrupted CSV, empty data, bad filters) | everywhere |
| 18 | Bundled demo data set that can be loaded with one click | Load Demo Data |
| 19 | Unit tests with the built-in `unittest` framework | `tests/` |

---

## 4. Technology Used

Only the following technologies are used — nothing else:

* **Python 3** (tested with Python 3.13)
* **Tkinter** (standard-library GUI toolkit; `tkinter`, `tkinter.ttk`, `tkinter.messagebox`, `tkinter.font`)
* **NumPy** (mean, minimum, maximum, range via `np.ptp`, standard deviation, z-scores, `np.polyfit`, `np.corrcoef`)
* **Pandas** (CSV reading/writing, filtering, grouping by location, sorting, date handling)
* **Matplotlib** (line chart, bar chart, scatter plot, saved PNG files, `FigureCanvasTkAgg` embedding)
* **Python standard library** (`pathlib`, `datetime`, `math`, `tempfile`, `unittest`, `sys`, …)

There is **no** database, web framework, web browser requirement, JavaScript,
HTML, CSS, API, cloud service, external charting library or machine-learning
library in this project.

---

## 5. Installation

Python 3.10 or newer is required (3.13 was used for development).
Tkinter ships with the standard CPython installer for Windows and macOS.
On Debian/Ubuntu it may need to be installed once with
`sudo apt install python3-tk`.

Install the three permitted third-party packages (only needed once):

```text
pip install numpy pandas matplotlib
```

No other package is required, and nothing is downloaded while the program runs.

---

## 6. Running the Application

From inside the `air_quality_project` folder:

```text
python main.py
```

The Tkinter main window opens immediately. The main menu offers:

```text
Add Observation      |  View Records
Search / Filter      |  Statistics
Compare Locations    |  Compare Periods
Poor Air Quality     |  Unusual Values
Visual Reports       |  Load Demo Data
```

The data file is created automatically on the first start, so the application
also works on a completely empty installation. The status bar always shows how
many records are stored and where the file lives.

> The bundled data set is **sample/demo data** (42 observations for three
> invented locations). It exists only to demonstrate the analysis and can be
> removed with **File → Clear All Data…**. Real observations can be added at any
> time with **Add Observation**; the application never depends on the demo data.

---

## 7. Project Structure

```text
air_quality_project/
│
├── main.py                 # starts the application (creates the CSV, opens the GUI)
├── data_manager.py         # CSV storage, observation IDs, input validation
├── analysis.py             # statistics, filtering, comparisons, z-score anomalies
├── visualization.py        # the Matplotlib charts (embedded + saved PNG files)
├── gui.py                  # all Tkinter screens and widgets
│
├── data/
│   ├── air_quality_data.csv        # the working data file (created automatically)
│   └── sample_air_quality_data.csv # bundled DEMO data (sample only)
│
├── reports/                # generated chart images (aqi_trend.png, ...)
│
├── tests/
│   ├── __init__.py
│   ├── test_analysis.py         # 38 tests for the analytical functions
│   ├── test_data_manager.py     # 24 tests for validation and CSV storage
│   ├── test_visualization.py    # 12 tests for the charts
│   └── test_gui_smoke.py        # 16 tests that drive the real Tkinter windows
│
└── README.md
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `main.py` | Entry point: prepares the data file and starts the Tkinter application. Kept intentionally small. |
| `data_manager.py` | Creates the data directory/CSV file, loads it into a `pandas.DataFrame`, appends observations, generates IDs (`AQ-0001`, `AQ-0002`, …), validates every form field, converts file problems into a readable `DataError`. |
| `analysis.py` | All calculations: descriptive statistics, filtering, location comparison, period comparison, poor-air-quality ranking, z-score anomaly detection and the plain-text report formatting. No file or GUI code. |
| `visualization.py` | The three Matplotlib charts. Each chart can be returned as a `Figure` (for embedding in Tkinter), saved as a PNG, or shown in a standalone window. |
| `gui.py` | The Tkinter interface: main window, ten screens, tables, forms, status bars, validation messages and error dialogs. |

---

## 8. Data Storage

Observations are stored in **one local CSV file** — no database is used.

```text
data/air_quality_data.csv
```

| Column | Example | Meaning |
|--------|---------|---------|
| `Observation ID` | `AQ-0007` | unique ID, generated automatically |
| `Date` | `2026-10-05` | `YYYY-MM-DD` |
| `Time` | `14:30` | 24-hour `HH:MM` |
| `Location` | `Campus Library` | non-empty text |
| `PM2.5` | `42.5` | µg/m³, 0 – 1000 |
| `PM10` | `68.0` | µg/m³, 0 – 2000 |
| `CO` | `0.8` | mg/m³, 0 – 100 |
| `NO2` | `25.5` | µg/m³, 0 – 2000 |
| `SO2` | `9.0` | µg/m³, 0 – 2000 |
| `O3` | `60.0` | µg/m³, 0 – 2000 |
| `AQI` | `152` | overall air-quality index, 0 – 500 |

* `data_manager.ensure_data_file()` creates the `data/` folder and a CSV
  containing only the header row when the file is missing, so the application
  never fails on a fresh installation.
* Pandas reads the file (`pd.read_csv`) and writes new observations with
  `to_csv(mode="a")`, which appends a single row without rewriting the file.
* Unreadable numbers inside the CSV are converted to *missing* values by
  `pd.to_numeric(errors="coerce")` instead of crashing the program.
* A file with missing columns or invalid content raises `DataError`, which the
  GUI shows as a friendly "Data file error" dialog.
* `AQI` is the main indicator used for comparisons, ranking and anomalies.

---

## 9. Analysis

### Descriptive statistics (`calculate_statistics`)
For **AQI**, **PM2.5** and **PM10**: number of valid values, average (mean),
minimum, maximum, range (`np.ptp`) and — for AQI — the standard deviation
(`np.std(..., ddof=1)`, the sample standard deviation).

### Filtering (`filter_data`)
Location (case-insensitive), start/end date (inclusive), minimum/maximum AQI and
a time-of-day window. Filters are optional and combinable; invalid values raise
a readable `ValueError` which the GUI shows as a warning. Times are normalised
to `HH:MM`, so hand-edited values such as `9:30` are still filtered correctly.

### Location comparison (`calculate_location_comparison`)
Pandas `groupby("Location")` with named aggregations produces one row per
location: observations, average/minimum/maximum AQI and average PM2.5/PM10,
sorted from worst to best. `best_and_worst_location()` names the best location
(lowest average AQI) and the worst (highest).

### Period comparison (`calculate_period_comparison`)
Two inclusive date ranges are filtered separately and summarised with the same
metrics; the difference in average AQI (`B − A`) and the better period are
reported.

### Poor air-quality periods (`get_poor_air_quality_periods`)
Observations sorted by AQI in descending order (default: top 10); the highest
recorded AQI is highlighted explicitly.

### Unusual values (`detect_unusual_values`)
An explainable statistical rule, **no machine learning**:

```text
z = (AQI − mean(AQI)) / standard_deviation(AQI)
flagged when |z| >= threshold        (default threshold: 2.0)
```

Both unusually *polluted* and unusually *clean* observations are reported.
When the data has no variation (standard deviation = 0) or fewer than two
values, the function returns an empty result and the GUI states
*"No unusual AQI observations detected."* instead of dividing by zero.

### Empty data
Every analysis function accepts an empty dataset and returns empty results,
the message *"No air-quality observations are currently available. Please add
observations before performing analysis."* or a `NoDataError`, which the GUI
turns into an informational dialog. The application never crashes because of
missing data.

---

## 10. Visualizations

Matplotlib is used directly; the charts are embedded in the Tkinter window with
`FigureCanvasTkAgg` (recommended by Matplotlib for Tkinter) and can be saved as
PNG files in `reports/`. No web dashboard or HTML chart is involved.

**Report 1 — AQI Trend Over Time** (`reports/aqi_trend.png`)
Line chart with date/time on the X axis and AQI on the Y axis, one line per
location (up to 8 locations, otherwise one aggregated daily-average line),
observations at or above the unhealthy threshold AQI 200 highlighted in red,
grid, legend and rotated date labels.

**Report 2 — Average AQI by Location** (`reports/location_comparison.png`)
Bar chart with locations on the X axis and the average AQI on the Y axis.
Every bar is labelled with its value; the highest (worst) average AQI is drawn
in red and explained in the legend.

**Additional report — PM2.5 vs PM10 Relationship** (`reports/pm25_vs_pm10.png`)
Scatter plot of PM2.5 against PM10 with a least-squares fit line
(`np.polyfit`) and the Pearson correlation (`np.corrcoef`) shown as an
annotation. This reveals how strongly the two particle fractions move together.

The "Save PNG Reports" button writes all three images into the local `reports/`
folder; nothing is uploaded anywhere.

---

## 11. Testing

The tests use only the built-in `unittest` framework (pytest is **not** used) and
need no internet access.

```text
python -m unittest discover
```

Individual modules can also be run directly, for example:

```text
python -m unittest tests.test_analysis
python -m unittest tests.test_gui_smoke -v
```

What is covered:

| Test module | Coverage |
|-------------|----------|
| `tests/test_analysis.py` | averages, minimum/maximum, range, standard deviation, location comparison, period comparison, poor-air-quality ranking, z-score anomalies, filters, empty datasets |
| `tests/test_data_manager.py` | validation of every field (bad date, impossible date, invalid time, empty location, non-numeric, negative, out-of-range AQI), CSV creation, append round-trip, ID generation, corrupted/short files, demo-data import |
| `tests/test_visualization.py` | the three charts build headlessly (`Agg`), correct titles/axis labels, PNG files are written, empty data raises `NoDataError` |
| `tests/test_gui_smoke.py` | builds the real Tkinter application, opens all nine screens, runs their actions (filter, compare, detect, save), checks that the screens are laid out visibly, and verifies that the form accepts valid input while rejecting invalid input |

Result of the last run: **90 tests, all passing** (`OK`). The GUI tests are
skipped automatically when no display is available, e.g. on a headless server.

Demonstrated results on the bundled demo data (42 observations, 3 locations):
AQI average 135.64, minimum 32, maximum 287, range 255, standard deviation
59.46; best location *Campus Library* (average AQI 70.86), worst *Industrial
Zone* (192.36); two observations flagged as unusual (z = 2.546 and 2.277).

---

## 12. Offline Capability

The application is completely offline:

* it runs locally with `python main.py` and only opens a Tkinter window;
* it stores data in a local CSV file next to the source code;
* it writes chart images into the local `reports/` folder;
* it contains no HTTP request, no API call, no web server, no socket
  communication, no database, no cloud storage, no login and no telemetry;
* it does not download anything at runtime and does not need a browser.

Only `pip install numpy pandas matplotlib` needs internet access, and that is a
one-time setup step performed by the user.

---

## 13. Limitations

* One observation is one row; there is no editing or deleting of a single row
  (Individual rows can be removed by editing the CSV in a text editor, or the
  whole file can be cleared from **File → Clear All Data…**).
* Duplicate rows (same date, time and location) are allowed — the application
  does not enforce uniqueness of the measurement time.
* AQI is taken as entered; the program does not recompute AQI from the
  pollutant concentrations (that would require an official breakpoint table).
* The z-score rule assumes the AQI values are roughly normally distributed and
  needs at least two observations with some variation.
* The period comparison accepts date ranges only (day / week / month windows are
  created by typing the corresponding dates).
* Charts for more than eight locations are aggregated into a single daily
  average line to keep the trend chart readable.
* `reports/` and `data/` must be writable; otherwise a clear "Data file error"
  message is shown.
