"""Matplotlib visual reports for the Air Quality Analysis System.

Three charts are produced (all drawing happens locally, nothing is uploaded):

1. ``AQI Trend Over Time``      - line chart of AQI against date/time.
2. ``Average AQI by Location``  - bar chart comparing locations.
3. ``PM2.5 vs PM10``            - scatter plot of the two particle sizes.

Every chart is available in three forms:

* ``create_*_figure()`` -> returns a :class:`matplotlib.figure.Figure` that the
  Tkinter GUI embeds with ``FigureCanvasTkAgg``,
* ``save_figure()`` / ``save_all_reports()`` -> writes a PNG into ``reports/``,
* ``plot_*()`` -> opens the chart in a standalone Matplotlib window.

The drawing routines (``_draw_*``) are shared by all variants, so the embedded
chart and the saved PNG always look identical.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from analysis import NO_DATA_MESSAGE, NoDataError
from data_manager import AQI, DATE, LOCATION, PM10, PM25, TIME, observation_datetimes

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

TREND_FILENAME = "aqi_trend.png"
LOCATION_FILENAME = "location_comparison.png"
SCATTER_FILENAME = "pm25_vs_pm10.png"

MAX_TREND_LINES = 8  # more locations than this are aggregated into one line
UNHEALTHY_AQI = 200  # observations at/above this AQI are highlighted in red
FIGURE_SIZE = (8.0, 4.6)
NO_AQI_MESSAGE = "No observations with a valid AQI value."


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------
def _require_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Raise a friendly error when there is nothing to plot."""
    if frame is None or frame.empty:
        raise NoDataError(NO_DATA_MESSAGE)
    return frame


def trend_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observations with a parsed datetime column, sorted in time.

    Records without a usable AQI value are dropped so the chart stays clean.
    """
    working = frame.copy()
    working["ObservationTime"] = observation_datetimes(working)
    working[AQI] = pd.to_numeric(working[AQI], errors="coerce")
    working = working.dropna(subset=["ObservationTime", AQI])
    return working.sort_values("ObservationTime").reset_index(drop=True)


def location_averages(frame: pd.DataFrame) -> pd.DataFrame:
    """Return average AQI per location, sorted from best to worst."""
    working = frame.copy()
    working[AQI] = pd.to_numeric(working[AQI], errors="coerce")
    working = working.dropna(subset=[AQI])
    if working.empty:
        return pd.DataFrame(columns=[LOCATION, "mean", "count"])
    return (
        working.groupby(LOCATION)[AQI]
        .agg(["mean", "count"])
        .sort_values("mean")
        .reset_index()
    )


def _paired_particulates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned PM2.5 / PM10 NumPy arrays for rows measuring both."""
    working = frame.copy()
    working[PM25] = pd.to_numeric(working[PM25], errors="coerce")
    working[PM10] = pd.to_numeric(working[PM10], errors="coerce")
    working = working.dropna(subset=[PM25, PM10])
    return (
        working[PM25].to_numpy(dtype=float),
        working[PM10].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Chart 1 - AQI trend over time
# ---------------------------------------------------------------------------
def _draw_aqi_trend(axes, frame: pd.DataFrame) -> bool:
    """Draw AQI over time; returns False when no record can be plotted."""
    data = trend_data(frame)
    if data.empty:
        return False

    locations = sorted(data[LOCATION].astype(str).unique())
    if len(locations) <= MAX_TREND_LINES:
        for location in locations:
            subset = data[data[LOCATION].astype(str) == location]
            axes.plot(
                subset["ObservationTime"],
                subset[AQI],
                marker="o",
                markersize=4,
                linewidth=1.4,
                label=location,
            )
    else:
        # Too many locations for a readable legend: one aggregated daily line.
        daily = data.set_index("ObservationTime")[AQI].resample("D").mean().dropna()
        axes.plot(daily.index, daily.to_numpy(), marker="o", markersize=4,
                  linewidth=1.6, color="#1f77b4", label="Daily average AQI")

    # Highlight clearly unhealthy observations so the reader notices them.
    worst = data[data[AQI] >= UNHEALTHY_AQI]
    if not worst.empty:
        axes.scatter(worst["ObservationTime"], worst[AQI], color="red", zorder=5,
                     label=f"AQI >= {UNHEALTHY_AQI}")

    axes.set_title("AQI Trend Over Time")
    axes.set_xlabel("Date / Time")
    axes.set_ylabel("AQI (Air Quality Index)")
    axes.grid(True, linestyle=":", alpha=0.6)
    axes.legend(loc="best", fontsize=8)
    return True


# ---------------------------------------------------------------------------
# Chart 2 - average AQI by location
# ---------------------------------------------------------------------------
def _draw_location_bars(axes, frame: pd.DataFrame) -> bool:
    """Draw average AQI per location; returns False when nothing can be drawn."""
    averages = location_averages(frame)
    if averages.empty:
        return False

    labels = [str(value) for value in averages[LOCATION]]
    values = averages["mean"].to_numpy(dtype=float)
    colors = ["#4c72b0"] * len(values)
    colors[int(np.argmax(values))] = "#c44e52"  # worst location highlighted

    bars = axes.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    axes.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    axes.set_title("Average AQI by Location")
    axes.set_xlabel("Location")
    axes.set_ylabel("Average AQI")
    axes.set_ylim(0, float(np.max(values)) * 1.15 if np.max(values) > 0 else 1)
    axes.grid(True, axis="y", linestyle=":", alpha=0.6)
    axes.legend(
        handles=[Rectangle((0, 0), 1, 1, color="#4c72b0"),
                 Rectangle((0, 0), 1, 1, color="#c44e52")],
        labels=["Location", "Highest average AQI"],
        fontsize=8,
    )
    if len(labels) > 6:
        axes.tick_params(axis="x", rotation=30)
    return True


# ---------------------------------------------------------------------------
# Chart 3 - PM2.5 vs PM10
# ---------------------------------------------------------------------------
def _draw_pm_scatter(axes, frame: pd.DataFrame) -> bool:
    """Draw PM2.5 against PM10; returns False when nothing can be drawn."""
    x, y = _paired_particulates(frame)
    if x.size < 2 or np.ptp(x) == 0:
        return False

    axes.scatter(x, y, s=28, color="#55a868", edgecolor="black", linewidth=0.5,
                 label="Observation")

    # Least-squares trend line, computed directly with NumPy.
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 50)
    axes.plot(x_line, slope * x_line + intercept, color="#c44e52", linewidth=1.6,
              label=f"Least-squares fit (slope {slope:.2f})")
    correlation = float(np.corrcoef(x, y)[0, 1])

    axes.set_title("PM2.5 vs PM10 Relationship")
    axes.set_xlabel("PM2.5 (µg/m³)")
    axes.set_ylabel("PM10 (µg/m³)")
    axes.grid(True, linestyle=":", alpha=0.6)
    axes.legend(loc="best", fontsize=8)
    axes.text(
        0.02, 0.95,
        f"Pearson correlation: {correlation:.3f}\nObservations: {x.size}",
        transform=axes.transAxes, fontsize=8, va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    return True


# ---------------------------------------------------------------------------
# Shared figure builder
# ---------------------------------------------------------------------------
def _build(frame: pd.DataFrame, drawer, empty_message: str, *, embed: bool,
           rotate_dates: bool = False) -> Figure:
    """Build a chart either as an embeddable Figure or a pyplot window figure."""
    frame = _require_data(frame)
    if embed:
        figure = Figure(figsize=FIGURE_SIZE, dpi=100)
        axes = figure.add_subplot(111)
    else:
        figure, axes = plt.subplots(figsize=FIGURE_SIZE, dpi=100)

    if not drawer(axes, frame):
        axes.text(0.5, 0.5, empty_message, ha="center", va="center", wrap=True)
        axes.axis("off")
    elif rotate_dates:
        figure.autofmt_xdate(rotation=30)

    figure.tight_layout()
    return figure


def create_aqi_trend_figure(frame: pd.DataFrame) -> Figure:
    """Build the 'AQI Trend Over Time' line chart as an embeddable Figure."""
    return _build(frame, _draw_aqi_trend, NO_AQI_MESSAGE, embed=True, rotate_dates=True)


def create_location_comparison_figure(frame: pd.DataFrame) -> Figure:
    """Build the 'Average AQI by Location' bar chart as an embeddable Figure."""
    return _build(frame, _draw_location_bars, NO_AQI_MESSAGE, embed=True)


def create_pm_scatter_figure(frame: pd.DataFrame) -> Figure:
    """Build the PM2.5 vs PM10 scatter plot as an embeddable Figure."""
    return _build(
        frame, _draw_pm_scatter,
        "At least two different paired PM2.5 / PM10 readings are needed.",
        embed=True,
    )


# ---------------------------------------------------------------------------
# Saving and standalone display
# ---------------------------------------------------------------------------
def save_figure(figure: Figure, filename: str, directory: Path | str = REPORTS_DIR) -> Path:
    """Save a figure as a PNG inside the reports directory."""
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        figure.savefig(path, dpi=100, bbox_inches="tight")
    except OSError as exc:
        raise NoDataError(f"The report image could not be saved:\n{exc}") from exc
    return path


def save_all_reports(frame: pd.DataFrame, directory: Path | str = REPORTS_DIR) -> list[Path]:
    """Create and save every chart; returns the list of written PNG paths."""
    _require_data(frame)
    figures = [
        (create_aqi_trend_figure(frame), TREND_FILENAME),
        (create_location_comparison_figure(frame), LOCATION_FILENAME),
        (create_pm_scatter_figure(frame), SCATTER_FILENAME),
    ]
    saved = [save_figure(figure, name, directory) for figure, name in figures]
    for figure, _ in figures:
        plt.close(figure)
    return saved


def plot_aqi_trend(frame: pd.DataFrame) -> Figure:
    """Open the AQI trend chart in a standalone Matplotlib window."""
    figure = _build(frame, _draw_aqi_trend, NO_AQI_MESSAGE, embed=False, rotate_dates=True)
    figure.show()
    return figure


def plot_location_comparison(frame: pd.DataFrame) -> Figure:
    """Open the location comparison chart in a standalone Matplotlib window."""
    figure = _build(frame, _draw_location_bars, NO_AQI_MESSAGE, embed=False)
    figure.show()
    return figure


def plot_pm_scatter(frame: pd.DataFrame) -> Figure:
    """Open the PM2.5 vs PM10 chart in a standalone Matplotlib window."""
    figure = _build(
        frame, _draw_pm_scatter,
        "At least two different paired PM2.5 / PM10 readings are needed.",
        embed=False,
    )
    figure.show()
    return figure


def report_summary_text(frame: pd.DataFrame) -> str:
    """Return a short text summary describing the visual reports."""
    if frame is None or frame.empty:
        return NO_DATA_MESSAGE
    data = trend_data(frame)
    if data.empty:
        return NO_DATA_MESSAGE
    first = data["ObservationTime"].min()
    last = data["ObservationTime"].max()
    return (
        "VISUAL REPORTS\n"
        + "=" * 40
        + f"\nObservations plotted : {len(data)}"
        + f"\nPeriod covered       : {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}"
        + f"\nLocations            : {data[LOCATION].nunique()}"
        + "\n\nCharts: AQI Trend Over Time | Average AQI by Location | PM2.5 vs PM10"
        + "\nUse 'Save PNG Reports' to write the charts into the reports/ folder."
        + f"\nData source: local CSV file ({DATE} / {TIME} columns)."
    )
