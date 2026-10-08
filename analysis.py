"""Analytical functions for the Air Quality Analysis System.

The module contains no GUI code and no file handling. It receives a
``pandas.DataFrame`` (as produced by :mod:`data_manager`) and returns either
plain data structures (dictionaries / DataFrames) or ready-to-display text.

Pandas is used for loading/selecting/grouping/sorting records, NumPy is used
for the numerical summaries and the z-score anomaly detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_manager import (
    AQI,
    DATE,
    LOCATION,
    NUMERIC_COLUMNS,
    PM10,
    PM25,
    TIME,
    observation_datetimes,
)

#: Default number of rows shown by the "poor air quality" report.
DEFAULT_TOP_N = 10

#: Default z-score threshold for anomaly detection.
DEFAULT_Z_THRESHOLD = 2.0

#: Columns shown in the poor-air-quality table.
POOR_AIR_COLUMNS = [DATE, TIME, LOCATION, AQI]

#: Columns shown in the unusual-values table.
ANOMALY_COLUMNS = [DATE, TIME, LOCATION, AQI, "Z-Score"]

NO_DATA_MESSAGE = (
    "No air-quality observations are currently available.\n"
    "Please add observations before performing analysis."
)


class NoDataError(Exception):
    """Raised when an analysis is requested but no usable observations exist."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def is_empty(frame: pd.DataFrame | None) -> bool:
    """Return True when a DataFrame is missing or has no rows."""
    return frame is None or frame.empty


def numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return the finite numeric values of ``column`` as a NumPy array."""
    if is_empty(frame) or column not in frame.columns:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def summarise_values(values: np.ndarray, label: str = "") -> dict:
    """Calculate mean/min/max/range/standard deviation for a NumPy array.

    The standard deviation is the sample standard deviation (``ddof=1``),
    which matches ``pandas``/``statistics.stdev``. When fewer than two values
    are available it falls back to ``0.0`` instead of dividing by zero.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "label": label,
            "count": 0,
            "mean": None,
            "minimum": None,
            "maximum": None,
            "range": None,
            "std": None,
        }
    return {
        "label": label,
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.ptp(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Filtering / searching
# ---------------------------------------------------------------------------
def filter_data(
    frame: pd.DataFrame,
    location: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_aqi: float | None = None,
    max_aqi: float | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Return the subset of observations matching every supplied filter.

    Unset (``None``) filters are ignored, so an empty call returns all data.
    Invalid filter values raise :class:`ValueError` with a readable message.
    """
    if is_empty(frame):
        return frame.copy()

    result = frame
    date_values = observation_datetimes(result).dt.normalize()

    if location:
        wanted = location.strip().lower()
        result = result[result[LOCATION].astype(str).str.strip().str.lower() == wanted]

    if start_date:
        bound = _parse_day(start_date, "Start date")
        result = result[date_values.loc[result.index] >= bound]
    if end_date:
        bound = _parse_day(end_date, "End date")
        result = result[date_values.loc[result.index] <= bound]

    if start_time or end_time:
        # Times are normalised to HH:MM so the comparison is reliable even when
        # the CSV was edited by hand (e.g. '9:30' instead of '09:30').
        clock = _clock_column(result)
        if start_time:
            bound = _parse_clock(start_time, "Start time")
            result = result[clock >= bound]
            clock = clock.loc[result.index]
        if end_time:
            bound = _parse_clock(end_time, "End time")
            result = result[clock <= bound]

    if min_aqi is not None or max_aqi is not None:
        aqi = pd.to_numeric(result[AQI], errors="coerce")
        if min_aqi is not None and max_aqi is not None and min_aqi > max_aqi:
            raise ValueError("Minimum AQI must not be greater than Maximum AQI.")
        if min_aqi is not None:
            result = result[aqi >= min_aqi]
            aqi = pd.to_numeric(result[AQI], errors="coerce")
        if max_aqi is not None:
            result = result[aqi <= max_aqi]

    return result.reset_index(drop=True)


def _parse_day(text: str, label: str) -> pd.Timestamp:
    """Convert a date string to a Timestamp, raising a readable error."""
    try:
        parsed = pd.to_datetime(str(text).strip(), format="%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a valid date (YYYY-MM-DD).") from exc
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a valid date (YYYY-MM-DD).")
    return pd.Timestamp(parsed)


def _clock_column(frame: pd.DataFrame) -> pd.Series:
    """Return the Time column normalised to zero-padded ``HH:MM`` strings."""
    parsed = pd.to_datetime(frame[TIME].astype(str).str.strip(),
                            format="%H:%M", errors="coerce")
    return parsed.dt.strftime("%H:%M").fillna("")


def _parse_clock(text: str, label: str) -> str:
    """Convert a time string to ``HH:MM``, raising a readable error."""
    try:
        parsed = pd.to_datetime(str(text).strip(), format="%H:%M")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a valid time (HH:MM).") from exc
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a valid time (HH:MM).")
    return parsed.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------
def calculate_statistics(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = (AQI, PM25, PM10),
) -> dict:
    """Calculate descriptive statistics for the main air-quality indicators."""
    summary: dict = {
        "total_observations": 0,
        "locations": 0,
        "summaries": {},
    }
    for column in columns:
        summary["summaries"][column] = summarise_values(numeric_values(frame, column), column)
    if is_empty(frame):
        return summary
    summary["total_observations"] = int(len(frame))
    summary["locations"] = int(frame[LOCATION].nunique())
    summary["date_range"] = (str(frame[DATE].min()), str(frame[DATE].max()))
    return summary


def format_statistics_text(stats: dict) -> str:
    """Render :func:`calculate_statistics` output as plain report text."""
    lines = ["AIR QUALITY SUMMARY", "=" * 40]
    if not stats or stats.get("total_observations", 0) == 0:
        return NO_DATA_MESSAGE

    lines.append(f"Total Observations : {stats['total_observations']}")
    lines.append(f"Locations          : {stats['locations']}")
    if "date_range" in stats:
        first, last = stats["date_range"]
        lines.append(f"Date Range         : {first} to {last}")
    lines.append("")

    for label, block in stats.get("summaries", {}).items():
        lines.append(label)
        lines.append("-" * 40)
        if block.get("count", 0) == 0:
            lines.append("  No valid values recorded.")
            lines.append("")
            continue
        lines.append(f"  Valid Values       : {block['count']}")
        lines.append(f"  Average            : {block['mean']:.2f}")
        lines.append(f"  Minimum            : {block['minimum']:.2f}")
        lines.append(f"  Maximum            : {block['maximum']:.2f}")
        lines.append(f"  Range              : {block['range']:.2f}")
        if label == AQI:
            lines.append(f"  Standard Deviation : {block['std']:.2f}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Location comparison
# ---------------------------------------------------------------------------
def calculate_location_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare locations using average AQI (primary metric) and pollutants."""
    columns = [LOCATION, "Observations", "Avg AQI", "Min AQI", "Max AQI",
               "Avg PM2.5", "Avg PM10"]
    if is_empty(frame) or AQI not in frame.columns or LOCATION not in frame.columns:
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    aggregations = {
        "Observations": (AQI, "size"),
        "Avg AQI": (AQI, "mean"),
        "Min AQI": (AQI, "min"),
        "Max AQI": (AQI, "max"),
    }
    if PM25 in working.columns:
        aggregations["Avg PM2.5"] = (PM25, "mean")
    if PM10 in working.columns:
        aggregations["Avg PM10"] = (PM10, "mean")

    grouped = working.groupby(LOCATION, dropna=True).agg(**aggregations).round(2)
    grouped = grouped.reset_index()
    for column in ("Avg PM2.5", "Avg PM10"):
        if column not in grouped.columns:
            grouped[column] = float("nan")
    grouped = grouped.sort_values("Avg AQI", ascending=False).reset_index(drop=True)
    return grouped[columns]


def best_and_worst_location(comparison: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return ``(best, worst)`` location names based on average AQI."""
    if is_empty(comparison):
        return None, None
    valid = comparison.dropna(subset=["Avg AQI"])
    if valid.empty:
        return None, None
    worst = str(valid.iloc[0][LOCATION])
    best = str(valid.iloc[-1][LOCATION])
    return best, worst


def format_location_comparison_text(comparison: pd.DataFrame) -> str:
    """Render the location comparison as an aligned plain-text table."""
    if is_empty(comparison):
        return NO_DATA_MESSAGE

    header = (
        f"{'Location':<16}{'Obs':>6}{'Avg AQI':>10}{'Min AQI':>10}"
        f"{'Max AQI':>10}{'Avg PM2.5':>12}{'Avg PM10':>11}"
    )
    lines = ["LOCATION COMPARISON", "=" * len(header), header, "-" * len(header)]
    for _, row in comparison.iterrows():
        lines.append(
            f"{str(row[LOCATION])[:15]:<16}"
            f"{int(row['Observations']):>6}"
            f"{row['Avg AQI']:>10.2f}"
            f"{row['Min AQI']:>10.2f}"
            f"{row['Max AQI']:>10.2f}"
            f"{row['Avg PM2.5']:>12.2f}"
            f"{row['Avg PM10']:>11.2f}"
        )
    best, worst = best_and_worst_location(comparison)
    lines.append("")
    lines.append(f"Best air quality (lowest average AQI) : {best}")
    lines.append(f"Worst air quality (highest average AQI): {worst}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Period comparison
# ---------------------------------------------------------------------------
def calculate_period_comparison(
    frame: pd.DataFrame,
    period_a: tuple[str, str],
    period_b: tuple[str, str],
    labels: tuple[str, str] = ("Period A", "Period B"),
) -> dict:
    """Compare two inclusive date ranges, e.g. two weeks or two months.

    Each period is described by ``(start_date, end_date)`` in ``YYYY-MM-DD``
    form. Returns a dictionary with one summary per period plus the
    difference in average AQI (``B - A``).
    """
    if is_empty(frame):
        raise NoDataError(NO_DATA_MESSAGE)

    start_a, end_a = period_a
    start_b, end_b = period_b
    if _parse_day(start_a, "Period A start") > _parse_day(end_a, "Period A end"):
        raise ValueError("Period A start date must not be after its end date.")
    if _parse_day(start_b, "Period B start") > _parse_day(end_b, "Period B end"):
        raise ValueError("Period B start date must not be after its end date.")

    results = {}
    for key, (start, end), label in (
        ("a", (start_a, end_a), labels[0]),
        ("b", (start_b, end_b), labels[1]),
    ):
        subset = filter_data(frame, start_date=start, end_date=end)
        block = summarise_values(numeric_values(subset, AQI), label)
        block["pm25"] = summarise_values(numeric_values(subset, PM25), PM25)
        block["pm10"] = summarise_values(numeric_values(subset, PM10), PM10)
        block["period"] = (start, end)
        block["label"] = label
        results[key] = block

    mean_a = results["a"]["mean"]
    mean_b = results["b"]["mean"]
    results["difference"] = None if mean_a is None or mean_b is None else mean_b - mean_a
    return results


def _number(value, digits: int = 2) -> str:
    """Format an optional number for display ('n/a' when missing)."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def period_comparison_table(comparison: dict) -> pd.DataFrame:
    """Return a display table: Metric | Period A | Period B | Difference (B - A)."""
    labels = (
        (comparison or {}).get("a", {}).get("label", "Period A"),
        (comparison or {}).get("b", {}).get("label", "Period B"),
    )
    columns = ["Metric", labels[0], labels[1], "Difference (B - A)"]
    if not comparison:
        return pd.DataFrame(columns=columns)

    period_a, period_b = comparison["a"], comparison["b"]
    metrics = [
        ("Observations", period_a["count"], period_b["count"], 0),
        ("Average AQI", period_a["mean"], period_b["mean"], 2),
        ("Minimum AQI", period_a["minimum"], period_b["minimum"], 2),
        ("Maximum AQI", period_a["maximum"], period_b["maximum"], 2),
        ("Average PM2.5", period_a["pm25"]["mean"], period_b["pm25"]["mean"], 2),
        ("Average PM10", period_a["pm10"]["mean"], period_b["pm10"]["mean"], 2),
    ]
    rows = []
    for metric, left, right, digits in metrics:
        if left is None or right is None:
            difference = "n/a"
        else:
            difference = f"{float(right) - float(left):+.{digits}f}"
        rows.append({
            "Metric": metric,
            labels[0]: _number(left, digits),
            labels[1]: _number(right, digits),
            "Difference (B - A)": difference,
        })
    return pd.DataFrame(rows, columns=columns)


def format_period_comparison_text(comparison: dict) -> str:
    """Render :func:`calculate_period_comparison` output as report text."""
    if not comparison:
        return NO_DATA_MESSAGE

    lines = ["PERIOD COMPARISON", "=" * 46]
    for key in ("a", "b"):
        block = comparison[key]
        lines.append(f"{block['label']}  ({block['period'][0]} to {block['period'][1]})")
        lines.append("-" * 46)
        if block["count"] == 0:
            lines.append("  No observations recorded in this period.")
        else:
            lines.append(f"  Observations   : {block['count']}")
            lines.append(f"  Average AQI    : {block['mean']:.2f}")
            lines.append(f"  Minimum AQI    : {block['minimum']:.2f}")
            lines.append(f"  Maximum AQI    : {block['maximum']:.2f}")
            lines.append(f"  Range          : {block['range']:.2f}")
            lines.append(f"  Average PM2.5  : {block['pm25']['mean']:.2f}")
            lines.append(f"  Average PM10   : {block['pm10']['mean']:.2f}")
        lines.append("")

    difference = comparison.get("difference")
    if difference is None:
        lines.append("Difference cannot be calculated (one period has no data).")
    else:
        better = comparison["a"]["label"] if difference > 0 else comparison["b"]["label"]
        lines.append(f"Change in average AQI (B - A): {difference:+.2f}")
        lines.append(f"Better air quality in this comparison: {better}")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Poor air quality periods
# ---------------------------------------------------------------------------
def get_poor_air_quality_periods(frame: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Return observations with the highest AQI, sorted descending."""
    if is_empty(frame):
        return pd.DataFrame(columns=POOR_AIR_COLUMNS)

    working = frame.copy()
    working[AQI] = pd.to_numeric(working[AQI], errors="coerce")
    working = working.dropna(subset=[AQI]).sort_values(AQI, ascending=False)
    if top_n is not None and top_n > 0:
        working = working.head(int(top_n))
    return working[POOR_AIR_COLUMNS].reset_index(drop=True)


def format_poor_air_quality_text(poor: pd.DataFrame) -> str:
    """Render the poor-air-quality table and the single worst observation."""
    if is_empty(poor):
        return NO_DATA_MESSAGE

    header = f"{'Date':<12}{'Time':<8}{'Location':<18}{'AQI':>7}"
    lines = ["POOR AIR QUALITY PERIODS", "=" * len(header), header, "-" * len(header)]
    for _, row in poor.iterrows():
        lines.append(
            f"{str(row[DATE]):<12}{str(row[TIME]):<8}"
            f"{str(row[LOCATION])[:17]:<18}{row[AQI]:>7.0f}"
        )
    worst = poor.iloc[0]
    lines.append("")
    lines.append(
        f"Highest recorded AQI: {worst[AQI]:.0f} on {worst[DATE]} at {worst[TIME]} "
        f"in {worst[LOCATION]}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unusual value (anomaly) detection
# ---------------------------------------------------------------------------
def detect_unusual_values(frame: pd.DataFrame,
                          z_threshold: float = DEFAULT_Z_THRESHOLD) -> pd.DataFrame:
    """Flag observations whose AQI z-score is ``abs(z) >= z_threshold``.

    The z-score is ``(value - mean) / standard_deviation``. A simple,
    explainable statistical rule is used - no machine learning.
    """
    if is_empty(frame):
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    working = frame.copy()
    working[AQI] = pd.to_numeric(working[AQI], errors="coerce")
    working = working.dropna(subset=[AQI])
    values = working[AQI].to_numpy(dtype=float)
    if values.size < 2:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        # Every observation is identical: nothing can be "unusual".
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    working["Z-Score"] = (values - mean) / std
    flagged = working[np.abs(working["Z-Score"]) >= float(z_threshold)].copy()
    if flagged.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    flagged = flagged.sort_values("Z-Score", key=lambda series: np.abs(series), ascending=False)
    flagged["Z-Score"] = flagged["Z-Score"].round(3)
    return flagged[ANOMALY_COLUMNS].reset_index(drop=True)


def format_unusual_values_text(anomalies: pd.DataFrame,
                              z_threshold: float = DEFAULT_Z_THRESHOLD) -> str:
    """Render the anomaly table, or a clear 'nothing found' message."""
    if is_empty(anomalies):
        return "No unusual AQI observations detected."

    header = f"{'Date':<12}{'Time':<8}{'Location':<18}{'AQI':>7}{'Z-Score':>10}"
    lines = [
        "UNUSUAL OBSERVATIONS",
        "=" * len(header),
        f"Method: z-score of AQI, flagging |z| >= {float(z_threshold):g}",
        "-" * len(header),
        header,
        "-" * len(header),
    ]
    for _, row in anomalies.iterrows():
        lines.append(
            f"{str(row[DATE]):<12}{str(row[TIME]):<8}"
            f"{str(row[LOCATION])[:17]:<18}{row[AQI]:>7.0f}{row['Z-Score']:>10.3f}"
        )
    return "\n".join(lines)


def aqi_z_scores(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return the AQI ``(mean, sample standard deviation)`` used for z-scores."""
    values = numeric_values(frame, AQI)
    if values.size < 2:
        return (float(values[0]) if values.size else None), None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def dataframe_to_rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Convert selected DataFrame columns into a list of display tuples."""
    if is_empty(frame):
        return []
    rows = []
    for _, row in frame.iterrows():
        row_values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (int, np.integer)):
                row_values.append(str(value))
            elif isinstance(value, (float, np.floating)):
                row_values.append("" if pd.isna(value) else f"{value:g}")
            else:
                row_values.append("" if value is None else str(value))
        rows.append(tuple(row_values))
    return rows
