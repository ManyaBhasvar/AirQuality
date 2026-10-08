"""Local CSV storage and input validation for the Air Quality Analysis System.

This module owns everything that touches the file system:

* creating the data directory / CSV file when it is missing,
* reading the CSV into a :class:`pandas.DataFrame`,
* appending a new observation,
* generating unique observation IDs,
* validating raw user input coming from the GUI.

Only the Python standard library plus NumPy/Pandas are used.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column names used in the CSV file (kept identical to the required layout)
# ---------------------------------------------------------------------------
OBS_ID = "Observation ID"
DATE = "Date"
TIME = "Time"
LOCATION = "Location"
PM25 = "PM2.5"
PM10 = "PM10"
CO = "CO"
NO2 = "NO2"
SO2 = "SO2"
O3 = "O3"
AQI = "AQI"

COLUMNS = [OBS_ID, DATE, TIME, LOCATION, PM25, PM10, CO, NO2, SO2, O3, AQI]
NUMERIC_COLUMNS = [PM25, PM10, CO, NO2, SO2, O3, AQI]
TEXT_COLUMNS = [OBS_ID, DATE, TIME, LOCATION]

# Sensible physical bounds for every measured quantity.
# Values outside these bounds are rejected as impossible / unrealistic.
MEASUREMENT_LIMITS = {
    PM25: (0.0, 1000.0),
    PM10: (0.0, 2000.0),
    CO: (0.0, 100.0),
    NO2: (0.0, 2000.0),
    SO2: (0.0, 2000.0),
    O3: (0.0, 2000.0),
    AQI: (0.0, 500.0),
}

MAX_LOCATION_LENGTH = 60
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"

# Project paths (relative to this file, never absolute/hard-coded).
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "air_quality_data.csv"
SAMPLE_FILE = DATA_DIR / "sample_air_quality_data.csv"


class DataError(Exception):
    """Raised when the CSV file is missing, unreadable or corrupted."""


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------
def ensure_data_file(path: Path | str = DATA_FILE) -> Path:
    """Create the data directory and CSV file (with a header) if needed."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Write only the header row so the empty file is still valid CSV.
            path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
    except OSError as exc:
        raise DataError(f"Could not create the data file:\n{exc}") from exc
    return path


def empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns and dtypes."""
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})
    return frame


def load_data(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Read the CSV file into a DataFrame, creating it when necessary.

    The returned frame always contains the expected columns and numeric
    measurement columns (invalid text becomes ``NaN`` instead of crashing).
    """
    path = Path(path)
    ensure_data_file(path)

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return empty_frame()
    except pd.errors.ParserError as exc:
        raise DataError(f"The data file appears to be corrupted:\n{exc}") from exc
    except OSError as exc:
        raise DataError(f"The data file could not be read:\n{exc}") from exc

    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(
            "The data file is missing these columns: " + ", ".join(missing)
        )

    frame = frame[COLUMNS].copy()
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Ignore completely blank lines produced by manual editing of the file.
    frame = frame[frame[DATE] != ""].reset_index(drop=True)
    return frame


def append_observation(observation: dict, path: Path | str = DATA_FILE) -> None:
    """Append a single observation dictionary to the CSV file."""
    path = Path(path)
    ensure_data_file(path)
    unknown = [key for key in observation if key not in COLUMNS]
    if unknown:
        raise DataError("Unknown CSV column(s): " + ", ".join(unknown))
    try:
        frame = pd.DataFrame([{column: observation.get(column, "") for column in COLUMNS}])
        frame.to_csv(path, mode="a", header=False, index=False)
    except OSError as exc:
        raise DataError(f"The observation could not be saved:\n{exc}") from exc


def save_observation(observation: dict, path: Path | str = DATA_FILE) -> str:
    """Save one observation, assigning a fresh observation ID.

    Returns the generated ID. The input dictionary is not modified.
    """
    record = dict(observation)
    record[OBS_ID] = generate_observation_id(path=path)
    append_observation(record, path=path)
    return record[OBS_ID]


def generate_observation_id(path: Path | str = DATA_FILE,
                           frame: pd.DataFrame | None = None) -> str:
    """Return the next sequential observation ID, e.g. ``AQ-0007``."""
    if frame is None:
        frame = load_data(path)
    highest = 0
    for value in frame[OBS_ID].astype(str):
        digits = "".join(character for character in value if character.isdigit())
        if digits:
            highest = max(highest, int(digits))
    return f"AQ-{highest + 1:04d}"


def replace_all(frame: pd.DataFrame, path: Path | str = DATA_FILE) -> None:
    """Overwrite the CSV file with the given DataFrame (used by tests/demo)."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    except OSError as exc:
        raise DataError(f"The data file could not be written:\n{exc}") from exc


def load_sample_data(sample_path: Path | str = SAMPLE_FILE) -> pd.DataFrame:
    """Load the bundled demo dataset (clearly labelled sample data)."""
    sample_path = Path(sample_path)
    if not sample_path.exists():
        raise DataError(f"Sample data file not found:\n{sample_path}")
    return load_data(sample_path)


def import_sample_data(path: Path | str = DATA_FILE,
                       sample_path: Path | str = SAMPLE_FILE) -> int:
    """Append the demo rows to the working CSV and return the number imported."""
    sample = load_sample_data(sample_path)
    if sample.empty:
        return 0
    existing = load_data(path)
    start = 1
    try:
        start = int(generate_observation_id(frame=existing).split("-")[1])
    except (IndexError, ValueError):
        start = len(existing) + 1

    rows = []
    for offset, (_, row) in enumerate(sample.iterrows()):
        record = {column: row[column] for column in COLUMNS}
        record[OBS_ID] = f"AQ-{start + offset:04d}"
        rows.append(record)

    merged = pd.concat([existing, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    replace_all(merged, path)
    return len(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _parse_with_format(text: str, date_format: str):
    """Parse a date/time string, returning ``None`` when it is not valid.

    ``pandas`` returns ``NaT`` for empty strings instead of raising, so the
    result is checked explicitly as well.
    """
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, format=date_format)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(parsed) else parsed


def validate_observation(raw: dict) -> tuple[dict | None, list[str]]:
    """Validate one raw observation coming from the GUI.

    Returns ``(cleaned_record, errors)``. When ``errors`` is not empty the
    cleaned record is ``None`` and the caller should show the messages.
    """
    errors: list[str] = []
    cleaned: dict = {}

    # Date
    date_text = _clean_text(raw.get(DATE))
    parsed_date = _parse_with_format(date_text, DATE_FORMAT)
    if parsed_date is None:
        errors.append(f"Date must be a valid date in the form YYYY-MM-DD (got {date_text!r}).")
    else:
        cleaned[DATE] = parsed_date.strftime(DATE_FORMAT)

    # Time
    time_text = _clean_text(raw.get(TIME))
    parsed_time = _parse_with_format(time_text, TIME_FORMAT)
    if parsed_time is None:
        errors.append(f"Time must be a valid time in 24-hour form HH:MM (got {time_text!r}).")
    else:
        cleaned[TIME] = parsed_time.strftime(TIME_FORMAT)

    # Location
    location_text = _clean_text(raw.get(LOCATION))
    if not location_text:
        errors.append("Location must not be empty.")
    elif len(location_text) > MAX_LOCATION_LENGTH:
        errors.append(f"Location must be at most {MAX_LOCATION_LENGTH} characters long.")
    else:
        cleaned[LOCATION] = location_text

    # Measurements
    for column in NUMERIC_COLUMNS:
        value_text = _clean_text(raw.get(column))
        if value_text == "":
            errors.append(f"{column} is required and must be numeric.")
            continue
        try:
            value = float(value_text)
        except ValueError:
            errors.append(f"{column} must be numeric (got {value_text!r}).")
            continue
        if not math.isfinite(value):
            errors.append(f"{column} must be a finite number.")
            continue
        low, high = MEASUREMENT_LIMITS[column]
        if value < low or value > high:
            errors.append(f"{column} must be between {low:g} and {high:g} (got {value:g}).")
            continue
        cleaned[column] = value

    if errors:
        return None, errors
    return cleaned, []


def unique_locations(frame: pd.DataFrame) -> list[str]:
    """Return the sorted list of locations present in the data."""
    if frame is None or frame.empty:
        return []
    values = frame[LOCATION].dropna().astype(str).str.strip()
    return sorted({value for value in values if value})


def observation_datetimes(frame: pd.DataFrame) -> pd.Series:
    """Return a datetime Series combining the Date and Time columns."""
    if frame is None or frame.empty:
        return pd.Series(dtype="datetime64[ns]")
    text = frame[DATE].astype(str).str.strip() + " " + frame[TIME].astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        # Fall back to flexible parsing for records written by hand.
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
    return parsed
