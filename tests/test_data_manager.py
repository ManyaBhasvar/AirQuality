"""Unit tests for the CSV storage and validation layer.

Every test works inside a temporary directory, so the real
``data/air_quality_data.csv`` file is never modified.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import data_manager as dm


def valid_input() -> dict:
    """A complete, valid observation as typed into the GUI form."""
    return {
        dm.DATE: "2026-10-05",
        dm.TIME: "14:30",
        dm.LOCATION: "Campus Library",
        dm.PM25: "42.5",
        dm.PM10: "68",
        dm.CO: "0.8",
        dm.NO2: "25.5",
        dm.SO2: "9",
        dm.O3: "60",
        dm.AQI: "152",
    }


class ValidationTests(unittest.TestCase):
    """Input validation for every form field."""

    def test_valid_input_is_accepted(self) -> None:
        record, errors = dm.validate_observation(valid_input())
        self.assertEqual(errors, [])
        self.assertIsNotNone(record)
        self.assertEqual(record[dm.DATE], "2026-10-05")
        self.assertEqual(record[dm.LOCATION], "Campus Library")
        self.assertAlmostEqual(record[dm.AQI], 152.0)

    def test_invalid_date_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.DATE] = "05/10/2026"
        record, errors = dm.validate_observation(raw)
        self.assertIsNone(record)
        self.assertTrue(any("Date" in message for message in errors))

    def test_impossible_date_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.DATE] = "2026-02-30"
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any("Date" in message for message in errors))

    def test_invalid_time_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.TIME] = "25:99"
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any("Time" in message for message in errors))

    def test_empty_location_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.LOCATION] = "   "
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any("Location" in message for message in errors))

    def test_non_numeric_measurement_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.PM25] = "bad value"
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any(dm.PM25 in message for message in errors))

    def test_negative_measurement_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.PM10] = "-5"
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any(dm.PM10 in message for message in errors))

    def test_out_of_range_aqi_is_rejected(self) -> None:
        raw = valid_input()
        raw[dm.AQI] = "900"
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any(dm.AQI in message for message in errors))

    def test_missing_measurements_are_reported(self) -> None:
        raw = valid_input()
        raw[dm.O3] = ""
        _record, errors = dm.validate_observation(raw)
        self.assertTrue(any(dm.O3 in message for message in errors))

    def test_all_errors_are_collected_together(self) -> None:
        _record, errors = dm.validate_observation({})
        self.assertEqual(len(errors), len(dm.COLUMNS) - 1)  # every field but the ID


class StorageTests(unittest.TestCase):
    """File creation, appending, ID generation and corrupted files."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.path = Path(self._temp_dir.name) / "data" / "air_quality_data.csv"

    def test_missing_file_is_created_with_header(self) -> None:
        created = dm.ensure_data_file(self.path)
        self.assertTrue(created.exists())
        self.assertEqual(created.read_text(encoding="utf-8").strip(),
                         ",".join(dm.COLUMNS))

    def test_loading_an_empty_file_gives_an_empty_frame(self) -> None:
        frame = dm.load_data(self.path)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), dm.COLUMNS)

    def test_save_and_reload_observation(self) -> None:
        record, errors = dm.validate_observation(valid_input())
        self.assertEqual(errors, [])
        observation_id = dm.save_observation(record, path=self.path)
        self.assertEqual(observation_id, "AQ-0001")

        frame = dm.load_data(self.path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0][dm.OBS_ID], "AQ-0001")
        self.assertEqual(frame.iloc[0][dm.LOCATION], "Campus Library")
        self.assertEqual(frame.iloc[0][dm.AQI], 152.0)

    def test_observation_ids_increase(self) -> None:
        record, _errors = dm.validate_observation(valid_input())
        first = dm.save_observation(record, path=self.path)
        second = dm.save_observation(record, path=self.path)
        third = dm.save_observation(record, path=self.path)
        self.assertEqual([first, second, third], ["AQ-0001", "AQ-0002", "AQ-0003"])

    def test_unknown_column_is_rejected(self) -> None:
        with self.assertRaises(dm.DataError):
            dm.append_observation({"Not A Column": "x"}, path=self.path)

    def test_file_missing_a_column_is_reported(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("Observation ID,Date,AQI\nAQ-0001,2026-10-01,100\n",
                             encoding="utf-8")
        with self.assertRaises(dm.DataError):
            dm.load_data(self.path)

    def test_unparsable_values_become_missing_instead_of_crashing(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            ",".join(dm.COLUMNS) + "\n"
            "AQ-0001,2026-10-01,10:00,Campus,not-a-number,50,1,10,5,40,120\n",
            encoding="utf-8",
        )
        frame = dm.load_data(self.path)
        self.assertEqual(len(frame), 1)
        self.assertTrue(pd.isna(frame.iloc[0][dm.PM25]))
        self.assertEqual(frame.iloc[0][dm.AQI], 120.0)

    def test_blank_lines_are_ignored(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            ",".join(dm.COLUMNS) + "\n,,,,,,,,,,\n", encoding="utf-8")
        self.assertTrue(dm.load_data(self.path).empty)

    def test_demo_data_import(self) -> None:
        sample = Path(self._temp_dir.name) / "sample.csv"
        record, _errors = dm.validate_observation(valid_input())
        dm.replace_all(pd.DataFrame([{**record, dm.OBS_ID: "AQ-0001"}],
                                    columns=dm.COLUMNS), sample)
        imported = dm.import_sample_data(path=self.path, sample_path=sample)
        self.assertEqual(imported, 1)
        frame = dm.load_data(self.path)
        self.assertEqual(len(frame), 1)

    def test_missing_sample_file_is_reported(self) -> None:
        with self.assertRaises(dm.DataError):
            dm.load_sample_data(Path(self._temp_dir.name) / "missing.csv")

    def test_unique_locations_and_datetimes(self) -> None:
        record, _errors = dm.validate_observation(valid_input())
        dm.save_observation(record, path=self.path)
        second = dict(record, **{dm.LOCATION: "City Centre", dm.OBS_ID: "AQ-0002"})
        dm.append_observation(second, path=self.path)

        frame = dm.load_data(self.path)
        self.assertEqual(dm.unique_locations(frame), ["Campus Library", "City Centre"])

        moments = dm.observation_datetimes(frame)
        self.assertEqual(len(moments), 2)
        self.assertFalse(moments.isna().any())
        self.assertEqual(moments.iloc[0].strftime("%Y-%m-%d %H:%M"), "2026-10-05 14:30")

    def test_datetime_helper_on_empty_frame(self) -> None:
        self.assertEqual(len(dm.observation_datetimes(dm.empty_frame())), 0)


class BundledDataTests(unittest.TestCase):
    """The shipped demo dataset must stay loadable and consistent."""

    def test_sample_file_has_expected_shape(self) -> None:
        frame = dm.load_sample_data()
        self.assertGreater(len(frame), 0)
        self.assertEqual(list(frame.columns), dm.COLUMNS)
        self.assertTrue(np.isfinite(frame[dm.AQI]).all())
        self.assertTrue((frame[dm.AQI] >= 0).all())

    def test_working_file_matches_expected_columns(self) -> None:
        frame = dm.load_data()
        self.assertEqual(list(frame.columns), dm.COLUMNS)


if __name__ == "__main__":
    unittest.main()
