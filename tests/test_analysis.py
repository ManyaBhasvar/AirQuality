"""Unit tests for the analytical functions (run with ``python -m unittest``).

The tests use small hand-written DataFrames so every expected value can be
verified independently of the CSV file on disk.
"""

import unittest

import numpy as np
import pandas as pd

import analysis
from data_manager import AQI, DATE, LOCATION, PM10, PM25, TIME, empty_frame


def sample_frame() -> pd.DataFrame:
    """Six observations in two locations with known AQI values."""
    return pd.DataFrame(
        {
            DATE: ["2026-10-01", "2026-10-01", "2026-10-02",
                   "2026-10-02", "2026-10-03", "2026-10-03"],
            TIME: ["08:00", "14:00", "09:00", "18:00", "07:00", "15:00"],
            LOCATION: ["Campus", "City", "Campus", "City", "Campus", "City"],
            PM25: [20.0, 60.0, 25.0, 70.0, 30.0, 80.0],
            PM10: [40.0, 110.0, 45.0, 120.0, 50.0, 130.0],
            AQI: [50.0, 150.0, 60.0, 160.0, 70.0, 170.0],
        }
    )


class StatisticsTests(unittest.TestCase):
    """Averages, minimum/maximum, range and standard deviation."""

    def setUp(self) -> None:
        self.data = sample_frame()

    def test_average_calculation(self) -> None:
        stats = analysis.calculate_statistics(self.data)
        expected = float(np.mean([50, 150, 60, 160, 70, 170]))
        self.assertAlmostEqual(stats["summaries"][AQI]["mean"], expected, places=6)

    def test_minimum_maximum_and_range(self) -> None:
        stats = analysis.calculate_statistics(self.data)["summaries"][AQI]
        self.assertEqual(stats["minimum"], 50.0)
        self.assertEqual(stats["maximum"], 170.0)
        self.assertEqual(stats["range"], 120.0)
        self.assertEqual(stats["count"], 6)

    def test_standard_deviation_matches_numpy(self) -> None:
        stats = analysis.calculate_statistics(self.data)["summaries"][AQI]
        self.assertAlmostEqual(stats["std"], float(np.std(self.data[AQI], ddof=1)), places=6)

    def test_range_is_not_reported_for_missing_values(self) -> None:
        frame = sample_frame()
        frame.loc[0, AQI] = np.nan
        stats = analysis.calculate_statistics(frame)["summaries"][AQI]
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["range"], 170.0 - 60.0, places=6)

    def test_single_value_has_zero_deviation(self) -> None:
        frame = sample_frame().head(1)
        block = analysis.summarise_values(analysis.numeric_values(frame, AQI), AQI)
        self.assertEqual(block["std"], 0.0)
        self.assertAlmostEqual(block["range"], 0.0)

    def test_summary_text_contains_values(self) -> None:
        text = analysis.format_statistics_text(analysis.calculate_statistics(self.data))
        self.assertIn("AIR QUALITY SUMMARY", text)
        self.assertIn("Average", text)
        self.assertIn("Range", text)


class LocationComparisonTests(unittest.TestCase):
    """Location grouping and best/worst identification."""

    def setUp(self) -> None:
        self.comparison = analysis.calculate_location_comparison(sample_frame())

    def test_one_row_per_location(self) -> None:
        self.assertEqual(len(self.comparison), 2)
        self.assertEqual(set(self.comparison[LOCATION]), {"Campus", "City"})

    def test_average_aqi_per_location(self) -> None:
        means = dict(zip(self.comparison[LOCATION], self.comparison["Avg AQI"]))
        self.assertAlmostEqual(means["Campus"], (50 + 60 + 70) / 3, places=6)
        self.assertAlmostEqual(means["City"], (150 + 160 + 170) / 3, places=6)

    def test_average_particulates_per_location(self) -> None:
        row = self.comparison[self.comparison[LOCATION] == "Campus"].iloc[0]
        self.assertAlmostEqual(row["Avg PM2.5"], (20 + 25 + 30) / 3, places=6)
        self.assertAlmostEqual(row["Avg PM10"], (40 + 45 + 50) / 3, places=6)

    def test_best_and_worst_location(self) -> None:
        best, worst = analysis.best_and_worst_location(self.comparison)
        self.assertEqual(best, "Campus")   # lowest average AQI
        self.assertEqual(worst, "City")    # highest average AQI

    def test_worst_location_is_listed_first(self) -> None:
        self.assertEqual(self.comparison.iloc[0][LOCATION], "City")


class PeriodComparisonTests(unittest.TestCase):
    """Comparison of two date ranges."""

    def test_period_summaries(self) -> None:
        result = analysis.calculate_period_comparison(
            sample_frame(),
            ("2026-10-01", "2026-10-01"),
            ("2026-10-03", "2026-10-03"),
        )
        self.assertEqual(result["a"]["count"], 2)
        self.assertEqual(result["b"]["count"], 2)
        self.assertAlmostEqual(result["a"]["mean"], (50 + 150) / 2, places=6)
        self.assertAlmostEqual(result["b"]["mean"], (70 + 170) / 2, places=6)
        self.assertAlmostEqual(result["difference"], 20.0, places=6)

    def test_comparison_table_has_difference_column(self) -> None:
        result = analysis.calculate_period_comparison(
            sample_frame(),
            ("2026-10-01", "2026-10-02"),
            ("2026-10-03", "2026-10-03"),
        )
        table = analysis.period_comparison_table(result)
        self.assertIn("Difference (B - A)", table.columns)
        self.assertIn("Metric", table.columns)
        self.assertEqual(len(table), 6)

    def test_empty_period_is_reported_as_no_data(self) -> None:
        result = analysis.calculate_period_comparison(
            sample_frame(),
            ("2026-01-01", "2026-01-02"),
            ("2026-10-01", "2026-10-01"),
        )
        self.assertEqual(result["a"]["count"], 0)
        self.assertIsNone(result["a"]["mean"])
        self.assertIsNone(result["difference"])

    def test_reversed_period_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.calculate_period_comparison(
                sample_frame(),
                ("2026-10-05", "2026-10-01"),
                ("2026-10-06", "2026-10-07"),
            )


class PoorAirQualityTests(unittest.TestCase):
    """Identification of the periods with the highest AQI."""

    def setUp(self) -> None:
        self.poor = analysis.get_poor_air_quality_periods(sample_frame(), top_n=3)

    def test_sorted_descending(self) -> None:
        values = list(self.poor[AQI])
        self.assertEqual(values, sorted(values, reverse=True))

    def test_top_n_is_respected(self) -> None:
        self.assertEqual(len(self.poor), 3)
        self.assertEqual(self.poor.iloc[0][AQI], 170.0)

    def test_highest_observation_is_reported_in_text(self) -> None:
        text = analysis.format_poor_air_quality_text(self.poor)
        self.assertIn("Highest recorded AQI: 170", text)
        self.assertIn("City", text)

    def test_top_n_larger_than_data_returns_everything(self) -> None:
        poor = analysis.get_poor_air_quality_periods(sample_frame(), top_n=100)
        self.assertEqual(len(poor), 6)


class UnusualValueTests(unittest.TestCase):
    """Z-score based anomaly detection."""

    def test_known_z_score(self) -> None:
        frame = sample_frame()
        anomalies = analysis.detect_unusual_values(frame, z_threshold=1.0)
        self.assertFalse(anomalies.empty)

        values = analysis.numeric_values(frame, AQI)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        for _, row in anomalies.iterrows():
            expected = (row[AQI] - mean) / std
            self.assertAlmostEqual(row["Z-Score"], round(expected, 3), places=3)

        # Both extremes of the sample are further than one deviation away.
        self.assertEqual(sorted(anomalies[AQI]), [50.0, 170.0])
        self.assertEqual(anomalies.iloc[0]["Z-Score"], min(anomalies["Z-Score"]))

    def test_no_anomalies_returns_empty_table(self) -> None:
        anomalies = analysis.detect_unusual_values(sample_frame(), z_threshold=3.0)
        self.assertTrue(anomalies.empty)
        self.assertEqual(list(anomalies.columns), analysis.ANOMALY_COLUMNS)
        self.assertEqual(analysis.format_unusual_values_text(anomalies),
                         "No unusual AQI observations detected.")

    def test_identical_values_are_never_unusual(self) -> None:
        frame = sample_frame()
        frame[AQI] = 100.0
        anomalies = analysis.detect_unusual_values(frame, z_threshold=2.0)
        self.assertTrue(anomalies.empty)

    def test_low_outlier_is_detected_too(self) -> None:
        # An unusually clean reading must be flagged as well as a polluted one.
        extra = {DATE: "2026-10-04", TIME: "12:00", LOCATION: "Campus",
                 PM25: 5.0, PM10: 10.0, AQI: 5.0}
        frame = pd.concat([sample_frame(), pd.DataFrame([extra])], ignore_index=True)

        anomalies = analysis.detect_unusual_values(frame, z_threshold=1.0)
        self.assertFalse(anomalies.empty)
        self.assertLess(anomalies["Z-Score"].min(), 0)
        self.assertIn(5.0, list(anomalies[AQI]))


class FilterTests(unittest.TestCase):
    """Searching and filtering with Pandas."""

    def setUp(self) -> None:
        self.data = sample_frame()

    def test_filter_by_location(self) -> None:
        result = analysis.filter_data(self.data, location="campus")
        self.assertEqual(len(result), 3)
        self.assertTrue((result[LOCATION] == "Campus").all())

    def test_filter_by_date_range(self) -> None:
        result = analysis.filter_data(self.data, start_date="2026-10-02",
                                      end_date="2026-10-02")
        self.assertEqual(len(result), 2)
        self.assertTrue((result[DATE] == "2026-10-02").all())

    def test_filter_by_aqi_range(self) -> None:
        result = analysis.filter_data(self.data, min_aqi=100, max_aqi=165)
        self.assertEqual(sorted(result[AQI]), [150.0, 160.0])

    def test_filter_by_time_range(self) -> None:
        result = analysis.filter_data(self.data, start_time="12:00", end_time="18:00")
        self.assertEqual(sorted(result[TIME]), ["14:00", "15:00", "18:00"])

    def test_combined_filters(self) -> None:
        result = analysis.filter_data(self.data, location="City", min_aqi=155,
                                      end_date="2026-10-02")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0][AQI], 160.0)

    def test_no_filters_returns_all_rows(self) -> None:
        self.assertEqual(len(analysis.filter_data(self.data)), len(self.data))

    def test_invalid_date_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            analysis.filter_data(self.data, start_date="02-10-2026")

    def test_reversed_aqi_range_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            analysis.filter_data(self.data, min_aqi=200, max_aqi=100)


class EmptyDataTests(unittest.TestCase):
    """Every analysis must cope with an empty dataset."""

    def setUp(self) -> None:
        self.empty = empty_frame()

    def test_is_empty(self) -> None:
        self.assertTrue(analysis.is_empty(self.empty))
        self.assertTrue(analysis.is_empty(None))

    def test_statistics_on_empty_data(self) -> None:
        stats = analysis.calculate_statistics(self.empty)
        self.assertEqual(stats["total_observations"], 0)
        self.assertIsNone(stats["summaries"][AQI]["mean"])
        self.assertEqual(analysis.format_statistics_text(stats), analysis.NO_DATA_MESSAGE)

    def test_other_reports_on_empty_data(self) -> None:
        self.assertTrue(analysis.calculate_location_comparison(self.empty).empty)
        self.assertTrue(analysis.get_poor_air_quality_periods(self.empty).empty)
        self.assertTrue(analysis.detect_unusual_values(self.empty).empty)
        self.assertTrue(analysis.filter_data(self.empty).empty)

    def test_text_reports_on_empty_data(self) -> None:
        self.assertEqual(analysis.format_location_comparison_text(self.empty),
                         analysis.NO_DATA_MESSAGE)
        self.assertEqual(analysis.format_poor_air_quality_text(self.empty),
                         analysis.NO_DATA_MESSAGE)

    def test_period_comparison_on_empty_data_raises(self) -> None:
        with self.assertRaises(analysis.NoDataError):
            analysis.calculate_period_comparison(
                self.empty, ("2026-10-01", "2026-10-02"), ("2026-10-03", "2026-10-04"))

    def test_best_and_worst_of_empty_comparison(self) -> None:
        self.assertEqual(analysis.best_and_worst_location(self.empty), (None, None))

    def test_dataframe_helpers_on_empty_data(self) -> None:
        self.assertEqual(analysis.dataframe_to_rows(self.empty, [AQI]), [])
        self.assertEqual(analysis.numeric_values(self.empty, AQI).size, 0)
        self.assertEqual(analysis.aqi_z_scores(self.empty), (None, None))


if __name__ == "__main__":
    unittest.main()
