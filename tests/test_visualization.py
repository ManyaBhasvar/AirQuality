"""Unit tests for the Matplotlib visual reports.

The Agg backend is selected so the tests never need to open a window.
"""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis  # noqa: E402
import visualization as viz  # noqa: E402
from data_manager import AQI, DATE, LOCATION, PM10, PM25, TIME, empty_frame  # noqa: E402


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE: ["2026-10-01", "2026-10-01", "2026-10-02", "2026-10-03"],
            TIME: ["08:00", "14:00", "09:00", "15:00"],
            LOCATION: ["Campus", "City", "Campus", "City"],
            PM25: [20.0, 60.0, 25.0, 80.0],
            PM10: [40.0, 110.0, 45.0, 130.0],
            AQI: [50.0, 150.0, 60.0, 210.0],
        }
    )


class FigureTests(unittest.TestCase):
    """Every chart must build without a display and carry proper labels."""

    def setUp(self) -> None:
        self.data = sample_frame()

    def test_trend_chart_labels(self) -> None:
        figure = viz.create_aqi_trend_figure(self.data)
        axes = figure.axes[0]
        self.assertEqual(axes.get_title(), "AQI Trend Over Time")
        self.assertEqual(axes.get_xlabel(), "Date / Time")
        self.assertIn("AQI", axes.get_ylabel())
        self.assertIsNotNone(axes.get_legend())
        plt.close(figure)

    def test_location_chart_labels(self) -> None:
        figure = viz.create_location_comparison_figure(self.data)
        axes = figure.axes[0]
        self.assertEqual(axes.get_title(), "Average AQI by Location")
        self.assertEqual(axes.get_xlabel(), "Location")
        self.assertEqual(axes.get_ylabel(), "Average AQI")
        self.assertEqual(len(axes.patches), 2)  # one bar per location
        plt.close(figure)

    def test_scatter_chart_labels(self) -> None:
        figure = viz.create_pm_scatter_figure(self.data)
        axes = figure.axes[0]
        self.assertEqual(axes.get_title(), "PM2.5 vs PM10 Relationship")
        self.assertIn("PM2.5", axes.get_xlabel())
        self.assertIn("PM10", axes.get_ylabel())
        plt.close(figure)

    def test_trend_uses_numpy_for_many_locations(self) -> None:
        frame = sample_frame()
        frame[LOCATION] = [f"Location {index}" for index in range(len(frame))]
        figure = viz.create_aqi_trend_figure(frame)
        self.assertTrue(figure.axes[0].lines)
        plt.close(figure)

    def test_missing_aqi_values_give_a_message_figure(self) -> None:
        frame = sample_frame()
        frame[AQI] = np.nan
        figure = viz.create_aqi_trend_figure(frame)
        self.assertTrue(figure.axes[0].texts)
        plt.close(figure)

    def test_empty_data_raises_friendly_error(self) -> None:
        with self.assertRaises(analysis.NoDataError):
            viz.create_aqi_trend_figure(empty_frame())
        with self.assertRaises(analysis.NoDataError):
            viz.create_location_comparison_figure(empty_frame())
        with self.assertRaises(analysis.NoDataError):
            viz.save_all_reports(empty_frame())


class SaveReportTests(unittest.TestCase):
    """Charts must be written as local PNG files."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.directory = Path(self._temp_dir.name) / "reports"

    def test_all_reports_are_saved(self) -> None:
        saved = viz.save_all_reports(sample_frame(), directory=self.directory)
        self.assertEqual(len(saved), 3)
        for path in saved:
            self.assertTrue(path.exists(), f"{path} was not written")
            self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(
            sorted(path.name for path in saved),
            sorted([viz.TREND_FILENAME, viz.LOCATION_FILENAME, viz.SCATTER_FILENAME]),
        )

    def test_save_figure_uses_given_filename(self) -> None:
        figure = viz.create_aqi_trend_figure(sample_frame())
        path = viz.save_figure(figure, "custom_name.png", directory=self.directory)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "custom_name.png")
        plt.close(figure)

    def test_report_summary_text(self) -> None:
        text = viz.report_summary_text(sample_frame())
        self.assertIn("VISUAL REPORTS", text)
        self.assertIn("AQI Trend Over Time", text)
        self.assertEqual(viz.report_summary_text(empty_frame()), analysis.NO_DATA_MESSAGE)


class HelperTests(unittest.TestCase):
    """Data preparation helpers used by the charts."""

    def test_trend_data_is_sorted_by_time(self) -> None:
        data = viz.trend_data(sample_frame())
        self.assertEqual(len(data), 4)
        self.assertTrue(data["ObservationTime"].is_monotonic_increasing)

    def test_location_averages_sorted_ascending(self) -> None:
        averages = viz.location_averages(sample_frame())
        self.assertEqual(list(averages[LOCATION]), ["Campus", "City"])
        self.assertAlmostEqual(averages.iloc[0]["mean"], (50 + 60) / 2, places=6)

    def test_location_averages_of_empty_frame(self) -> None:
        self.assertTrue(viz.location_averages(empty_frame()).empty)


if __name__ == "__main__":
    unittest.main()
