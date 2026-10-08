"""Smoke tests for the Tkinter interface.

The tests build the real windows, run the real actions and check the results,
while every message box is replaced by a mock and saving is redirected to a
fake function so the project data file is never modified.

The whole class is skipped automatically when no display is available.
"""

import unittest
from unittest import mock

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - Tkinter is part of CPython
    tk = None


def display_available() -> bool:
    """Return True when a Tk window can be created on this machine."""
    if tk is None:
        return False
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(display_available(), "no display available for Tkinter")
class GuiSmokeTests(unittest.TestCase):
    """Build every screen and run its main action."""

    @classmethod
    def setUpClass(cls) -> None:
        import gui

        cls.gui = gui

    def setUp(self) -> None:
        gui = self.gui
        self.dialogs = {
            name: mock.patch.object(gui.messagebox, name, return_value=True).start()
            for name in ("showinfo", "showerror", "showwarning", "askyesno")
        }
        self.addCleanup(mock.patch.stopall)

        self.app = gui.AirQualityApp()
        self.app.withdraw()
        self.addCleanup(self._destroy_app)

    def _destroy_app(self) -> None:
        try:
            self.app.close_app()
        except Exception:
            pass

    # -- main window --------------------------------------------------------
    def test_main_window_lists_records(self) -> None:
        self.assertIn("Records:", self.app.status_records.get())
        self.assertFalse(self.app.get_data().empty)

    def test_every_screen_can_be_opened(self) -> None:
        factories = {
            "add": self.gui.AddObservationWindow,
            "records": self.gui.ViewRecordsWindow,
            "filter": self.gui.SearchFilterWindow,
            "stats": self.gui.StatisticsWindow,
            "locations": self.gui.LocationComparisonWindow,
            "periods": self.gui.PeriodComparisonWindow,
            "poor": self.gui.PoorAirQualityWindow,
            "unusual": self.gui.UnusualValuesWindow,
            "reports": self.gui.ReportsWindow,
        }
        for key, factory in factories.items():
            self.app.open_window(key, factory)
            self.app.update()

        self.assertEqual(len(self.app._windows), len(factories))
        for window in self.app._windows.values():
            self.assertTrue(window.winfo_exists())

        # Opening the same screen twice must reuse the existing window.
        self.app.open_window("stats", self.gui.StatisticsWindow)
        self.assertEqual(len(self.app._windows), len(factories))
        self.app.update()

    def test_screens_are_laid_out_visibly(self) -> None:
        """Every screen must actually be mapped with a usable size."""
        factories = {
            "records": self.gui.ViewRecordsWindow,
            "filter": self.gui.SearchFilterWindow,
            "reports": self.gui.ReportsWindow,
        }
        # Child windows are only mapped while the main window is visible.
        self.app.deiconify()
        self.addCleanup(self.app.withdraw)
        for key, factory in factories.items():
            self.app.open_window(key, factory)
            self.app.update()
            window = self.app._windows[key]
            self.assertTrue(window.winfo_ismapped(), f"{key} window is not mapped")
            self.assertGreater(window.winfo_width(), 300, f"{key} window is too narrow")
            self.assertGreater(window.winfo_height(), 200, f"{key} window is too short")
            self.assertTrue(window.body.winfo_ismapped())
            self.assertGreater(window.body.winfo_width(), 100)

    def test_windows_close_and_unregister(self) -> None:
        self.app.open_window("records", self.gui.ViewRecordsWindow)
        window = self.app._windows["records"]
        window.close()
        self.app.update()
        self.assertNotIn("records", self.app._windows)

    # -- individual screens -------------------------------------------------
    def test_records_window_is_populated(self) -> None:
        window = self.gui.ViewRecordsWindow(self.app)
        self.addCleanup(window.close)
        self.assertEqual(len(window.tree.get_children()), len(self.app.get_data()))
        self.assertIn("Showing all", window.status_var.get())

    def test_filter_window_applies_and_clears(self) -> None:
        window = self.gui.SearchFilterWindow(self.app)
        self.addCleanup(window.close)
        data = self.app.get_data()

        window.apply_filter()
        self.assertEqual(len(window.tree.get_children()), len(data))

        location = str(data.iloc[0][self.gui.dm.LOCATION])
        window.location_box.set(location)
        window.min_aqi.insert(0, "1000")  # impossible AQI range
        window.apply_filter()
        self.assertEqual(len(window.tree.get_children()), 0)

        window.clear_filter()
        self.assertEqual(len(window.tree.get_children()), 0)

    def test_filter_window_rejects_bad_values(self) -> None:
        window = self.gui.SearchFilterWindow(self.app)
        self.addCleanup(window.close)
        window.min_aqi.insert(0, "not a number")
        window.apply_filter()
        self.dialogs["showwarning"].assert_called()
        window.min_aqi.delete(0, "end")
        window.min_aqi.insert(0, "100")
        window.apply_filter()
        self.assertGreater(len(window.tree.get_children()), 0)

    def test_statistics_window_shows_summary(self) -> None:
        window = self.gui.StatisticsWindow(self.app)
        self.addCleanup(window.close)
        content = window.text.get("1.0", "end")
        self.assertIn("AIR QUALITY SUMMARY", content)
        self.assertIn("Standard Deviation", content)

    def test_location_window_shows_comparison(self) -> None:
        window = self.gui.LocationComparisonWindow(self.app)
        self.addCleanup(window.close)
        self.assertGreater(len(window.tree.get_children()), 0)
        self.assertIn("Best air quality", window.text.get("1.0", "end"))

    def test_period_window_compares_default_ranges(self) -> None:
        window = self.gui.PeriodComparisonWindow(self.app)
        self.addCleanup(window.close)
        window.compare()
        self.assertEqual(len(window.tree.get_children()), 6)

        window.a_start.delete(0, "end")
        window.a_start.insert(0, "not-a-date")
        window.compare()
        self.dialogs["showwarning"].assert_called()

    def test_poor_quality_window_sorts_descending(self) -> None:
        window = self.gui.PoorAirQualityWindow(self.app)
        self.addCleanup(window.close)
        rows = window.tree.get_children()
        self.assertTrue(rows)
        first_aqi = float(window.tree.item(rows[0])["values"][3])
        last_aqi = float(window.tree.item(rows[-1])["values"][3])
        self.assertGreaterEqual(first_aqi, last_aqi)

        window.top_n.set("abc")
        window.load_poor_periods()
        self.dialogs["showwarning"].assert_called()

    def test_unusual_values_window_reports_threshold(self) -> None:
        window = self.gui.UnusualValuesWindow(self.app)
        self.addCleanup(window.close)
        window.threshold.set("2.0")
        window.detect()

        window.threshold.set("0")
        window.detect()
        self.dialogs["showwarning"].assert_called()

    def test_reports_window_builds_three_charts(self) -> None:
        window = self.gui.ReportsWindow(self.app)
        self.addCleanup(window.close)
        self.assertEqual(len(window.notebook.tabs()), 3)
        self.assertIn("VISUAL REPORTS", window.summary.get())

        with mock.patch.object(self.gui.viz, "save_all_reports",
                               return_value=[]) as fake_save:
            window.save_reports()
        fake_save.assert_called_once()

    # -- add observation ----------------------------------------------------
    def test_add_window_saves_a_valid_observation(self) -> None:
        window = self.gui.AddObservationWindow(self.app)
        self.addCleanup(window.close)

        values = {
            self.gui.dm.DATE: "2026-10-07",
            self.gui.dm.TIME: "09:15",
            self.gui.dm.LOCATION: "Test Lab",
            self.gui.dm.PM25: "35",
            self.gui.dm.PM10: "60",
            self.gui.dm.CO: "0.5",
            self.gui.dm.NO2: "20",
            self.gui.dm.SO2: "5",
            self.gui.dm.O3: "70",
            self.gui.dm.AQI: "90",
        }
        for field, text in values.items():
            window.entries[field].delete(0, "end")
            window.entries[field].insert(0, text)

        with mock.patch.object(self.gui.dm, "save_observation",
                               return_value="AQ-9999") as fake_save:
            window.save()

        fake_save.assert_called_once()
        record = fake_save.call_args[0][0]
        self.assertEqual(record[self.gui.dm.LOCATION], "Test Lab")
        self.assertEqual(record[self.gui.dm.AQI], 90.0)
        self.dialogs["showinfo"].assert_called()
        # The form is cleared for the next entry after a successful save.
        self.assertEqual(window.entries[self.gui.dm.LOCATION].get(), "")

    def test_add_window_rejects_invalid_observation(self) -> None:
        window = self.gui.AddObservationWindow(self.app)
        self.addCleanup(window.close)
        window.entries[self.gui.dm.PM25].delete(0, "end")
        window.entries[self.gui.dm.PM25].insert(0, "-12")
        window.entries[self.gui.dm.DATE].delete(0, "end")
        window.entries[self.gui.dm.DATE].insert(0, "31-12-2026")

        with mock.patch.object(self.gui.dm, "save_observation") as fake_save:
            window.save()

        fake_save.assert_not_called()
        self.dialogs["showwarning"].assert_called()
        self.assertIn("not saved", window.status_var.get())

    def test_demo_data_loader_uses_the_import_function(self) -> None:
        with mock.patch.object(self.gui.dm, "import_sample_data",
                               return_value=42) as fake_import:
            self.app._load_demo_data()
        fake_import.assert_called_once()
        self.dialogs["showinfo"].assert_called()


if __name__ == "__main__":
    unittest.main()
