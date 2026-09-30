"""CDF, sector statistics, PbP bands, Δ Ref, markers, slider, and time gate."""

from __future__ import annotations

import os
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

import GRIM_Backend.ui.app as grim_cut_gui
from GRIM_Backend.ui.dataset_actions import DATASET_ID_ROLE, DATASET_PATH_ROLE
from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import down_range_profile, gate_geometry, time_gate
from GRIM_Backend.plotting.dataset_style import pbp_band_key
from GRIM_Backend.plotting.modes import common, sector_stats_mode
from test_gui_shell import (
    _FakeFeatureWorkflow, _FakeFreddyIntegration, _FakeGhostIntegration,
    _MemorySettings, _RecordingWindow,
)
from test_plot_renderer_correctness import _grid

POWER_SHAPE = np.asarray([1.0, 2.0, 5.0, 3.0, 2.0])


class _WindowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for name, replacement in (
            ("GhostIntegrationWidget", _FakeGhostIntegration),
            ("FreddyIntegrationWidget", _FakeFreddyIntegration),
        ):
            patch = mock.patch.object(grim_cut_gui, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(grim_cut_gui, "load_ghost_module", return_value=_FakeFeatureWorkflow())
        patch.start()
        self.addCleanup(patch.stop)
        self.window = _RecordingWindow(settings=_MemorySettings())
        for button in (self.window.btn_auto_plot, self.window.btn_zoom_box, self.window.btn_pan):
            button.setChecked(False)
        for index in range(3):
            dataset = _grid()
            freq_scale = np.linspace(1.0, 2.0, dataset.frequencies.size)[None, None, :, None]
            dataset.rcs_power[:] = POWER_SHAPE[:, None, None, None] * (index + 1) ** 2 * freq_scale
            self.window._add_dataset_row(dataset, f"Run {index + 1}", "", file_name="")
        self.datasets = [self.window.table.item(row, 0).data(Qt.UserRole) for row in range(3)]
        self.keys = [self.window._dataset_plot_key(dataset) for dataset in self.datasets]
        self.select_rows(0, 1, 2)
        self.window.resize(1300, 850)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.plot_slider.stop_play()
        self.window.deleteLater()
        self.app.processEvents()

    def select_rows(self, *rows, freqs=(0,)):
        window = self.window
        window.table.clearSelection()
        for row in rows:
            window.table.selectionModel().select(
                window.table.model().index(row, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows,
            )
        window.table.setCurrentCell(rows[0], 0, QItemSelectionModel.NoUpdate)
        window._on_dataset_selection_changed()
        window.list_az.selectAll()
        window.list_elev.selectAll()
        window.list_freq.clearSelection()
        for row in freqs:
            window.list_freq.item(row).setSelected(True)
        window.list_pol.clearSelection()
        window.list_pol.item(0).setSelected(True)

    def lines(self, key):
        return [line for ax in self.window.plot_figure.axes for line in ax.lines
                if getattr(line, "_grim_dataset_key", None) == key and len(line.get_xdata())]

    def plot(self, method):
        getattr(self.window, method)()
        self.window.plot_canvas.draw()
        self.app.processEvents()

    def legend_labels(self):
        legend = self.window.plot_ax.get_legend()
        return [text.get_text() for text in legend.get_texts()] if legend else []

    def dbsm(self, linear):
        return 10.0 * np.log10(linear)


class CdfTests(_WindowCase):
    def test_cdf_ranks_pooled_samples_and_exceedance_mirrors_it(self):
        window = self.window
        self.select_rows(0, 1, freqs=(0, 1))
        self.plot("_plot_cdf")
        self.assertIn("CDF plot updated", window.status.currentMessage())
        (line,) = self.lines(self.keys[0])
        expected = np.sort(self.dbsm(np.r_[POWER_SHAPE, POWER_SHAPE * np.linspace(1, 2, 8)[1]]))
        np.testing.assert_allclose(line.get_xdata(), expected)
        np.testing.assert_allclose(line.get_ydata(), 100.0 * np.arange(1, 11) / 10)
        self.assertEqual(line.get_drawstyle(), "steps-post")
        self.assertIn("10 samples", line.get_label())
        self.assertIn("at or below", window.plot_ax.get_ylabel())

        window.analysis_controls.combo_cdf.setCurrentIndex(1)
        (line,) = self.lines(self.keys[0])
        np.testing.assert_allclose(line.get_ydata(), 100.0 * (1 - np.arange(10) / 10))
        self.assertIn("at or above", window.plot_ax.get_ylabel())
        self.assertEqual(window.plot_ax.get_ylim(), (0.0, 100.0))

    def test_cdf_rejects_phase_and_delta_and_keeps_hold_overlays(self):
        window = self.window
        window.btn_phase.setChecked(True)
        self.plot("_plot_cdf")
        self.assertIn("Turn off Phase", window.status.currentMessage())
        window.btn_phase.setChecked(False)
        window.btn_delta_ref.setChecked(True)
        self.plot("_plot_cdf")
        self.assertIn("Δ Ref works on", window.status.currentMessage())
        window.btn_delta_ref.setChecked(False)
        self.select_rows(0)
        self.plot("_plot_cdf")
        window.btn_hold.setChecked(True)
        self.select_rows(1)
        self.plot("_plot_cdf")
        self.assertEqual(len(self.lines(self.keys[0])), 1)
        self.assertEqual(len(self.lines(self.keys[1])), 1)
        self.plot("_plot_azimuth_rect")
        self.assertIn("Hold blocked", window.status.currentMessage())


class SectorStatisticsTests(_WindowCase):
    def test_sector_levels_use_linear_power_and_copy_table(self):
        window = self.window
        controls = window.analysis_controls
        controls.edit_sectors.setText("-2:0, 0:2")
        controls.edit_sectors.editingFinished.emit()
        self.select_rows(0)
        self.plot("_plot_sector_stats")
        self.assertIn("2 sectors, mean", window.status.currentMessage())
        (line,) = self.lines(self.keys[0])
        first, second = POWER_SHAPE[:3].mean(), POWER_SHAPE[2:].mean()
        np.testing.assert_allclose(line.get_xdata(), [-2, 0, np.nan, 0, 2, np.nan])
        np.testing.assert_allclose(
            line.get_ydata(), self.dbsm([first, first, np.nan, second, second, np.nan])
        )
        self.assertEqual(window.plot_ax.get_ylabel(), "RCS mean (dBsm)")

        controls.combo_sector_stat.setCurrentIndex(controls.combo_sector_stat.findData("percentile"))
        controls.spin_sector_percentile.setValue(50.0)
        (line,) = self.lines(self.keys[0])
        np.testing.assert_allclose(line.get_ydata()[0], self.dbsm(np.median(POWER_SHAPE[:3])))
        table = window.plot_figure._grim_sector_table
        text = sector_stats_mode.table_text(table)
        self.assertTrue(text.startswith("Dataset\tPol\tFrequency\tElevation\tSector\tSamples"))
        self.assertIn("Run 1\tHH\t9\t0\t-2 to 0\t3", text)

    def test_tiled_sectors_hold_over_azimuth_cut_and_bad_text_blocks(self):
        window = self.window
        self.select_rows(0)
        self.plot("_plot_azimuth_rect")
        window.btn_hold.setChecked(True)
        window.analysis_controls.edit_sectors.setText("2")
        self.plot("_plot_sector_stats")
        self.assertNotIn("blocked", window.status.currentMessage().lower())
        self.assertEqual(len(self.lines(self.keys[0])), 2)
        window.btn_hold.setChecked(False)
        window.analysis_controls.edit_sectors.setText("0:0")
        self.plot("_plot_sector_stats")
        self.assertIn("Sector Stats blocked: sector '0:0' is empty", window.status.currentMessage())

    def test_parse_sectors_tiles_wraps_and_counts_each_sample_once(self):
        azimuths = np.arange(-180.0, 181.0)
        tiles = common.parse_sectors("30", azimuths)
        self.assertEqual(len(tiles), 12)
        self.assertEqual(sum(int(s.contains(azimuths).sum()) for s in tiles), azimuths.size)
        wrap = common.parse_sectors("170:-170", azimuths)[0]
        self.assertEqual(wrap.label(), "170 to -170")
        self.assertEqual(int(wrap.contains(azimuths).sum()), 22)
        self.assertEqual(wrap.display_pieces(-180.0, 180.0), [(170.0, 180.0), (-180.0, -170.0)])
        self.assertEqual(
            [s.label() for s in common.parse_sectors("0:90:360", np.arange(0.0, 360.0))],
            ["0 to 90", "90 to 180", "180 to 270", "270 to 360"],
        )
        for text in ("", "x", "5:5", "0:-1:5", "0:0.001:10"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                common.parse_sectors(text, azimuths)


class PbpBandTests(_WindowCase):
    def band(self, group=""):
        artists = self.window._plot_item_artists(pbp_band_key(group))
        edges = [a for a in artists if hasattr(a, "get_ydata") and len(a.get_ydata())]
        return artists, edges

    def test_percentile_band_uses_chosen_percentiles(self):
        window = self.window
        controls = window.analysis_controls
        window.btn_pbp.setChecked(True)
        self.plot("_plot_azimuth_rect")
        _artists, edges = self.band()
        np.testing.assert_allclose(edges[0].get_ydata(), self.dbsm(POWER_SHAPE))
        np.testing.assert_allclose(edges[1].get_ydata(), self.dbsm(POWER_SHAPE * 9))

        controls.set_pbp_band("percentile")
        controls.spin_pbp_low.setValue(25.0)
        controls.spin_pbp_high.setValue(75.0)
        _artists, edges = self.band()
        levels = self.dbsm(POWER_SHAPE[None, :] * np.array([1.0, 4.0, 9.0])[:, None])
        np.testing.assert_allclose(edges[0].get_ydata(), np.percentile(levels, 25, axis=0))
        np.testing.assert_allclose(edges[1].get_ydata(), np.percentile(levels, 75, axis=0))
        self.assertTrue(self.legend_labels()[0].startswith("PBP P25–P75 Pol HH"))
        controls.spin_pbp_high.setValue(20.0)
        self.assertLess(controls.spin_pbp_low.value(), controls.spin_pbp_high.value())

    def test_groups_draw_one_coloured_band_each_and_remove_separately(self):
        window = self.window
        window.table.item(0, 3).setText("Baseline")
        window.table.item(1, 3).setText("Baseline")
        window.table.item(2, 3).setText("Treated")
        window.btn_pbp.setChecked(True)
        self.plot("_plot_azimuth_rect")
        baseline, baseline_edges = self.band("Baseline")
        treated, treated_edges = self.band("Treated")
        self.assertTrue(baseline and treated)
        np.testing.assert_allclose(baseline_edges[1].get_ydata(), self.dbsm(POWER_SHAPE * 4))
        np.testing.assert_allclose(treated_edges[0].get_ydata(), self.dbsm(POWER_SHAPE * 9))
        self.assertEqual(
            self.legend_labels(),
            ["PBP [Baseline] Pol HH, Freq 9 GHz, Elevation 0 deg",
             "PBP [Treated] Pol HH, Freq 9 GHz, Elevation 0 deg"],
        )
        self.assertNotEqual(baseline_edges[0].get_color(), treated_edges[0].get_color())
        self.assertTrue(window._remove_plot_dataset(pbp_band_key("Baseline")))
        self.assertEqual(self.band("Baseline")[0], [])
        self.assertTrue(self.band("Treated")[0])
        self.assertEqual(len(self.legend_labels()), 1)


class DeltaReferenceTests(_WindowCase):
    def test_each_dataset_minus_active_with_tolerance_share(self):
        window = self.window
        window.analysis_controls.spin_delta_tolerance.setValue(7.0)
        window.btn_delta_ref.setChecked(True)
        self.plot("_plot_azimuth_rect")
        self.assertEqual(self.lines(self.keys[0]), [])
        (second,) = self.lines(self.keys[1])
        (third,) = self.lines(self.keys[2])
        np.testing.assert_allclose(second.get_ydata(), 10 * np.log10(4.0))
        np.testing.assert_allclose(third.get_ydata(), 10 * np.log10(9.0))
        self.assertTrue(second.get_label().startswith("Run 2 − Run 1 | Pol HH"))
        self.assertEqual(window.plot_ax.get_ylabel(), "Level difference from Run 1 (dB)")
        # Run 2 sits 6.0 dB above Run 1 (inside ±7 dB); Run 3 sits 9.5 dB above.
        self.assertIn("50.0% of compared samples are within ±7 dB of Run 1",
                      window.status.currentMessage())
        guides = [a for a in [*window.plot_ax.lines, *window.plot_ax.patches]
                  if getattr(a, "_grim_delta_guide", False)]
        self.assertEqual(len(guides), 2)

        for method in ("_plot_frequency", "_plot_elevation_sweep"):
            with self.subTest(method=method):
                if method == "_plot_elevation_sweep":
                    self.select_rows(0, 1, 2, freqs=(0, 1))
                self.plot(method)
                (second,) = self.lines(self.keys[1])[:1]
                np.testing.assert_allclose(second.get_ydata(), 10 * np.log10(4.0))

    def marked_rows(self):
        return [row for row in range(self.window.table.rowCount())
                if self.window.table.item(row, 0).data(Qt.DecorationRole) is not None]

    def test_picked_reference_is_marked_used_unselected_and_cleared(self):
        window = self.window
        window._set_delta_reference_row(2)
        self.assertEqual(self.marked_rows(), [2])
        self.assertIn("Δ reference set to Run 3", window.status.currentMessage())
        window.table.item(2, 0).setText("Run 3 renamed")
        window.btn_delta_ref.setChecked(True)
        self.select_rows(0, 1)
        self.plot("_plot_azimuth_rect")
        self.assertEqual(self.lines(self.keys[2]), [])
        (first,) = self.lines(self.keys[0])
        (second,) = self.lines(self.keys[1])
        np.testing.assert_allclose(first.get_ydata(), 10 * np.log10(1.0 / 9.0))
        np.testing.assert_allclose(second.get_ydata(), 10 * np.log10(4.0 / 9.0))
        self.assertTrue(first.get_label().startswith("Run 1 − Run 3 renamed | "))
        self.assertEqual(window.plot_ax.get_ylabel(), "Level difference from Run 3 renamed (dB)")

        # Picking another row re-renders against it and moves the marker.
        window._set_delta_reference_row(1)
        self.assertEqual(self.marked_rows(), [1])
        (first,) = self.lines(self.keys[0])
        np.testing.assert_allclose(first.get_ydata(), 10 * np.log10(1.0 / 4.0))
        self.assertEqual(self.lines(self.keys[2]), [])

        window._set_delta_reference_row(None)
        self.assertEqual(self.marked_rows(), [])
        self.assertIn("subtracts the active row again", window.status.currentMessage())
        (second,) = self.lines(self.keys[1])
        np.testing.assert_allclose(second.get_ydata(), 10 * np.log10(4.0))

    def test_deleted_reference_falls_back_and_undo_restores_marker(self):
        window = self.window
        window._set_delta_reference_row(2)
        window.table.clearSelection()
        window.table.selectRow(2)
        with mock.patch(
            "GRIM_Backend.ui.dataset_actions.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._delete_selected_datasets()
        self.assertIsNone(window._explicit_delta_reference())
        window._set_delta_reference_row(0)
        window._undo_last_deleted_datasets()
        self.app.processEvents()
        self.assertEqual(self.marked_rows(), [0])
        window._set_delta_reference_row(None)
        window._delta_reference_id = window.table.item(2, 0).data(DATASET_ID_ROLE)
        window._refresh_delta_reference_marker()
        self.assertEqual(self.marked_rows(), [2])

    def test_blocks_polar_linear_missing_reference_and_mixed_hold(self):
        window = self.window
        window.btn_delta_ref.setChecked(True)
        self.plot("_plot_azimuth_polar")
        self.assertIn("Δ Ref works on", window.status.currentMessage())
        window.combo_plot_scale.setCurrentIndex(window.combo_plot_scale.findData("linear"))
        self.plot("_plot_azimuth_rect")
        self.assertIn("compares levels in dB", window.status.currentMessage())
        window.combo_plot_scale.setCurrentIndex(window.combo_plot_scale.findData("dbsm"))
        window.table.setCurrentCell(0, 0, QItemSelectionModel.NoUpdate)
        window.table.selectionModel().select(
            window.table.model().index(0, 0),
            QItemSelectionModel.Deselect | QItemSelectionModel.Rows,
        )
        window.active_dataset = self.datasets[0]
        self.plot("_plot_azimuth_rect")
        self.assertIn("Select it with the datasets to compare", window.status.currentMessage())

        window.btn_delta_ref.setChecked(False)
        self.select_rows(0, 1)
        self.plot("_plot_azimuth_rect")
        window.btn_hold.setChecked(True)
        window.btn_delta_ref.setChecked(True)
        self.plot("_plot_azimuth_rect")
        self.assertIn("Hold blocked", window.status.currentMessage())
        window._record_python_plot("azimuth_rect", emit=False)
        self.assertEqual(window.last_python_plot_spec[0], "unsupported")


class MarkerTests(_WindowCase):
    def press(self, x, y, kind="button_press_event", button=MouseButton.LEFT):
        event = MouseEvent(kind, self.window.plot_canvas, x, y, button=button)
        handler = {
            "button_press_event": self.window._on_plot_mouse_press,
            "motion_notify_event": self.window._on_plot_mouse_move,
            "button_release_event": self.window._on_plot_mouse_release,
        }[kind]
        handler(event)

    def point_pixels(self, line, index):
        return line.get_transform().transform(
            [[line.get_xdata()[index], line.get_ydata()[index]]]
        )[0]

    def test_markers_snap_drag_step_follow_replots_and_clear(self):
        window = self.window
        self.select_rows(0)
        self.plot("_plot_azimuth_rect")
        (line,) = self.lines(self.keys[0])
        window.btn_zoom_box.setChecked(True)
        window.btn_markers.setChecked(True)
        self.assertFalse(window.btn_zoom_box.isChecked())
        x, y = self.point_pixels(line, 2)
        self.press(x + 6, y - 6)
        self.press(x + 6, y - 6, "button_release_event")
        (marker,) = window._plot_markers
        self.assertEqual(marker["index"], 2)
        text = marker["text"].get_text()
        self.assertIn("M1  Run 1", text)
        self.assertIn("x 0 deg", text)
        self.assertIn(f"y {self.dbsm(5.0):.6g} dBsm", text)
        self.assertEqual(getattr(window, "_highlighted_plot_datasets", set()), set())

        x3, y3 = self.point_pixels(line, 3)
        self.press(x, y)
        self.press(x3, y3, "motion_notify_event")
        self.press(x3, y3, "button_release_event")
        self.assertEqual(marker["index"], 3)
        window._on_plot_key_press(mock.Mock(canvas=window.plot_canvas, key="left"))
        window._on_plot_key_press(mock.Mock(canvas=window.plot_canvas, key="left"))
        self.assertEqual(marker["index"], 1)

        x4, y4 = self.point_pixels(line, 4)
        self.press(x4, y4)
        self.press(x4, y4, "button_release_event")
        second = window._plot_markers[1]
        self.assertIn("Δ M1: +3, ", second["text"].get_text())

        far_x, far_y = window.plot_ax.transAxes.transform((0.02, 0.98))
        self.press(far_x, far_y)
        self.assertIn("No curve near the click", window.status.currentMessage())
        self.assertEqual(len(window._plot_markers), 2)

        window.list_freq.clearSelection()
        window.list_freq.item(3).setSelected(True)
        self.plot("_plot_azimuth_rect")
        self.assertEqual([m["index"] for m in window._plot_markers], [1, 4])
        self.assertIs(window._plot_markers[0]["line"], self.lines(self.keys[0])[0])
        self.assertIn(window._plot_markers[0]["artist"], window.plot_ax.lines)

        window._remove_plot_marker(window._plot_markers[0])
        self.assertEqual([m["number"] for m in window._plot_markers], [2])
        window._clear_plot()
        self.assertEqual(window._plot_markers, [])


class SliderTests(_WindowCase):
    def move(self, row):
        slider = self.window.plot_slider
        slider.slider.setValue(row)
        slider._debounce.stop()
        self.window._on_plot_slider_moved(row)
        self.app.processEvents()

    def test_slider_scrubs_frequency_and_replaces_its_curves_under_hold(self):
        window = self.window
        self.select_rows(0)
        self.plot("_plot_azimuth_rect")
        window.btn_slider.setChecked(True)
        slider = window.plot_slider
        self.assertTrue(slider.isVisible())
        self.assertEqual(slider.slider.maximum(), 7)
        self.move(5)
        self.assertEqual([window.list_freq.row(i) for i in window.list_freq.selectedItems()], [5])
        (line,) = self.lines(self.keys[0])
        self.assertIn(f"Freq {self.datasets[0].frequencies[5]:.12g} GHz", line.get_label())
        np.testing.assert_allclose(line.get_ydata(), self.dbsm(POWER_SHAPE * np.linspace(1, 2, 8)[5]))

        window.btn_hold.setChecked(True)
        self.move(6)
        self.move(7)
        labels = [l.get_label() for l in self.lines(self.keys[0])]
        self.assertEqual(len(labels), 2)  # the cut plotted before Hold plus the scrubbed one
        self.assertTrue(any("Freq 10 GHz" in label for label in labels))

        slider.combo_axis.setCurrentIndex(slider.combo_axis.findData("azimuth"))
        self.move(1)
        self.assertIn("This plot sweeps azimuth", window.status.currentMessage())

    def test_play_steps_and_wraps(self):
        window = self.window
        window.btn_slider.setChecked(True)
        slider = window.plot_slider
        slider.slider.setValue(7)
        slider.step(1, wrap=True)
        self.assertEqual(slider.slider.value(), 0)
        slider.btn_play.setChecked(True)
        self.assertEqual(slider.btn_play.text(), "Pause")
        window.btn_slider.setChecked(False)
        self.assertFalse(slider.btn_play.isChecked())


class TimeGateTests(unittest.TestCase):
    def two_scatterers(self, *, conjugate=False):
        frequencies = np.linspace(8.0, 12.0, 201)
        hz = frequencies * 1e9
        field = (np.exp(-4j * np.pi * hz * 0.3 / C0)
                 + 0.5 * np.exp(-4j * np.pi * hz * -1.0 / C0))
        field = np.broadcast_to(field, (2, 1, frequencies.size)).copy()[..., None]
        units = {"frequency": "GHz"}
        if conjugate:
            field = np.conj(field)
            units["time_convention"] = "exp(-j*omega*t)"
        return RcsGrid([0.0, 1.0], [0.0], frequencies, ["HH"], rcs=field, units=units), hz

    def test_keep_and_remove_isolate_scatterers_in_either_time_convention(self):
        middle = slice(40, 160)
        for conjugate in (False, True):
            with self.subTest(conjugate=conjugate):
                grid, hz = self.two_scatterers(conjugate=conjugate)
                kept = time_gate(grid, start_m=0.0, stop_m=0.6).rcs_slice((0, 0, slice(None), 0))
                want = np.exp(-4j * np.pi * hz * 0.3 / C0)
                want = np.conj(want) if conjugate else want
                np.testing.assert_allclose(kept[middle], want[middle], atol=5e-3)
                self.assertLess(np.max(np.abs(kept - want)), 0.1)
        grid, hz = self.two_scatterers()
        removed = time_gate(grid, start_m=0.0, stop_m=0.6, mode="remove")
        np.testing.assert_allclose(
            removed.rcs_slice((0, 0, slice(None), 0))[middle],
            0.5 * np.exp(-4j * np.pi * hz[middle] * -1.0 / C0), atol=0.05,
        )
        self.assertIn('"mode": "remove"', removed.extra["time_gate_json"])
        ranges, profile = down_range_profile(grid, elevation_index=0, polarization_index=0)
        self.assertAlmostEqual(ranges[np.argmax(profile)], 0.3, delta=0.02)

    def test_rejects_bad_gates_and_grids(self):
        grid, _hz = self.two_scatterers()
        half = gate_geometry(grid)["unambiguous_m"] / 2
        for kwargs, message in (
            ({"start_m": 1.0, "stop_m": 0.0}, "beyond gate start"),
            ({"start_m": -half - 1, "stop_m": 0.0}, "unambiguous down range"),
            ({"start_m": 0.0, "stop_m": 0.01}, "narrower than"),
            ({"start_m": 0.0, "stop_m": 1.0, "mode": "notch"}, "mode must be"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                time_gate(grid, **kwargs)
        uneven = RcsGrid([0.0], [0.0], np.r_[np.linspace(8, 9, 9), 9.5], ["HH"],
                         rcs=np.ones((1, 1, 10, 1), complex), units={"frequency": "GHz"})
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            time_gate(uneven, start_m=-1.0, stop_m=1.0)


class TimeGateGuiTests(_WindowCase):
    def test_button_creates_gated_rows_and_records_script(self):
        window = self.window
        self.assertTrue(window.btn_time_gate.isEnabled())
        # Eight 142.9 MHz steps: 0.13 m resolution, 1.05 m unambiguous range.
        params = {"start_m": -0.3, "stop_m": 0.3, "taper": 0.2, "mode": "keep",
                  "compensate": True}
        # A saved source makes the operation replayable in the Python script.
        window.table.item(0, 1).setData(DATASET_PATH_ROLE, "C:/data/run1.grim")
        with mock.patch("GRIM_Backend.ui.dataset_actions.TimeGateDialog") as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = params
            self.select_rows(0)
            window._time_gate_selected()
            deadline = time.monotonic() + 10
            while window._background_job_active() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
        self.assertIn("Time gate created 1 dataset(s)", window.status.currentMessage())
        self.assertEqual(window.table.rowCount(), 4)
        self.assertEqual(window.table.item(3, 0).text(), "Run 1 [Gate Keep -0.3 to 0.3 m]")
        self.assertIn("Time gate (keep -0.3 to 0.3 m, taper 20%)", window.table.item(3, 2).text())
        script = window.python_recorder.script
        self.assertIn("time_gate(", script)
        self.assertIn("start_m=-0.3", script)


if __name__ == "__main__":
    unittest.main()
