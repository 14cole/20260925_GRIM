"""Plotting-tab analysis settings and the parameter slider bar."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QToolButton,
)

from GRIM_Backend.plotting.modes.sector_stats_mode import STATISTICS


def _percent_spin(value: float, tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setDecimals(1)
    spin.setSingleStep(5.0)
    spin.setSuffix(" %")
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setToolTip(tooltip)
    return spin


class PlotAnalysisControls(QObject):
    """PBP band, CDF, sector-statistics, and Δ Ref settings.

    ``changed`` carries which feature changed ("pbp", "cdf", "sector",
    "delta") so only the matching plot type re-renders.
    """

    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_pbp_band = QComboBox()
        self.combo_pbp_band.addItem("Min–Max", "minmax")
        self.combo_pbp_band.addItem("Percentiles", "percentile")
        self.combo_pbp_band.setToolTip(
            "Min–Max bounds every selected series; Percentiles bounds the chosen "
            "lower and upper percentiles at each point, ignoring outliers."
        )
        self.spin_pbp_low = _percent_spin(10.0, "Lower percentile of the PBP band.")
        self.spin_pbp_high = _percent_spin(90.0, "Upper percentile of the PBP band.")

        self.combo_cdf = QComboBox()
        self.combo_cdf.addItem("Cumulative: samples at or below level", "cdf")
        self.combo_cdf.addItem("Exceedance: samples at or above level", "exceedance")

        self.edit_sectors = QLineEdit("30")
        self.edit_sectors.setPlaceholderText("30   or   -180:30:180   or   -45:45, 45:135, 170:-170")
        self.edit_sectors.setToolTip(
            "A width (30) tiles the selected azimuths; start:step:stop tiles a range; "
            "start:stop pairs separated by commas list sectors, wrapping through "
            "±180 when stop is below start. Values use the azimuth list's units."
        )
        self.combo_sector_stat = QComboBox()
        for key, label in STATISTICS:
            self.combo_sector_stat.addItem(label, key)
        self.combo_sector_stat.setToolTip(
            "Statistic of linear power inside each sector, shown on the dB scale."
        )
        self.spin_sector_percentile = _percent_spin(90.0, "Percentile used by the Percentile statistic.")

        self.spin_delta_tolerance = QDoubleSpinBox()
        self.spin_delta_tolerance.setRange(0.0, 100.0)
        self.spin_delta_tolerance.setDecimals(2)
        self.spin_delta_tolerance.setSingleStep(0.5)
        self.spin_delta_tolerance.setPrefix("± ")
        self.spin_delta_tolerance.setSuffix(" dB")
        self.spin_delta_tolerance.setSpecialValueText("Off")
        self.spin_delta_tolerance.setKeyboardTracking(False)
        self.spin_delta_tolerance.setToolTip(
            "Shade ± this band around zero on Δ Ref plots and report the share of "
            "samples inside it. Off at zero."
        )

        self.spin_range_subband = QDoubleSpinBox()
        self.spin_range_subband.setRange(5.0, 100.0)
        self.spin_range_subband.setDecimals(0)
        self.spin_range_subband.setSingleStep(5.0)
        self.spin_range_subband.setSuffix(" % of band")
        self.spin_range_subband.setValue(25.0)
        self.spin_range_subband.setKeyboardTracking(False)
        self.spin_range_subband.setToolTip(
            "Width of each sliding sub-band. Narrower sub-bands show more "
            "frequency detail but coarser down-range resolution."
        )
        self.combo_range_window = QComboBox()
        self.combo_range_window.addItems(
            ["Hanning", "Hamming", "Blackman", "Blackman-Harris", "Kaiser β=15", "Rectangular"]
        )
        self.combo_range_window.setToolTip("Window applied to each sub-band before the inverse FFT.")
        self.combo_range_unit = QComboBox()
        self.combo_range_unit.addItems(["m", "cm", "mm", "in", "ft"])

        self.combo_pbp_band.currentIndexChanged.connect(self._pbp_band_changed)
        self.spin_range_subband.valueChanged.connect(lambda: self.changed.emit("range"))
        self.combo_range_window.currentIndexChanged.connect(lambda: self.changed.emit("range"))
        self.combo_range_unit.currentIndexChanged.connect(lambda: self.changed.emit("range"))
        self.spin_pbp_low.valueChanged.connect(self._pbp_low_changed)
        self.spin_pbp_high.valueChanged.connect(self._pbp_high_changed)
        self.combo_cdf.currentIndexChanged.connect(lambda: self.changed.emit("cdf"))
        self.edit_sectors.editingFinished.connect(lambda: self.changed.emit("sector"))
        self.combo_sector_stat.currentIndexChanged.connect(self._sector_stat_changed)
        self.spin_sector_percentile.valueChanged.connect(lambda: self.changed.emit("sector"))
        self.spin_delta_tolerance.valueChanged.connect(lambda: self.changed.emit("delta"))
        self._pbp_band_changed(emit=False)
        self._sector_stat_changed(emit=False)

    def add_rows(self, grid, row: int) -> int:
        """Add labelled rows to the Plot Settings grid; returns the next row."""
        rows = (
            (("PbP Band", self.combo_pbp_band), ("Lower", self.spin_pbp_low),
             ("Upper", self.spin_pbp_high)),
            (("CDF", self.combo_cdf),),
            (("Sectors", self.edit_sectors),),
            (("Sector Statistic", self.combo_sector_stat),
             ("Percentile", self.spin_sector_percentile)),
            (("Δ Ref Tolerance", self.spin_delta_tolerance),),
            (("Range–Freq Sub-band", self.spin_range_subband),
             ("Window", self.combo_range_window), ("Range Unit", self.combo_range_unit)),
        )
        for cells in rows:
            for column, (label, widget) in enumerate(cells):
                grid.addWidget(QLabel(label), row, 2 * column)
                span = 5 if len(cells) == 1 else 1
                grid.addWidget(widget, row, 2 * column + 1, 1, span)
            row += 1
        return row

    # --- values ------------------------------------------------------------

    def pbp_percentiles(self) -> tuple[float, float] | None:
        if self.combo_pbp_band.currentData() != "percentile":
            return None
        return float(self.spin_pbp_low.value()), float(self.spin_pbp_high.value())

    def set_pbp_band(self, mode: str) -> None:
        index = self.combo_pbp_band.findData(mode)
        if index >= 0:
            self.combo_pbp_band.setCurrentIndex(index)

    def cdf_exceedance(self) -> bool:
        return self.combo_cdf.currentData() == "exceedance"

    def sector_text(self) -> str:
        return self.edit_sectors.text()

    def sector_statistic(self) -> str:
        return str(self.combo_sector_stat.currentData())

    def sector_percentile(self) -> float:
        return float(self.spin_sector_percentile.value())

    def delta_tolerance(self) -> float:
        return float(self.spin_delta_tolerance.value())

    def range_subband_percent(self) -> float:
        return float(self.spin_range_subband.value())

    def range_window(self) -> str:
        return self.combo_range_window.currentText()

    def range_unit(self) -> str:
        return self.combo_range_unit.currentText()

    # --- keep percentile pairs valid -----------------------------------------

    def _pbp_band_changed(self, *_args, emit: bool = True) -> None:
        enabled = self.combo_pbp_band.currentData() == "percentile"
        self.spin_pbp_low.setEnabled(enabled)
        self.spin_pbp_high.setEnabled(enabled)
        if emit:
            self.changed.emit("pbp")

    def _pbp_low_changed(self, value: float) -> None:
        if value >= self.spin_pbp_high.value():
            blocked = self.spin_pbp_high.blockSignals(True)
            self.spin_pbp_high.setValue(min(100.0, value + 1.0))
            self.spin_pbp_high.blockSignals(blocked)
            if value >= self.spin_pbp_high.value():
                self.spin_pbp_low.setValue(self.spin_pbp_high.value() - 1.0)
                return
        self.changed.emit("pbp")

    def _pbp_high_changed(self, value: float) -> None:
        if value <= self.spin_pbp_low.value():
            blocked = self.spin_pbp_low.blockSignals(True)
            self.spin_pbp_low.setValue(max(0.0, value - 1.0))
            self.spin_pbp_low.blockSignals(blocked)
            if value <= self.spin_pbp_low.value():
                self.spin_pbp_high.setValue(self.spin_pbp_low.value() + 1.0)
                return
        self.changed.emit("pbp")

    def _sector_stat_changed(self, *_args, emit: bool = True) -> None:
        self.spin_sector_percentile.setEnabled(
            self.combo_sector_stat.currentData() == "percentile"
        )
        if emit:
            self.changed.emit("sector")


class PlotSliderBar(QFrame):
    """Axis choice, slider, step buttons, and Play for scrubbing one axis."""

    moved = Signal(int)
    axis_changed = Signal(str)
    shown = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("plotSliderBar")
        self.combo_axis = QComboBox()
        for label, axis in (("Frequency", "frequency"), ("Elevation", "elevation"),
                            ("Azimuth", "azimuth")):
            self.combo_axis.addItem(label, axis)
        self.combo_axis.setToolTip("Parameter list the slider steps through.")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setPageStep(1)
        self.btn_prev = QToolButton(text="◀")
        self.btn_prev.setToolTip("Previous value")
        self.btn_next = QToolButton(text="▶")
        self.btn_next.setToolTip("Next value")
        self.value_label = QLabel("--")
        self.value_label.setMinimumWidth(90)
        self.btn_play = QToolButton(text="Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Step through every value, wrapping at the end.")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Slider"))
        layout.addWidget(self.combo_axis)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.value_label)
        layout.addWidget(self.btn_play)

        self._labels: list[str] = []
        # Coalesce drags and key repeats into one render per settled value.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(lambda: self.moved.emit(self.slider.value()))
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(350)
        self._play_timer.timeout.connect(lambda: self.step(1, wrap=True))
        self.slider.valueChanged.connect(self._value_changed)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_play.toggled.connect(self._play_toggled)
        self.combo_axis.currentIndexChanged.connect(
            lambda: self.axis_changed.emit(self.axis())
        )
        self.hide()

    def axis(self) -> str:
        return str(self.combo_axis.currentData())

    def set_positions(self, labels: list[str], current: int) -> None:
        """Replace the slider positions without emitting a move."""
        self._labels = list(labels)
        blocked = self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, len(self._labels) - 1))
        self.slider.setValue(max(0, min(int(current), len(self._labels) - 1)))
        self.slider.blockSignals(blocked)
        self.slider.setEnabled(len(self._labels) > 1)
        self._show_label(self.slider.value())

    def step(self, delta: int, *, wrap: bool = False) -> None:
        count = self.slider.maximum() + 1
        if count <= 1:
            return
        value = self.slider.value() + int(delta)
        value = value % count if wrap else max(0, min(value, count - 1))
        self.slider.setValue(value)

    def stop_play(self) -> None:
        self.btn_play.setChecked(False)

    def _value_changed(self, value: int) -> None:
        self._show_label(value)
        self._debounce.start()

    def _show_label(self, value: int) -> None:
        self.value_label.setText(
            self._labels[value] if 0 <= value < len(self._labels) else "--"
        )

    def _play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("Pause" if checked else "Play")
        if checked:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Lists can change while the bar is hidden (another tab, another row).
        self.shown.emit()

    def hideEvent(self, event) -> None:
        self.stop_play()
        super().hideEvent(event)
