"""Grafické rozhraní aplikace Dark-Field Contamination Analyzer (PyQt6).

Hlavní oprava pádu
------------------
Původní verze se v ``start_analysis()`` připojovala na neexistující metodu::

    self.worker.error_signal.connect(self.on_worker_error)   # AttributeError

Metoda ``on_worker_error`` v třídě nikdy nebyla definována. PyQt6 při
neodchycené výjimce ve slotu (což je i obsluha stisku tlačítka) volá
``qFatal`` a **ukončí celý proces** – aplikace tedy zmizela přesně ve chvíli,
kdy uživatel klikl na „SPUSTIT ANALÝZU“, a to bez jediné hlášky.

Nyní jsou ošetřeny všechny tři vrstvy:

1. slot ``on_worker_error`` existuje a zobrazí dialog s podrobnostmi,
2. pracovní vlákno posílá kompletní traceback místo holého textu výjimky,
3. ``install_exception_hook()`` zachytí i jakoukoliv jinou neočekávanou
   výjimku a místo pádu ukáže okno s chybou.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from typing import List, Optional, Tuple

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from analyzer import AnalysisParams, FrameMetrics, SeriesResult, analyze_series
from exporter import export_all, summarize
from frameio import list_image_files, list_measurement_folders
from help_text import HELP_HTML
from viewer import ImageViewerWidget

APP_ORG = "DarkFieldAnalyzer"
APP_NAME = "DarkFieldAnalyzer"


# ---------------------------------------------------------------------------
# Ošetření neočekávaných výjimek
# ---------------------------------------------------------------------------

def install_exception_hook() -> None:
    """Zobrazí neočekávanou výjimku v dialogu místo tichého pádu aplikace."""
    original_hook = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(message)
        app = QApplication.instance()
        if app is not None:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Neočekávaná chyba")
            box.setText(
                "V aplikaci došlo k neočekávané chybě.\n"
                "Analýza byla přerušena, aplikaci ale můžete dál používat."
            )
            box.setDetailedText(message)
            box.exec()
        else:
            original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


# ---------------------------------------------------------------------------
# Pracovní vlákno
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    """Analýza série v samostatném vlákně, aby GUI nezamrzlo."""

    progress_signal = pyqtSignal(int, int, object)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str, str)

    def __init__(self, image_paths: List[str], params: AnalysisParams):
        super().__init__()
        self.image_paths = image_paths
        self.params = params
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = analyze_series(
                self.image_paths,
                self.params,
                progress=lambda done, total, metrics: self.progress_signal.emit(done, total, metrics),
                should_cancel=lambda: self._cancelled,
            )
            self.finished_signal.emit(result)
        except Exception as exc:  # noqa: BLE001 – vlákno nesmí propustit výjimku
            self.error_signal.emit(str(exc), traceback.format_exc())


# ---------------------------------------------------------------------------
# Hlavní okno
# ---------------------------------------------------------------------------

class DarkfieldAnalyzerGUI(QMainWindow):
    """Hlavní okno aplikace."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analýza kontaminace sklíčka v temném poli (Dark-Field Analyzer)")
        self.resize(1440, 900)

        self.settings = QSettings(APP_ORG, APP_NAME)
        self.current_base_dir = self.settings.value("base_dir", "", type=str)
        if not self.current_base_dir or not os.path.isdir(self.current_base_dir):
            fallback = os.path.join(os.path.expanduser("~"), "Downloads")
            self.current_base_dir = fallback if os.path.isdir(fallback) else os.path.expanduser("~")

        self.selected_folder_path: Optional[str] = None
        self.current_image_paths: List[str] = []
        self.current_result: Optional[SeriesResult] = None
        self.worker: Optional[AnalysisWorker] = None
        self._last_progress_update = 0.0

        self.init_ui()
        self.restore_settings()
        self.refresh_folder_list()

    # -- sestavení rozhraní -------------------------------------------------

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_tabs())
        splitter.setSizes([460, 980])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(4, 4, 4, 4)

        # 1. Složka -----------------------------------------------------------
        dir_group = QGroupBox("1. Složka s měřeními")
        dir_layout = QVBoxLayout(dir_group)

        row = QHBoxLayout()
        self.lbl_base_dir = QLabel(self.current_base_dir)
        self.lbl_base_dir.setStyleSheet(
            "font-size: 11px; background: #2B2B2B; color: #FFF; padding: 4px; border-radius: 4px;"
        )
        self.lbl_base_dir.setWordWrap(True)
        row.addWidget(self.lbl_base_dir, 1)
        btn_browse = QPushButton("Procházet…")
        btn_browse.clicked.connect(self.browse_base_dir)
        row.addWidget(btn_browse)
        dir_layout.addLayout(row)

        dir_layout.addWidget(QLabel("Nalezené složky se snímky:"))
        self.folder_table = QTableWidget()
        self.folder_table.setColumnCount(3)
        self.folder_table.setHorizontalHeaderLabels(["Složka", "Snímků", "Stav"])
        header = self.folder_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.folder_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.folder_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.folder_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.folder_table.itemSelectionChanged.connect(self.on_folder_selected)
        dir_layout.addWidget(self.folder_table)

        btn_refresh = QPushButton("Obnovit seznam")
        btn_refresh.clicked.connect(self.refresh_folder_list)
        dir_layout.addWidget(btn_refresh)
        column.addWidget(dir_group)

        # 2. Parametry --------------------------------------------------------
        params_group = QGroupBox("2. Parametry analýzy")
        grid = QVBoxLayout(params_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Rozlišení analýzy:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems([
            "Automaticky (doporučeno)",
            "Plné rozlišení (nejpřesnější, nejpomalejší)",
            "Poloviční – binning 2×2",
            "Čtvrtinové – binning 4×4 (nejrychlejší)",
        ])
        self.combo_res.setToolTip(
            "Automaticky = 4K se zpracuje v polovičním rozlišení, FHD v plném.\n"
            "Binning průměruje sousední pixely, takže zlepšuje poměr signál/šum."
        )
        row.addWidget(self.combo_res, 1)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Bias snímků:"))
        self.spin_bias_frames = QSpinBox()
        self.spin_bias_frames.setRange(1, 50)
        self.spin_bias_frames.setValue(3)
        self.spin_bias_frames.setToolTip("Z kolika prvních snímků se vytvoří referenční pozadí.")
        row.addWidget(self.spin_bias_frames)
        row.addWidget(QLabel("Metoda:"))
        self.combo_bias_method = QComboBox()
        self.combo_bias_method.addItems(["medián (odolný)", "průměr"])
        row.addWidget(self.combo_bias_method, 1)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Režim prahu:"))
        self.radio_sigma = QRadioButton("Sigma (šum)")
        self.radio_sigma.setChecked(True)
        self.radio_absolute = QRadioButton("Absolutní")
        row.addWidget(self.radio_sigma)
        row.addWidget(self.radio_absolute)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sigma:"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.5, 30.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(4.0)
        row.addWidget(self.spin_sigma)
        row.addWidget(QLabel("Absolutní [ADU]:"))
        self.spin_abs = QDoubleSpinBox()
        self.spin_abs.setRange(1.0, 255.0)
        self.spin_abs.setValue(12.0)
        row.addWidget(self.spin_abs)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Práh zamlžení [ADU]:"))
        self.spin_haze = QDoubleSpinBox()
        self.spin_haze.setRange(0.5, 60.0)
        self.spin_haze.setSingleStep(0.5)
        self.spin_haze.setValue(4.0)
        row.addWidget(self.spin_haze)
        row.addWidget(QLabel("Hotspot od [ADU]:"))
        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(100.0, 255.0)
        self.spin_saturation.setValue(250.0)
        row.addWidget(self.spin_saturation)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min. plocha [px]:"))
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 200)
        self.spin_min_area.setValue(3)
        row.addWidget(self.spin_min_area)
        row.addWidget(QLabel("Shluk od [px]:"))
        self.spin_cluster_area = QSpinBox()
        self.spin_cluster_area.setRange(20, 20000)
        self.spin_cluster_area.setSingleStep(20)
        self.spin_cluster_area.setValue(100)
        row.addWidget(self.spin_cluster_area)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Protáhlost vlákna:"))
        self.spin_aspect_ratio = QDoubleSpinBox()
        self.spin_aspect_ratio.setRange(1.5, 20.0)
        self.spin_aspect_ratio.setValue(2.8)
        self.spin_aspect_ratio.setToolTip("Poměr hlavní a vedlejší osy ekvivalentní elipsy.")
        row.addWidget(self.spin_aspect_ratio)
        row.addWidget(QLabel("Min. délka [px]:"))
        self.spin_fiber_len = QSpinBox()
        self.spin_fiber_len.setRange(3, 500)
        self.spin_fiber_len.setValue(12)
        row.addWidget(self.spin_fiber_len)
        grid.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Měřítko [µm/px]:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.0, 1000.0)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setValue(1.0)
        row.addWidget(self.spin_scale)
        row.addWidget(QLabel("Vláken CPU:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(0)
        self.spin_workers.setToolTip("0 = automaticky podle počtu jader procesoru.")
        row.addWidget(self.spin_workers)
        grid.addLayout(row)

        row = QHBoxLayout()
        self.chk_roi = QCheckBox("Analyzovat jen výřez (ROI):")
        self.chk_roi.setToolTip(
            "Souřadnice v pixelech plného rozlišení. Hodí se, když má snímek\n"
            "zajímavou jen část plochy (např. okraj s vinětací se má vynechat)."
        )
        row.addWidget(self.chk_roi)
        self.spin_roi = []
        for label, maximum in (("x", 20000), ("y", 20000), ("š", 20000), ("v", 20000)):
            row.addWidget(QLabel(label))
            box = QSpinBox()
            box.setRange(0, maximum)
            box.setMaximumWidth(70)
            row.addWidget(box)
            self.spin_roi.append(box)
        self.spin_roi[2].setValue(0)
        self.spin_roi[3].setValue(0)
        grid.addLayout(row)

        self.chk_exclude_bias = QCheckBox("Vynechat bias snímky z výsledné řady (doporučeno)")
        self.chk_exclude_bias.setChecked(True)
        self.chk_exclude_bias.setToolTip(
            "Snímky použité pro bias se porovnávají samy se sebou, takže jejich\n"
            "hodnoty nejsou srovnatelné se zbytkem řady."
        )
        grid.addWidget(self.chk_exclude_bias)

        btn_defaults = QPushButton("Obnovit výchozí hodnoty")
        btn_defaults.clicked.connect(self.reset_defaults)
        grid.addWidget(btn_defaults)

        column.addWidget(params_group)

        # 3. Spuštění ---------------------------------------------------------
        action_group = QGroupBox("3. Spuštění a export")
        action_layout = QVBoxLayout(action_group)

        row = QHBoxLayout()
        self.btn_start = QPushButton("▶  SPUSTIT ANALÝZU")
        self.btn_start.setStyleSheet(
            "background-color: #27AE60; color: white; font-weight: bold; font-size: 13px;"
            " padding: 8px; border-radius: 4px;"
        )
        self.btn_start.clicked.connect(self.start_analysis)
        row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹  Zastavit")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background-color: #C0392B; color: white; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        self.btn_stop.clicked.connect(self.stop_analysis)
        row.addWidget(self.btn_stop)
        action_layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        action_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Vyberte složku v seznamu výše.")
        self.lbl_status.setStyleSheet("font-size: 11px; color: #555;")
        self.lbl_status.setWordWrap(True)
        action_layout.addWidget(self.lbl_status)

        self.btn_export = QPushButton("💾  Exportovat výsledky (CSV + grafy + JSON)")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_export.clicked.connect(self.export_results)
        action_layout.addWidget(self.btn_export)

        column.addWidget(action_group)
        column.addStretch()
        return panel

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()

        self.fig_coverage = Figure(figsize=(7, 5), dpi=100)
        self.canvas_coverage = self._add_plot_tab(self.fig_coverage, "📈 Pokrytí a čistota")

        self.fig_particles = Figure(figsize=(7, 5), dpi=100)
        self.canvas_particles = self._add_plot_tab(self.fig_particles, "🔬 Typy kontaminace")

        self.fig_signal = Figure(figsize=(7, 5), dpi=100)
        self.canvas_signal = self._add_plot_tab(self.fig_signal, "💡 Signál a hotspoty")

        self.fig_rates = Figure(figsize=(7, 5), dpi=100)
        self.canvas_rates = self._add_plot_tab(self.fig_rates, "⚡ Rychlost a nehomogenita")

        self.viewer_widget = ImageViewerWidget()
        self.tabs.addTab(self.viewer_widget, "🖼️ Vizuální kontrola")

        self.table_results = QTableWidget()
        self.table_results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table_results, "📋 Datová tabulka")

        self.summary_browser = QTextBrowser()
        self.summary_browser.setHtml("<p style='color:#666;'>Souhrn se zobrazí po dokončení analýzy.</p>")
        self.tabs.addTab(self.summary_browser, "🧾 Souhrn měření")

        help_browser = QTextBrowser()
        help_browser.setHtml(HELP_HTML)
        help_browser.setOpenExternalLinks(True)
        self.tabs.addTab(help_browser, "📖 Průvodce")
        return self.tabs

    def _add_plot_tab(self, figure: Figure, title: str) -> FigureCanvas:
        canvas = FigureCanvas(figure)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas)
        self.tabs.addTab(tab, title)
        return canvas

    # -- nastavení ----------------------------------------------------------

    def restore_settings(self) -> None:
        s = self.settings
        self.combo_res.setCurrentIndex(s.value("res_index", 0, type=int))
        self.spin_bias_frames.setValue(s.value("bias_frames", 3, type=int))
        self.combo_bias_method.setCurrentIndex(s.value("bias_method", 0, type=int))
        self.radio_absolute.setChecked(s.value("absolute_mode", False, type=bool))
        self.radio_sigma.setChecked(not self.radio_absolute.isChecked())
        self.spin_sigma.setValue(s.value("sigma", 4.0, type=float))
        self.spin_abs.setValue(s.value("absolute", 12.0, type=float))
        self.spin_haze.setValue(s.value("haze", 4.0, type=float))
        self.spin_saturation.setValue(s.value("saturation", 250.0, type=float))
        self.spin_min_area.setValue(s.value("min_area", 3, type=int))
        self.spin_cluster_area.setValue(s.value("cluster_area", 100, type=int))
        self.spin_aspect_ratio.setValue(s.value("aspect", 2.8, type=float))
        self.spin_fiber_len.setValue(s.value("fiber_len", 12, type=int))
        self.spin_scale.setValue(s.value("um_per_px", 1.0, type=float))
        self.spin_workers.setValue(s.value("workers", 0, type=int))
        self.chk_exclude_bias.setChecked(s.value("exclude_bias", True, type=bool))
        self.chk_roi.setChecked(s.value("roi_enabled", False, type=bool))
        for index, box in enumerate(self.spin_roi):
            box.setValue(s.value(f"roi_{index}", 0, type=int))

    def save_settings(self) -> None:
        s = self.settings
        s.setValue("base_dir", self.current_base_dir)
        s.setValue("res_index", self.combo_res.currentIndex())
        s.setValue("bias_frames", self.spin_bias_frames.value())
        s.setValue("bias_method", self.combo_bias_method.currentIndex())
        s.setValue("absolute_mode", self.radio_absolute.isChecked())
        s.setValue("sigma", self.spin_sigma.value())
        s.setValue("absolute", self.spin_abs.value())
        s.setValue("haze", self.spin_haze.value())
        s.setValue("saturation", self.spin_saturation.value())
        s.setValue("min_area", self.spin_min_area.value())
        s.setValue("cluster_area", self.spin_cluster_area.value())
        s.setValue("aspect", self.spin_aspect_ratio.value())
        s.setValue("fiber_len", self.spin_fiber_len.value())
        s.setValue("um_per_px", self.spin_scale.value())
        s.setValue("workers", self.spin_workers.value())
        s.setValue("exclude_bias", self.chk_exclude_bias.isChecked())
        s.setValue("roi_enabled", self.chk_roi.isChecked())
        for index, box in enumerate(self.spin_roi):
            s.setValue(f"roi_{index}", box.value())

    def reset_defaults(self) -> None:
        defaults = AnalysisParams()
        self.combo_res.setCurrentIndex(0)
        self.spin_bias_frames.setValue(defaults.bias_frames)
        self.combo_bias_method.setCurrentIndex(0)
        self.radio_sigma.setChecked(True)
        self.spin_sigma.setValue(defaults.sigma)
        self.spin_abs.setValue(defaults.absolute_threshold)
        self.spin_haze.setValue(defaults.haze_threshold)
        self.spin_saturation.setValue(defaults.saturation_adu)
        self.spin_min_area.setValue(defaults.min_area_px)
        self.spin_cluster_area.setValue(defaults.cluster_min_area_px)
        self.spin_aspect_ratio.setValue(defaults.fiber_aspect_ratio)
        self.spin_fiber_len.setValue(defaults.fiber_min_length_px)
        self.spin_scale.setValue(defaults.um_per_px)
        self.spin_workers.setValue(defaults.workers)
        self.chk_roi.setChecked(False)
        for box in self.spin_roi:
            box.setValue(0)

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Vrátí zvolený výřez, nebo ``None`` pro celý snímek."""
        if not self.chk_roi.isChecked():
            return None
        x, y, width, height = (box.value() for box in self.spin_roi)
        if width <= 0 or height <= 0:
            return None
        return (x, y, width, height)

    def get_current_params(self) -> AnalysisParams:
        binning = {0: 0, 1: 1, 2: 2, 3: 4}[self.combo_res.currentIndex()]
        return AnalysisParams(
            bias_frames=self.spin_bias_frames.value(),
            bias_method="median" if self.combo_bias_method.currentIndex() == 0 else "mean",
            threshold_mode="sigma" if self.radio_sigma.isChecked() else "absolute",
            sigma=self.spin_sigma.value(),
            absolute_threshold=self.spin_abs.value(),
            haze_threshold=self.spin_haze.value(),
            min_area_px=self.spin_min_area.value(),
            cluster_min_area_px=self.spin_cluster_area.value(),
            fiber_aspect_ratio=self.spin_aspect_ratio.value(),
            fiber_min_length_px=self.spin_fiber_len.value(),
            saturation_adu=self.spin_saturation.value(),
            um_per_px=self.spin_scale.value(),
            binning=binning,
            roi=self.get_roi(),
            workers=self.spin_workers.value(),
            exclude_bias_from_series=self.chk_exclude_bias.isChecked(),
        )

    # -- složky -------------------------------------------------------------

    def browse_base_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Vyberte složku s měřeními", self.current_base_dir)
        if folder:
            self.current_base_dir = folder
            self.lbl_base_dir.setText(folder)
            self.save_settings()
            self.refresh_folder_list()

    def refresh_folder_list(self) -> None:
        self.folder_table.setRowCount(0)
        self.lbl_base_dir.setText(self.current_base_dir)

        if not os.path.isdir(self.current_base_dir):
            self.lbl_status.setText("Zvolená složka neexistuje.")
            return

        folders = list_measurement_folders(self.current_base_dir)
        for row, (path, count) in enumerate(folders):
            self.folder_table.insertRow(row)
            label = os.path.basename(path) or path
            if os.path.normpath(path) == os.path.normpath(self.current_base_dir):
                label = f"{label}  (tato složka)"
            item_name = QTableWidgetItem(label)
            item_name.setData(Qt.ItemDataRole.UserRole, path)
            item_name.setToolTip(path)
            item_count = QTableWidgetItem(str(count))
            item_count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_status = QTableWidgetItem("Připraveno")
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.folder_table.setItem(row, 0, item_name)
            self.folder_table.setItem(row, 1, item_count)
            self.folder_table.setItem(row, 2, item_status)

        if folders:
            self.folder_table.selectRow(0)
            self.lbl_status.setText(f"Nalezeno {len(folders)} složek se snímky.")
        else:
            self.selected_folder_path = None
            self.lbl_status.setText(
                "V této složce ani v jejích podsložkách nejsou žádné snímky (PNG/TIFF/BMP/JPG)."
            )

    def on_folder_selected(self) -> None:
        row = self.folder_table.currentRow()
        item = self.folder_table.item(row, 0) if row >= 0 else None
        if item:
            self.selected_folder_path = item.data(Qt.ItemDataRole.UserRole)
            self.lbl_status.setText(f"Vybráno: {os.path.basename(self.selected_folder_path)}")

    # -- analýza ------------------------------------------------------------

    def start_analysis(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.selected_folder_path or not os.path.isdir(self.selected_folder_path):
            QMessageBox.warning(self, "Upozornění", "Vyberte prosím platnou složku k analýze.")
            return

        image_paths = list_image_files(self.selected_folder_path)
        if not image_paths:
            QMessageBox.warning(self, "Chyba", "Ve vybrané složce nejsou žádné snímky.")
            return

        self.current_image_paths = image_paths
        params = self.get_current_params()
        self.save_settings()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Počítám referenční bias…")

        self.worker = AnalysisWorker(image_paths, params)
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.start()

    def stop_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("Zastavuji analýzu…")

    def on_worker_progress(self, done: int, total: int, metrics: FrameMetrics) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(done)
        # Text se obnovuje jen občas – u tisíců snímků by přepis každého řádku
        # zbytečně vytěžoval hlavní vlákno.
        if done == total or done % 5 == 0:
            self.lbl_status.setText(
                f"Zpracováno {done}/{total} · {metrics.filename} · "
                f"pokrytí {metrics.total_coverage_pct:.2f} % · čistota {metrics.cleanliness_score:.0f} %"
            )

    def on_worker_error(self, message: str, details: str) -> None:
        """Chyba v pracovním vlákně – dřív tento slot chyběl a aplikace padala."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText(f"Analýza selhala: {message}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Chyba při analýze")
        box.setText(f"Analýzu se nepodařilo dokončit:\n{message}")
        box.setDetailedText(details)
        box.exec()

    def on_worker_finished(self, result: SeriesResult) -> None:
        self.current_result = result
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if not result.metrics:
            self.progress_bar.setValue(0)
            self.lbl_status.setText("Analýza neposkytla žádná data.")
            self._report_problems(result)
            return

        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())

        count = len(result.metrics)
        per_frame_ms = result.elapsed_s / count * 1000.0
        state = "Zastaveno uživatelem" if result.cancelled else "Hotovo"
        self.lbl_status.setText(
            f"{state}: {count} snímků za {result.elapsed_s:.1f} s "
            f"({per_frame_ms:.0f} ms/snímek, {count / max(result.elapsed_s, 1e-6):.1f} sn./s)."
        )

        row = self.folder_table.currentRow()
        if row >= 0 and self.folder_table.item(row, 2):
            item = self.folder_table.item(row, 2)
            item.setText("Analyzováno")
            item.setForeground(QColor("#27AE60"))

        self.viewer_widget.set_dataset(self.current_image_paths, result.bias, result.params)
        self.plot_all_graphs()
        self.populate_results_table()
        self.populate_summary()
        self._report_problems(result)

    def _report_problems(self, result: SeriesResult) -> None:
        if not result.warnings and not result.failed_files:
            return
        lines = list(result.warnings)
        if result.failed_files:
            lines.append("")
            lines.append("Nezpracované soubory:")
            for path, reason in result.failed_files[:20]:
                lines.append(f"  • {os.path.basename(path)} – {reason}")
            if len(result.failed_files) > 20:
                lines.append(f"  … a dalších {len(result.failed_files) - 20}")
        QMessageBox.information(self, "Upozornění k analýze", "\n".join(lines))

    # -- výstupy ------------------------------------------------------------

    def plot_all_graphs(self) -> None:
        if not self.current_result or not self.current_result.metrics:
            return
        res = self.current_result.metrics
        times = [m.time_s for m in res]

        self.fig_coverage.clear()
        ax = self.fig_coverage.add_subplot(111)
        lines = ax.plot(times, [m.total_coverage_pct for m in res], "k-", label="Celkové pokrytí [%]", linewidth=2.2)
        lines += ax.plot(times, [m.haze_coverage_pct for m in res], "b--", label="Zamlžení / opar [%]", alpha=0.85)
        ax.set_xlabel("Čas [s]", fontweight="bold")
        ax.set_ylabel("Pokrytí sklíčka [%]", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        twin = ax.twinx()
        lines += twin.plot(times, [m.cleanliness_score for m in res], color="#27AE60", linestyle="-.",
                           label="Skóre čistoty [0–100 %]", linewidth=2.0)
        twin.set_ylabel("Skóre čistoty [%]", color="#27AE60", fontweight="bold")
        twin.tick_params(axis="y", labelcolor="#27AE60")
        twin.set_ylim(0, 105)
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
        ax.set_title("Vývoj znečištění a skóre čistoty sklíčka", fontsize=12, fontweight="bold")
        self.fig_coverage.tight_layout()
        self.canvas_coverage.draw_idle()

        self.fig_particles.clear()
        ax = self.fig_particles.add_subplot(111)
        ax.plot(times, [m.point_count for m in res], color="#E67E22", label="Mikročástice (prach)", marker="o", markersize=3)
        ax.plot(times, [m.cluster_count for m in res], color="#C0392B", label="Velké shluky", marker="s", markersize=3)
        ax.plot(times, [m.fiber_count for m in res], color="#2980B9", label="Vlákna / škrábance", marker="^", markersize=3)
        ax.set_xlabel("Čas [s]", fontweight="bold")
        ax.set_ylabel("Počet [ks]", fontweight="bold")
        ax.set_title("Vývoj počtu částic podle kategorií", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        self.fig_particles.tight_layout()
        self.canvas_particles.draw_idle()

        self.fig_signal.clear()
        ax = self.fig_signal.add_subplot(111)
        lines = ax.plot(times, [m.mean_signal_adu for m in res], color="#8E44AD", label="Průměrný jas kontaminace [ADU]", linewidth=2)
        lines += ax.plot(times, [m.bg_noise_sigma for m in res], color="#95A5A6", linestyle=":", label="Šum pozadí σ [ADU]")
        ax.set_xlabel("Čas [s]", fontweight="bold")
        ax.set_ylabel("Signál [ADU]", color="#8E44AD", fontweight="bold")
        ax.tick_params(axis="y", labelcolor="#8E44AD")
        ax.grid(True, linestyle="--", alpha=0.5)
        twin = ax.twinx()
        lines += twin.plot(times, [m.saturated_pixels_count for m in res], color="#D35400", linestyle="--",
                           label="Hotspoty [px]", linewidth=1.8)
        twin.set_ylabel("Hotspoty [px]", color="#D35400", fontweight="bold")
        twin.tick_params(axis="y", labelcolor="#D35400")
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
        ax.set_title("Intenzita signálu, šum pozadí a hotspoty", fontsize=12, fontweight="bold")
        self.fig_signal.tight_layout()
        self.canvas_signal.draw_idle()

        self.fig_rates.clear()
        ax = self.fig_rates.add_subplot(111)
        ax.axhline(0, color="gray", linewidth=1, alpha=0.7)
        lines = ax.plot(times, [m.rate_coverage_pct_per_s for m in res], color="#C0392B",
                        label="d(Pokrytí)/dt [%/s]", linewidth=2)
        ax.set_xlabel("Čas [s]", fontweight="bold")
        ax.set_ylabel("Rychlost změny pokrytí [%/s]", color="#C0392B", fontweight="bold")
        ax.tick_params(axis="y", labelcolor="#C0392B")
        ax.grid(True, linestyle="--", alpha=0.5)
        twin = ax.twinx()
        lines += twin.plot(times, [m.spatial_heterogeneity_pct for m in res], color="#2C3E50", linestyle=":",
                           label="Nehomogenita rozložení [%]", linewidth=1.8)
        twin.set_ylabel("Nehomogenita [%]", color="#2C3E50", fontweight="bold")
        twin.tick_params(axis="y", labelcolor="#2C3E50")
        twin.set_ylim(0, 105)
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
        ax.set_title("Dynamika změn v čase a prostorová nehomogenita", fontsize=12, fontweight="bold")
        self.fig_rates.tight_layout()
        self.canvas_rates.draw_idle()

    def populate_results_table(self) -> None:
        if not self.current_result:
            return
        res = self.current_result.metrics
        headers = [
            "#", "Čas [s]", "Hodiny", "Pokrytí [%]", "Zamlžení [%]", "Mikročástic", "Shluků",
            "Vláken", "Hotspotů", "Nehomogenita [%]", "Čistota [%]", "Jas [ADU]", "Práh [ADU]",
            "Rychlost [%/s]", "Fáze",
        ]
        table = self.table_results
        table.setSortingEnabled(False)
        table.setUpdatesEnabled(False)
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(res))

        for row, m in enumerate(res):
            values = [
                str(m.index + 1),
                f"{m.time_s:.2f}",
                m.timestamp.strftime("%H:%M:%S"),
                f"{m.total_coverage_pct:.3f}",
                f"{m.haze_coverage_pct:.3f}",
                str(m.point_count),
                str(m.cluster_count),
                str(m.fiber_count),
                str(m.saturated_pixels_count),
                f"{m.spatial_heterogeneity_pct:.1f}",
                f"{m.cleanliness_score:.1f}",
                f"{m.mean_signal_adu:.1f}",
                f"{m.applied_threshold:.1f}",
                f"{m.rate_coverage_pct_per_s:.3f}",
                m.phase,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 10:
                    score = m.cleanliness_score
                    item.setForeground(QColor("#27AE60" if score >= 90 else "#D4AC0D" if score >= 70 else "#C0392B"))
                elif col == 14:
                    if "nárůst" in value:
                        item.setForeground(QColor("#C0392B"))
                    elif "odpařování" in value or "ústup" in value:
                        item.setForeground(QColor("#2980B9"))
                    else:
                        item.setForeground(QColor("#27AE60"))
                if m.is_bias_frame:
                    item.setBackground(QColor("#F2F3F4"))
                table.setItem(row, col, item)

        table.setUpdatesEnabled(True)
        table.resizeColumnsToContents()

    def populate_summary(self) -> None:
        if not self.current_result:
            return
        data = summarize(self.current_result)
        if not data:
            return
        phases = "".join(f"<li>{name}: <b>{count}</b> snímků</li>" for name, count in data["faze"].items())
        warnings = "".join(f"<li>{item}</li>" for item in data["varovani"]) or "<li>žádná</li>"
        html = f"""
        <h2 style='color:#1B4F72;'>Souhrn měření</h2>
        <table cellpadding='6' style='border-collapse:collapse;'>
          <tr><td><b>Snímků</b></td><td>{data['pocet_snimku']}</td></tr>
          <tr><td><b>Délka měření</b></td><td>{data['delka_mereni_s']:.2f} s</td></tr>
          <tr><td><b>Průměrné pokrytí</b></td><td>{data['pokryti_prumer_pct']:.3f} %</td></tr>
          <tr><td><b>Maximum pokrytí</b></td>
              <td>{data['pokryti_max_pct']:.3f} % v čase {data['pokryti_max_cas_s']:.2f} s
              ({data['pokryti_max_snimek']})</td></tr>
          <tr><td><b>Průměrná čistota</b></td><td>{data['cistota_prumer_pct']:.1f} / 100</td></tr>
          <tr><td><b>Nejhorší snímek</b></td>
              <td>čistota {data['cistota_min_pct']:.1f} ({data['cistota_min_snimek']})</td></tr>
          <tr><td><b>Částic na začátku → na konci</b></td>
              <td>{data['castic_na_zacatku']} → {data['castic_na_konci']}</td></tr>
          <tr><td><b>Nejrychlejší nárůst</b></td><td>{data['max_rychlost_pokryti_pct_s']:.3f} %/s</td></tr>
          <tr><td><b>Nejrychlejší ústup</b></td><td>{data['min_rychlost_pokryti_pct_s']:.3f} %/s</td></tr>
          <tr><td><b>Doba výpočtu</b></td><td>{data['cas_analyzy_s']:.2f} s</td></tr>
        </table>
        <h3>Zastoupení fází</h3><ul>{phases}</ul>
        <h3>Upozornění</h3><ul>{warnings}</ul>
        """
        self.summary_browser.setHtml(html)

    def export_results(self) -> None:
        if not self.current_result or not self.current_result.metrics:
            return

        folder_name = os.path.basename(self.selected_folder_path or "mereni")
        default_csv = os.path.join(
            self.selected_folder_path or self.current_base_dir,
            f"analyza_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        csv_path, _ = QFileDialog.getSaveFileName(self, "Uložit výsledky", default_csv, "CSV soubory (*.csv)")
        if not csv_path:
            return

        try:
            outputs = export_all(self.current_result, csv_path, folder_name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Chyba při exportu",
                f"Soubory se nepodařilo uložit:\n{exc}\n\n"
                "Zkontrolujte, zda cílový soubor není otevřený v Excelu.",
            )
            return

        QMessageBox.information(
            self, "Export dokončen",
            "Uloženo:\n"
            f"• Tabulka: {os.path.basename(outputs['csv'])}\n"
            f"• Grafy: {os.path.basename(outputs['png'])}\n"
            f"• Souhrn: {os.path.basename(outputs['json'])}",
        )

    # -- zavření okna -------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 (Qt API)
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self.save_settings()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    install_exception_hook()

    window = DarkfieldAnalyzerGUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
