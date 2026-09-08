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

from PyQt6.QtCore import QSettings, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
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

from analyzer import AnalysisParams, FrameMetrics, SeriesResult, analyze_series, resolve_reference
from exporter import export_all, plot_composition, summarize
from frameio import list_image_files, list_measurement_folders
from help_text import HELP_HTML
from viewer import ImageViewerWidget

APP_ORG = "DarkFieldAnalyzer"
APP_NAME = "DarkFieldAnalyzer"

#: Výchozí kořenová složka s měřeními (počítač, pro který je aplikace určená).
DEFAULT_BASE_DIR = r"C:\Users\Programovani\Downloads\BMS fotky"

#: Volby režimu referenčního pozadí – pořadí odpovídá položkám v rozbalovacím seznamu.
REFERENCE_MODES: Tuple[str, ...] = ("auto", "reference", "serie")

#: Volby převodu barevného snímku na intenzitu (klíč pro ``AnalysisParams.mono_mode``).
MONO_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("luma", "Vážený jas – Rec.601 (doporučeno)"),
    ("prumer", "Průměr kanálů"),
    ("maximum", "Maximum kanálů"),
    ("r", "Jen červený kanál"),
    ("g", "Jen zelený kanál"),
    ("b", "Jen modrý kanál"),
)


def default_base_dir() -> str:
    """Vrátí výchozí složku s měřeními, případně nejbližší existující náhradu."""
    candidates = [
        DEFAULT_BASE_DIR,
        os.path.join(os.path.expanduser("~"), "Downloads", "BMS fotky"),
        os.path.join(os.path.expanduser("~"), "Downloads"),
        os.path.expanduser("~"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return os.path.expanduser("~")


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
        self.setMinimumSize(940, 560)

        self.settings = QSettings(APP_ORG, APP_NAME)
        self.current_base_dir = self.settings.value("base_dir", "", type=str)
        if not self.current_base_dir or not os.path.isdir(self.current_base_dir):
            self.current_base_dir = default_base_dir()

        self.selected_folder_path: Optional[str] = None
        self.reference_dir_override: Optional[str] = None
        self.current_image_paths: List[str] = []
        self.current_result: Optional[SeriesResult] = None
        self.worker: Optional[AnalysisWorker] = None
        self._last_progress_update = 0.0

        self.init_ui()
        self.restore_settings()
        self.restore_geometry()
        self.refresh_folder_list()

    # -- velikost okna ------------------------------------------------------

    def restore_geometry(self) -> None:
        """Nastaví velikost okna tak, aby se vždy vešlo na obrazovku.

        Pevná velikost 1440×900 byla problém na noteboocích s Full HD a
        škálováním Windows 150 %: plocha má pak jen 1280×720 logických bodů,
        takže okno bylo větší než displej a spodní tlačítka byla mimo obraz.
        Uloženou geometrii z minula proto vždy ještě ořízneme na aktuální
        obrazovku – po přepojení na jiný monitor se okno nemůže „ztratit“.
        """
        available = self._available_geometry()
        max_w, max_h = available.width(), available.height()

        saved = self.settings.value("window_geometry", None)
        restored = bool(saved) and self.restoreGeometry(saved)

        width = min(self.width() if restored else 1380, int(max_w * 0.96))
        height = min(self.height() if restored else 880, int(max_h * 0.96))
        width = max(width, min(self.minimumWidth(), max_w))
        height = max(height, min(self.minimumHeight(), max_h))
        self.resize(width, height)

        if not restored or not available.contains(self.geometry()):
            frame = self.frameGeometry()
            frame.moveCenter(available.center())
            self.move(max(available.left(), frame.left()), max(available.top(), frame.top()))

        # Levý panel dostane rozumný podíl šířky i na úzkém displeji.
        left = max(300, min(430, int(width * 0.32)))
        self.splitter.setSizes([left, max(320, width - left)])

    def _available_geometry(self):
        """Plocha obrazovky bez hlavního panelu (v logických bodech)."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            from PyQt6.QtCore import QRect

            return QRect(0, 0, 1280, 720)
        return screen.availableGeometry()

    # -- sestavení rozhraní -------------------------------------------------

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Levý panel je vysoký; na nízkém displeji se musí dát rolovat,
        # jinak by roztáhl celé okno mimo obrazovku.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(self._build_left_panel())
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(520)

        self.splitter.addWidget(left_scroll)
        self.splitter.addWidget(self._build_tabs())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(6)

        # 1. Složka -----------------------------------------------------------
        dir_group = QGroupBox("1. Složka s měřeními")
        dir_layout = QVBoxLayout(dir_group)

        row = QHBoxLayout()
        self.lbl_base_dir = QLabel(self.current_base_dir)
        self.lbl_base_dir.setStyleSheet(
            "font-size: 11px; background: #2B2B2B; color: #FFF; padding: 4px; border-radius: 4px;"
        )
        self.lbl_base_dir.setWordWrap(False)
        self.lbl_base_dir.setMinimumWidth(120)
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
        self.folder_table.setMinimumHeight(96)
        self.folder_table.setMaximumHeight(200)
        self.folder_table.verticalHeader().setDefaultSectionSize(22)
        self.folder_table.itemSelectionChanged.connect(self.on_folder_selected)
        dir_layout.addWidget(self.folder_table)

        btn_refresh = QPushButton("Obnovit seznam")
        btn_refresh.clicked.connect(self.refresh_folder_list)
        dir_layout.addWidget(btn_refresh)
        column.addWidget(dir_group)

        # 1b. Referenční pozadí ----------------------------------------------
        ref_group = QGroupBox("1b. Referenční pozadí (bias)")
        ref_layout = QVBoxLayout(ref_group)
        ref_layout.setSpacing(4)

        self.combo_reference = QComboBox()
        self.combo_reference.addItems([
            "Automaticky – reference ze složky, jinak série",
            "Vždy ze složky s referencemi",
            "Vždy z prvních snímků série",
        ])
        self.combo_reference.setToolTip(
            "Reference (.npz z BMS Cam Control) se hledá ve složce „reference“\n"
            "vedle měření. Vybere se ta, která vznikla naposledy PŘED měřením."
        )
        self.combo_reference.currentIndexChanged.connect(self.update_reference_preview)
        ref_layout.addWidget(self.combo_reference)

        row = QHBoxLayout()
        self.lbl_reference_dir = QLabel("(hledá se automaticky)")
        self.lbl_reference_dir.setStyleSheet("font-size: 10px; color: #555;")
        self.lbl_reference_dir.setMinimumWidth(100)
        row.addWidget(self.lbl_reference_dir, 1)
        btn_ref_dir = QPushButton("Složka…")
        btn_ref_dir.setToolTip("Ručně určit složku s .npz referencemi.")
        btn_ref_dir.clicked.connect(self.browse_reference_dir)
        row.addWidget(btn_ref_dir)
        btn_ref_auto = QPushButton("Auto")
        btn_ref_auto.setToolTip("Zrušit ruční volbu a hledat složku „reference“ automaticky.")
        btn_ref_auto.clicked.connect(self.clear_reference_dir)
        row.addWidget(btn_ref_auto)
        ref_layout.addLayout(row)

        self.lbl_reference_pick = QLabel("Vyberte složku s měřením.")
        self.lbl_reference_pick.setWordWrap(True)
        self.lbl_reference_pick.setStyleSheet(
            "font-size: 10px; color: #1D3B5C; background: #EAF2FA; padding: 4px; border-radius: 4px;"
        )
        ref_layout.addWidget(self.lbl_reference_pick)

        self.chk_match_level = QCheckBox("Srovnat úroveň reference se snímky")
        self.chk_match_level.setChecked(True)
        self.chk_match_level.setToolTip(
            "Reference vznikla dřív, takže se od snímků může lišit konstantním\n"
            "posunem jasu (teplota senzoru, jas zdroje). Bez srovnání se takový\n"
            "posun projeví jako plošné zamlžení přes celý snímek.\n"
            "Posun se odhaduje jednou pro celou sérii z nejtmavšího snímku."
        )
        ref_layout.addWidget(self.chk_match_level)
        column.addWidget(ref_group)

        # 2. Parametry --------------------------------------------------------
        # Jeden sloupec (QFormLayout): dvojice ovládacích prvků vedle sebe se
        # na úzkém panelu ořezávaly, protože se nevešly do dostupné šířky.
        params_group = QGroupBox("2. Parametry analýzy")
        form = QFormLayout(params_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)

        self.combo_res = QComboBox()
        self.combo_res.addItems([
            "Automaticky (doporučeno)",
            "Plné rozlišení",
            "Poloviční – binning 2×2",
            "Čtvrtinové – binning 4×4",
        ])
        self.combo_res.setToolTip(
            "Automaticky = 4K se zpracuje v polovičním rozlišení, FHD v plném.\n"
            "Binning průměruje sousední pixely, takže zlepšuje poměr signál/šum."
        )
        form.addRow("Rozlišení analýzy:", self.combo_res)

        self.spin_bias_frames = QSpinBox()
        self.spin_bias_frames.setRange(1, 50)
        self.spin_bias_frames.setValue(3)
        self.spin_bias_frames.setToolTip("Z kolika prvních snímků se vytvoří referenční pozadí.")
        form.addRow("Bias snímků:", self.spin_bias_frames)

        self.combo_bias_method = QComboBox()
        self.combo_bias_method.addItems(["medián (odolný)", "průměr"])
        form.addRow("Metoda biasu:", self.combo_bias_method)

        self.combo_mono = QComboBox()
        for _key, label in MONO_CHOICES:
            self.combo_mono.addItem(label)
        self.combo_mono.setToolTip(
            "Týká se jen barevných snímků – mono snímky se použijí tak, jak jsou.\n"
            "Vážený jas odpovídá tomu, jak černobílý obraz počítá sama kamera,\n"
            "takže sedí na referenci. Průměr a maximum kanálů jsou citlivější\n"
            "na modré rozptylové halo částic, ale zvyšují i šum pozadí."
        )
        form.addRow("Barevný snímek jako:", self.combo_mono)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.radio_sigma = QRadioButton("Sigma (šum)")
        self.radio_sigma.setChecked(True)
        self.radio_absolute = QRadioButton("Absolutní")
        mode_layout.addWidget(self.radio_sigma)
        mode_layout.addWidget(self.radio_absolute)
        mode_layout.addStretch()
        form.addRow("Režim prahu:", mode_row)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.5, 30.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(4.0)
        form.addRow("Sigma [× σ]:", self.spin_sigma)

        self.spin_abs = QDoubleSpinBox()
        self.spin_abs.setRange(1.0, 255.0)
        self.spin_abs.setValue(12.0)
        form.addRow("Absolutní práh [ADU]:", self.spin_abs)

        self.spin_haze = QDoubleSpinBox()
        self.spin_haze.setRange(0.5, 60.0)
        self.spin_haze.setSingleStep(0.5)
        self.spin_haze.setValue(4.0)
        form.addRow("Práh zamlžení [ADU]:", self.spin_haze)

        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(100.0, 255.0)
        self.spin_saturation.setValue(250.0)
        form.addRow("Hotspot od [ADU]:", self.spin_saturation)

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 200)
        self.spin_min_area.setValue(3)
        form.addRow("Min. plocha částice [px]:", self.spin_min_area)

        self.spin_cluster_area = QSpinBox()
        self.spin_cluster_area.setRange(20, 20000)
        self.spin_cluster_area.setSingleStep(20)
        self.spin_cluster_area.setValue(100)
        form.addRow("Velký shluk od [px]:", self.spin_cluster_area)

        self.spin_aspect_ratio = QDoubleSpinBox()
        self.spin_aspect_ratio.setRange(1.5, 20.0)
        self.spin_aspect_ratio.setValue(2.8)
        self.spin_aspect_ratio.setToolTip("Poměr hlavní a vedlejší osy ekvivalentní elipsy.")
        form.addRow("Protáhlost vlákna:", self.spin_aspect_ratio)

        self.spin_fiber_len = QSpinBox()
        self.spin_fiber_len.setRange(3, 500)
        self.spin_fiber_len.setValue(12)
        form.addRow("Min. délka vlákna [px]:", self.spin_fiber_len)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.0, 1000.0)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setValue(1.0)
        form.addRow("Měřítko [µm/px]:", self.spin_scale)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(0)
        self.spin_workers.setToolTip("0 = automaticky podle počtu jader procesoru.")
        form.addRow("Vláken CPU:", self.spin_workers)

        # Výřez (ROI): zaškrtávátko a pod ním dvě dvojice souřadnic
        self.chk_roi = QCheckBox("Analyzovat jen výřez (ROI)")
        self.chk_roi.setToolTip(
            "Souřadnice v pixelech plného rozlišení. Hodí se, když má snímek\n"
            "zajímavou jen část plochy (např. okraj s vinětací se má vynechat)."
        )
        form.addRow(self.chk_roi)

        self.spin_roi = []
        roi_row = QWidget()
        roi_layout = QHBoxLayout(roi_row)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(4)
        for label in ("x", "y", "š", "v"):
            roi_layout.addWidget(QLabel(label))
            box = QSpinBox()
            box.setRange(0, 20000)
            box.setMinimumWidth(52)
            box.setEnabled(False)
            roi_layout.addWidget(box, 1)
            self.spin_roi.append(box)
        self.chk_roi.toggled.connect(lambda on: [box.setEnabled(on) for box in self.spin_roi])
        form.addRow("Výřez [px]:", roi_row)

        self.chk_exclude_bias = QCheckBox("Vynechat bias snímky z výsledků")
        self.chk_exclude_bias.setChecked(True)
        self.chk_exclude_bias.setToolTip(
            "Snímky použité pro bias se porovnávají samy se sebou, takže jejich\n"
            "hodnoty nejsou srovnatelné se zbytkem řady."
        )
        form.addRow(self.chk_exclude_bias)

        btn_defaults = QPushButton("Obnovit výchozí hodnoty")
        btn_defaults.clicked.connect(self.reset_defaults)
        form.addRow(btn_defaults)

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
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.setDocumentMode(True)
        self.tabs.setMinimumWidth(320)

        self.fig_coverage = Figure(figsize=(6.0, 4.0), dpi=96)
        self.canvas_coverage = self._add_plot_tab(self.fig_coverage, "📈 Pokrytí")

        self.fig_particles = Figure(figsize=(6.0, 4.0), dpi=96)
        self.canvas_particles = self._add_plot_tab(self.fig_particles, "🔬 Typy")

        self.fig_signal = Figure(figsize=(6.0, 4.0), dpi=96)
        self.canvas_signal = self._add_plot_tab(self.fig_signal, "💡 Signál")

        self.fig_rates = Figure(figsize=(6.0, 4.0), dpi=96)
        self.canvas_rates = self._add_plot_tab(self.fig_rates, "⚡ Rychlost")

        self.fig_composition = Figure(figsize=(6.0, 6.2), dpi=96)
        self.canvas_composition = self._add_plot_tab(self.fig_composition, "🧩 Složení")

        self.viewer_widget = ImageViewerWidget()
        self.tabs.addTab(self.viewer_widget, "🖼️ Snímky")

        self.table_results = QTableWidget()
        self.table_results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table_results, "📋 Tabulka")

        self.summary_browser = QTextBrowser()
        self.summary_browser.setHtml("<p style='color:#666;'>Souhrn se zobrazí po dokončení analýzy.</p>")
        self.tabs.addTab(self.summary_browser, "🧾 Souhrn")

        help_browser = QTextBrowser()
        help_browser.setHtml(HELP_HTML)
        help_browser.setOpenExternalLinks(True)
        self.tabs.addTab(help_browser, "📖 Průvodce")
        return self.tabs

    def _add_plot_tab(self, figure: Figure, title: str) -> FigureCanvas:
        canvas = FigureCanvas(figure)
        # Bez malé minimální velikosti si plátno vynutí šířku podle figsize
        # a okno pak nejde zmenšit pod rozlišení displeje.
        canvas.setMinimumSize(240, 180)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setIconSize(QSize(18, 18))
        layout.addWidget(toolbar)
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
        self.combo_reference.setCurrentIndex(s.value("reference_mode", 0, type=int))
        self.chk_match_level.setChecked(s.value("match_reference_level", True, type=bool))
        self.combo_mono.setCurrentIndex(s.value("mono_mode", 0, type=int))
        stored_dir = s.value("reference_dir", "", type=str)
        self.reference_dir_override = stored_dir if stored_dir and os.path.isdir(stored_dir) else None
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
        s.setValue("reference_mode", self.combo_reference.currentIndex())
        s.setValue("match_reference_level", self.chk_match_level.isChecked())
        s.setValue("mono_mode", self.combo_mono.currentIndex())
        s.setValue("reference_dir", self.reference_dir_override or "")
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
        self.combo_reference.setCurrentIndex(0)
        self.chk_match_level.setChecked(defaults.match_reference_level)
        self.combo_mono.setCurrentIndex(0)
        self.reference_dir_override = None
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
            reference_mode=REFERENCE_MODES[self.combo_reference.currentIndex()],
            reference_dir=self.reference_dir_override,
            match_reference_level=self.chk_match_level.isChecked(),
            mono_mode=MONO_CHOICES[self.combo_mono.currentIndex()][0],
        )

    # -- složky -------------------------------------------------------------

    def browse_base_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Vyberte složku s měřeními", self.current_base_dir)
        if folder:
            self.current_base_dir = folder
            self._update_base_dir_label()
            self.save_settings()
            self.refresh_folder_list()

    def _update_base_dir_label(self) -> None:
        """Zobrazí zkrácenou cestu (celá je v tooltipu), aby netlačila na šířku."""
        parts = os.path.normpath(self.current_base_dir).split(os.sep)
        short = os.sep.join(parts[-3:]) if len(parts) > 3 else self.current_base_dir
        self.lbl_base_dir.setText(("…" + os.sep + short) if short != self.current_base_dir else short)
        self.lbl_base_dir.setToolTip(self.current_base_dir)

    def refresh_folder_list(self) -> None:
        self.folder_table.setRowCount(0)
        self._update_base_dir_label()

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
        self.update_reference_preview()

    # -- reference ----------------------------------------------------------

    def browse_reference_dir(self) -> None:
        start = self.reference_dir_override or self.current_base_dir
        folder = QFileDialog.getExistingDirectory(self, "Vyberte složku s referencemi (.npz)", start)
        if folder:
            self.reference_dir_override = folder
            self.save_settings()
            self.update_reference_preview()

    def clear_reference_dir(self) -> None:
        """Zruší ruční volbu složky – reference se zase bude hledat automaticky."""
        self.reference_dir_override = None
        self.save_settings()
        self.update_reference_preview()

    def update_reference_preview(self) -> None:
        """Ukáže, která reference by se pro vybranou složku právě použila.

        Je to jen náhled – závazný výběr provádí analýza znovu, protože se
        složka s referencemi mezitím mohla změnit.
        """
        if not hasattr(self, "lbl_reference_pick"):
            return

        override = self.reference_dir_override
        self.lbl_reference_dir.setText(
            os.path.basename(os.path.normpath(override)) if override else "(hledá se automaticky)"
        )
        self.lbl_reference_dir.setToolTip(override or "Složka „reference“ se hledá vedle měření.")

        mode = REFERENCE_MODES[self.combo_reference.currentIndex()]
        enabled = mode != "serie"
        self.chk_match_level.setEnabled(enabled)
        if not enabled:
            self.lbl_reference_pick.setText(
                f"Pozadí se spočítá z prvních {self.spin_bias_frames.value()} snímků série."
            )
            return

        folder = self.selected_folder_path
        if not folder or not os.path.isdir(folder):
            self.lbl_reference_pick.setText("Vyberte složku s měřením.")
            return

        paths = list_image_files(folder)
        if not paths:
            self.lbl_reference_pick.setText("Ve složce nejsou žádné snímky.")
            return

        try:
            params = self.get_current_params()
            choice, directory = resolve_reference(paths, params, folder=folder)
        except Exception as exc:  # noqa: BLE001 – náhled nesmí shodit okno
            self.lbl_reference_pick.setText(f"Referenci nelze načíst: {exc}")
            return

        if directory:
            self.lbl_reference_dir.setText(os.path.basename(os.path.normpath(directory)))
            self.lbl_reference_dir.setToolTip(directory)

        if choice is None:
            fallback = "Použije se bias ze série." if mode == "auto" else "Analýza skončí chybou."
            self.lbl_reference_pick.setText(f"Složka s referencemi nenalezena. {fallback}")
        elif not choice.ok:
            self.lbl_reference_pick.setText(choice.reason)
        else:
            self.lbl_reference_pick.setText(choice.reason)

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

        # Složení kontaminace: kolik procent plochy zabírá který typ
        self.fig_composition.clear()
        grid = self.fig_composition.add_gridspec(3, 1, height_ratios=[1, 1, 0.3], hspace=0.62)
        plot_composition(
            self.fig_composition.add_subplot(grid[0, 0]),
            self.fig_composition.add_subplot(grid[2, 0]),
            res,
            ax_particles=self.fig_composition.add_subplot(grid[1, 0]),
        )
        self.fig_composition.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.08)
        self.canvas_composition.draw_idle()

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
        notes = "".join(f"<li>{item}</li>" for item in data.get("poznamky", []))

        bias = self.current_result.bias
        if bias is None:
            background = "neznámé"
        elif bias.is_external:
            offset = bias.level_offset_adu
            background = (
                f"reference <b>{os.path.basename(bias.reference_path or '')}</b> "
                f"({bias.reference_label}), srovnání úrovně {offset:+.2f} ADU"
            )
        else:
            background = f"prvních {bias.frames_used} snímků série ({self.current_result.params.bias_method})"

        html = f"""
        <h2 style='color:#1B4F72;'>Souhrn měření</h2>
        <table cellpadding='6' style='border-collapse:collapse;'>
          <tr><td><b>Referenční pozadí</b></td><td>{background}</td></tr>
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
        {"<h3>Poznámky k průběhu</h3><ul>" + notes + "</ul>" if notes else ""}
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
        self.settings.setValue("window_geometry", self.saveGeometry())
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
