"""Testy exportu a headless test grafického rozhraní.

GUI testy se přeskočí, pokud v prostředí není použitelná knihovna Qt.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import AnalysisParams, analyze_series  # noqa: E402
from exporter import CSV_COLUMNS, export_all, export_to_csv, summarize  # noqa: E402
from frameio import list_image_files  # noqa: E402
from tests.test_analyzer import make_frame, write_series  # noqa: E402


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    folder = tmp_path_factory.mktemp("export")
    frames = []
    for i in range(10):
        frames.append(make_frame(
            dots=[(30 + 8 * i, 40), (120, 80), (200, 160)],
            lines=[(60, 60, 140, 140)],
            haze=float(max(0, 10 - abs(i - 5) * 3)),
            noise=1.5,
            seed=i,
        ))
    paths = write_series(str(folder), frames)
    params = AnalysisParams(bias_frames=2, binning=1, um_per_px=0.5)
    return analyze_series(paths, params), str(folder)


def test_csv_has_all_columns_and_rows(series, tmp_path):
    result, _folder = series
    path = str(tmp_path / "vysledek.csv")
    export_to_csv(result.metrics, path, result.params, "test")

    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))

    header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Index")
    assert rows[header_index] == [name for name, _attr, _spec in CSV_COLUMNS]
    data_rows = [row for row in rows[header_index + 1:] if row]
    assert len(data_rows) == len(result.metrics)
    assert all(len(row) == len(CSV_COLUMNS) for row in data_rows)


def test_csv_uses_czech_decimal_comma_and_bom(series, tmp_path):
    result, _folder = series
    path = str(tmp_path / "carka.csv")
    export_to_csv(result.metrics, path, result.params, "test", decimal_comma=True)

    with open(path, "rb") as handle:
        assert handle.read(3) == b"\xef\xbb\xbf"          # BOM kvůli Excelu

    text = open(path, encoding="utf-8-sig").read()
    assert ";" in text
    body = text.split("Index;")[1]
    assert "," in body                                    # desetinná čárka

    path_dot = str(tmp_path / "tecka.csv")
    export_to_csv(result.metrics, path_dot, result.params, "test", decimal_comma=False)
    body_dot = open(path_dot, encoding="utf-8-sig").read().split("Index;")[1]
    assert "." in body_dot


def test_export_all_creates_three_files(series, tmp_path):
    result, folder = series
    outputs = export_all(result, str(tmp_path / "vse.csv"), os.path.basename(folder))
    assert set(outputs) == {"csv", "png", "json"}
    for path in outputs.values():
        assert os.path.getsize(path) > 0

    payload = json.load(open(outputs["json"], encoding="utf-8"))
    assert payload["souhrn"]["pocet_snimku"] == len(result.metrics)
    assert payload["parametry"]["bias_frames"] == 2


def test_summary_reports_extremes(series):
    result, _folder = series
    data = summarize(result)
    coverages = [m.total_coverage_pct for m in result.metrics]
    assert data["pokryti_max_pct"] == pytest.approx(max(coverages), abs=1e-4)
    assert data["pocet_snimku"] == len(result.metrics)
    assert sum(data["faze"].values()) == len(result.metrics)


def test_export_refuses_empty_data(tmp_path):
    with pytest.raises(ValueError):
        export_to_csv([], str(tmp_path / "prazdno.csv"))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 není v tomto prostředí dostupné")


@pytest.fixture(scope="module")
def app():
    application = qt.QApplication.instance() or qt.QApplication([])
    yield application


def test_gui_runs_full_analysis(app, tmp_path_factory):
    """Projde celý řetězec: výběr složky → analýza → tabulka → prohlížeč.

    Regrese na hlavní pád: dřív chyběl slot ``on_worker_error`` a stisk
    tlačítka „SPUSTIT ANALÝZU“ shodil celou aplikaci.
    """
    import time

    from PyQt6.QtCore import Qt

    import gui as gui_module

    folder = tmp_path_factory.mktemp("gui_data") / "mereni"
    frames = [make_frame(dots=[(40, 40), (100, 90)], noise=1.5, seed=i) for i in range(6)]
    write_series(str(folder), frames)

    window = gui_module.DarkfieldAnalyzerGUI()
    assert hasattr(window, "on_worker_error")     # dřív chybělo → pád aplikace

    window.current_base_dir = str(folder.parent)
    window.refresh_folder_list()
    assert window.folder_table.rowCount() >= 1
    for row in range(window.folder_table.rowCount()):
        path = window.folder_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if os.path.basename(str(path)) == "mereni":
            window.folder_table.selectRow(row)
            break
    assert window.selected_folder_path

    window.combo_res.setCurrentIndex(1)           # plné rozlišení
    window.spin_bias_frames.setValue(2)
    window.chk_exclude_bias.setChecked(True)      # nezávisle na uloženém nastavení
    window.combo_reference.setCurrentIndex(2)     # bias ze série (složka nemá reference)
    window.reference_dir_override = None
    window.start_analysis()

    deadline = time.time() + 120
    while window.worker and window.worker.isRunning() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(10):
        app.processEvents()

    assert window.current_result is not None
    assert len(window.current_result.metrics) == 4
    assert window.table_results.rowCount() == 4
    assert window.btn_export.isEnabled()
    assert window.btn_start.isEnabled()

    for index in range(window.viewer_widget.view_mode_combo.count()):
        window.viewer_widget.view_mode_combo.setCurrentIndex(index)
        window.viewer_widget.render_current_frame()
        assert "nelze zobrazit" not in window.viewer_widget.status_lbl.text()

    window.close()


def test_viewer_survives_bias_of_different_size(app, tmp_path_factory):
    """Regrese: prohlížeč počítal náhled ve 4× zmenšení proti 2× zmenšenému biasu."""
    import gui as gui_module
    from analyzer import compute_bias

    folder = tmp_path_factory.mktemp("viewer_data")
    frames = [make_frame(dots=[(60, 60)], noise=1.0, seed=i) for i in range(3)]
    paths = write_series(str(folder), frames)

    params = AnalysisParams(bias_frames=2, binning=2)
    bias = compute_bias(paths, params)

    viewer = gui_module.ImageViewerWidget()
    viewer.set_dataset(paths, bias, params)
    viewer.chk_fast.setChecked(True)              # náhled ve 4× zmenšení proti 2× biasu
    viewer.render_current_frame()
    assert "nelze zobrazit" not in viewer.status_lbl.text()
    assert viewer._last_rgb is not None


def test_gui_params_survive_round_trip(app):
    import gui as gui_module

    window = gui_module.DarkfieldAnalyzerGUI()
    window.spin_sigma.setValue(5.5)
    window.spin_haze.setValue(2.5)
    window.spin_min_area.setValue(7)
    window.combo_res.setCurrentIndex(2)
    window.radio_absolute.setChecked(True)

    params = window.get_current_params()
    assert params.sigma == 5.5
    assert params.haze_threshold == 2.5
    assert params.min_area_px == 7
    assert params.binning == 2
    assert params.threshold_mode == "absolute"
    window.close()


def test_cli_roi_parsing():
    from main import parse_roi

    assert parse_roi(None) is None
    assert parse_roi("10,20,300,200") == (10, 20, 300, 200)
    assert parse_roi(" 10 ; 20 ; 300 ; 200 ") == (10, 20, 300, 200)
    with pytest.raises(ValueError):
        parse_roi("10,20,300")
    with pytest.raises(ValueError):
        parse_roi("10,20,0,200")


def test_gui_roi_checkbox_controls_params(app):
    import gui as gui_module

    window = gui_module.DarkfieldAnalyzerGUI()
    window.chk_roi.setChecked(False)
    assert window.get_current_params().roi is None

    window.chk_roi.setChecked(True)
    for box, value in zip(window.spin_roi, (10, 20, 640, 480)):
        box.setValue(value)
    assert window.get_current_params().roi == (10, 20, 640, 480)
    window.close()


def test_composition_series_is_disjoint_and_ordered(series):
    from exporter import COMPOSITION_LAYERS, composition_series

    result, _folder = series
    layers = composition_series(result.metrics)

    assert [label for label, _color, _values in layers] == [
        label for _attr, label, _color in COMPOSITION_LAYERS
    ]
    assert len({color for _label, color, _values in layers}) == 4
    for index, metrics in enumerate(result.metrics):
        total = sum(values[index] for _label, _color, values in layers)
        assert total == pytest.approx(metrics.total_coverage_pct, abs=1e-6)


def test_composition_plot_draws_all_layers(series):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from exporter import plot_composition

    result, _folder = series
    fig = plt.figure(figsize=(8, 8))
    grid = fig.add_gridspec(3, 1)
    ax_area = fig.add_subplot(grid[0, 0])
    ax_particles = fig.add_subplot(grid[1, 0])
    ax_share = fig.add_subplot(grid[2, 0])

    plot_composition(ax_area, ax_share, result.metrics, ax_particles=ax_particles)

    assert len(ax_area.collections) == 4          # čtyři vrstvy
    assert len(ax_particles.collections) == 3     # bez zamlžení
    assert len(ax_share.patches) == 4             # čtyři segmenty pruhu
    assert ax_area.get_legend() is not None
    plt.close(fig)


def test_summary_contains_composition(series):
    result, _folder = series
    data = summarize(result)
    composition = data["slozeni_prumerne_pokryti_pct"]
    assert len(composition) == 4
    assert sum(composition.values()) == pytest.approx(
        sum(m.total_coverage_pct for m in result.metrics) / len(result.metrics), abs=1e-3
    )


def test_gui_has_composition_tab(app, tmp_path_factory):
    import gui as gui_module

    window = gui_module.DarkfieldAnalyzerGUI()
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert any("Složení" in title for title in titles)
    window.close()


def test_default_base_dir_prefers_bms_folder(monkeypatch):
    import gui as gui_module

    monkeypatch.setattr(os.path, "isdir", lambda path: path == gui_module.DEFAULT_BASE_DIR)
    assert gui_module.default_base_dir() == gui_module.DEFAULT_BASE_DIR
    assert gui_module.DEFAULT_BASE_DIR == r"C:\Users\Programovani\Downloads\BMS fotky"


def test_default_base_dir_falls_back_when_missing(monkeypatch):
    import gui as gui_module

    home = os.path.expanduser("~")
    monkeypatch.setattr(os.path, "isdir", lambda path: path == home)
    assert gui_module.default_base_dir() == home


@pytest.mark.parametrize(
    "screen_w, screen_h",
    [(1280, 680), (1366, 728), (1536, 824), (1920, 1040)],
    ids=["FHD@150%", "1366x768", "FHD@125%", "FHD@100%"],
)
def test_window_fits_on_screen(app, screen_w, screen_h):
    """Regrese: pevná velikost 1440×900 byla větší než plocha displeje.

    Na Full HD se škálováním Windows 150 % má plocha jen 1280×720 logických
    bodů, takže spodní část okna (včetně tlačítka pro spuštění) byla mimo obraz.
    """
    from PyQt6.QtCore import QRect

    import gui as gui_module

    window = gui_module.DarkfieldAnalyzerGUI()
    window.settings.remove("window_geometry")
    window._available_geometry = lambda: QRect(0, 0, screen_w, screen_h)
    window.restore_geometry()

    assert window.width() <= screen_w
    assert window.height() <= screen_h
    # Rozvržení se musí umět zmenšit ještě víc, jinak okno nejde zvětšovat/menšit
    hint = window.minimumSizeHint()
    assert hint.width() <= screen_w and hint.height() <= screen_h
    window.close()


def test_window_geometry_is_clamped_to_smaller_screen(app):
    """Uložená geometrie z většího monitoru se ořízne na aktuální obrazovku."""
    from PyQt6.QtCore import QRect

    import gui as gui_module

    big = gui_module.DarkfieldAnalyzerGUI()
    big._available_geometry = lambda: QRect(0, 0, 2560, 1400)
    big.restore_geometry()
    big.resize(2400, 1300)
    big.settings.setValue("window_geometry", big.saveGeometry())
    big.close()

    small = gui_module.DarkfieldAnalyzerGUI()
    small._available_geometry = lambda: QRect(0, 0, 1280, 680)
    small.restore_geometry()
    assert small.width() <= 1280 and small.height() <= 680
    small.settings.remove("window_geometry")
    small.close()


def test_left_panel_is_scrollable(app):
    """Panel s parametry musí jít rolovat, jinak roztáhne okno mimo displej."""
    import gui as gui_module

    window = gui_module.DarkfieldAnalyzerGUI()
    left = window.splitter.widget(0)
    assert isinstance(left, qt.QScrollArea)
    assert left.widgetResizable()
    assert left.maximumWidth() <= 520
    window.close()


# ---------------------------------------------------------------------------
# Referenční pozadí v GUI
# ---------------------------------------------------------------------------

def _reference_workspace(root):
    """Vytvoří strukturu „BMS fotky“ s referencemi a jednou sérií."""
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_reference import background, write_reference, write_series

    ref_dir = os.path.join(root, "reference")
    base = background()
    write_reference(ref_dir, datetime(2026, 9, 8, 9, 0, 0), base * 0.5)
    write_reference(ref_dir, datetime(2026, 9, 8, 14, 29, 10), base)
    folder = os.path.join(root, "mereni")
    write_series(folder, base)
    return folder


def test_gui_previews_the_chosen_reference(app, tmp_path):
    """Po výběru složky musí panel ukázat, která reference se použije."""
    from PyQt6.QtCore import Qt

    import gui as gui_module

    root = str(tmp_path / "BMS fotky")
    folder = _reference_workspace(root)

    window = gui_module.DarkfieldAnalyzerGUI()
    window.reference_dir_override = None
    window.combo_reference.setCurrentIndex(0)          # automaticky
    window.current_base_dir = root
    window.refresh_folder_list()
    for row in range(window.folder_table.rowCount()):
        path = window.folder_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if os.path.basename(str(path)) == "mereni":
            window.folder_table.selectRow(row)
            break

    text = window.lbl_reference_pick.text()
    assert "reference_20260908_142910.npz" in text
    assert "14:29:10" in text

    window.combo_reference.setCurrentIndex(2)          # vždy ze série
    assert "prvních" in window.lbl_reference_pick.text()
    assert not window.chk_match_level.isEnabled()
    window.close()


def test_gui_analysis_uses_the_reference_and_keeps_all_frames(app, tmp_path):
    """Celý řetězec v GUI: s referencí se nespotřebují snímky série na bias."""
    import time

    from PyQt6.QtCore import Qt

    import gui as gui_module

    root = str(tmp_path / "BMS fotky")
    folder = _reference_workspace(root)
    expected = len(list_image_files(folder))

    window = gui_module.DarkfieldAnalyzerGUI()
    window.reference_dir_override = None
    window.combo_reference.setCurrentIndex(0)
    window.chk_match_level.setChecked(True)
    window.combo_mono.setCurrentIndex(0)
    window.combo_res.setCurrentIndex(1)                # plné rozlišení
    window.current_base_dir = root
    window.refresh_folder_list()
    for row in range(window.folder_table.rowCount()):
        path = window.folder_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if os.path.basename(str(path)) == "mereni":
            window.folder_table.selectRow(row)
            break

    window.start_analysis()
    deadline = time.time() + 120
    while window.worker and window.worker.isRunning() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(10):
        app.processEvents()

    assert window.current_result is not None
    assert window.current_result.bias.is_external
    assert len(window.current_result.metrics) == expected      # nic neubylo
    assert window.table_results.rowCount() == expected
    assert "reference_20260908_142910.npz" in window.summary_browser.toPlainText()
    window.close()


def test_gui_params_carry_reference_and_color_settings(app, tmp_path):
    import gui as gui_module

    window = gui_module.DarkfieldAnalyzerGUI()
    window.combo_reference.setCurrentIndex(1)
    window.combo_mono.setCurrentIndex(2)
    window.chk_match_level.setChecked(False)
    window.reference_dir_override = str(tmp_path)

    params = window.get_current_params()
    assert params.reference_mode == "reference"
    assert params.mono_mode == "maximum"
    assert params.match_reference_level is False
    assert params.reference_dir == str(tmp_path)
    assert params.mono_label == "maximum kanálů"

    # Nastavení se ukládá do QSettings a přežilo by do dalších testů i spuštění.
    window.combo_reference.setCurrentIndex(0)
    window.chk_match_level.setChecked(True)
    window.combo_mono.setCurrentIndex(0)
    window.reference_dir_override = None
    window.close()


def test_help_text_has_no_stray_escape_sequences():
    """Windows cesty v nápovědě se musí escapovat, jinak z nich zmizí znaky.

    Regrese: ``<code>…\\BMS fotky\\reference</code>`` v běžném (ne raw) řetězci
    Pythonu udělá z ``\\r`` návrat vozíku – v nápovědě se pak zobrazilo
    „BMS fotky eference“.
    """
    from help_text import HELP_HTML

    stray = {ch for ch in HELP_HTML if ord(ch) < 32 and ch != "\n"}
    assert not stray, f"nápověda obsahuje řídicí znaky: {[hex(ord(c)) for c in stray]}"
    assert "\\BMS fotky\\reference" in HELP_HTML
