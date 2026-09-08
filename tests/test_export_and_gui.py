"""Testy exportu a headless test grafického rozhraní.

GUI testy se přeskočí, pokud v prostředí není použitelná knihovna Qt.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import AnalysisParams, analyze_series  # noqa: E402
from exporter import CSV_COLUMNS, export_all, export_to_csv, summarize  # noqa: E402
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
