"""Testy analytického jádra.

Součástí jsou i regresní testy na tři konkrétní příčiny pádů původní verze:
přetečení typu labelů (CV_16U), nesoulad rozměrů biasu a snímku a rozbitý
odhad šumu u snímků, které samy vstoupily do biasu.

Spuštění::

    py -m pytest -q
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (  # noqa: E402
    AnalysisParams,
    analyze_prepared_frame,
    analyze_series,
    apply_geometry,
    compute_bias,
    compute_cleanliness_score,
    compute_rates_and_phases,
    estimate_noise,
    match_shape,
    spatial_distribution,
)
from frameio import (  # noqa: E402
    imwrite_unicode,
    list_image_files,
    list_measurement_folders,
    load_frame,
    parse_timestamp_from_filename,
)

SHAPE = (240, 320)
T0 = datetime(2026, 9, 7, 14, 56, 21)


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def make_frame(dots=(), lines=(), disks=(), haze=0.0, noise=0.0, seed=0) -> np.ndarray:
    """Sestaví syntetický snímek v temném poli jako float32 ADU."""
    img = np.zeros(SHAPE, dtype=np.float32)
    for x, y in dots:
        cv2.circle(img, (x, y), 1, 200.0, -1)
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), 180.0, 2)
    for x, y, r in disks:
        cv2.circle(img, (x, y), r, 150.0, -1)
    if haze:
        yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(np.float32)
        img += haze * np.exp(-(((xx - SHAPE[1] / 2) / (SHAPE[1] * 0.6)) ** 2
                               + ((yy - SHAPE[0] / 2) / (SHAPE[0] * 0.6)) ** 2))
    if noise:
        rng = np.random.default_rng(seed)
        img += rng.normal(0.0, noise, SHAPE).astype(np.float32)
    return np.clip(img, 0, 255)


def analyze(image, bias=None, params=None, binning=1, generate_masks=False):
    params = params or AnalysisParams(binning=1, min_threshold_adu=1.5)
    bias = np.zeros_like(image) if bias is None else bias
    return analyze_prepared_frame(
        image=image, bias=bias, params=params, index=0, filename="test.png",
        filepath="test.png", timestamp=T0, t0=T0, binning=binning,
        generate_masks=generate_masks,
    )


def write_series(folder, frames, start=T0, fps=2.0):
    """Uloží sérii uint8 PNG s časovými razítky v názvu."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        stamp = start + timedelta(seconds=i / fps)
        name = f"df_{i + 1:05d}_{stamp.strftime('%Y%m%d_%H%M%S')}_{stamp.microsecond // 1000:03d}.png"
        path = os.path.join(folder, name)
        imwrite_unicode(path, frame.astype(np.uint8))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Časová razítka a hledání souborů
# ---------------------------------------------------------------------------

def test_timestamp_with_milliseconds():
    stamp = parse_timestamp_from_filename("df_00001_20260907_145622_018.png", 0.0)
    assert (stamp.year, stamp.month, stamp.day) == (2026, 9, 7)
    assert (stamp.hour, stamp.minute, stamp.second) == (14, 56, 22)
    assert stamp.microsecond == 18_000


def test_timestamp_ignores_digits_in_path(tmp_path):
    """Razítko se čte z názvu souboru, ne z cesty (jinak ho přebije číslo ve složce)."""
    folder = tmp_path / "BMS_20200101_000000"
    folder.mkdir()
    target = folder / "df_00002_20260907_145622_500.png"
    target.write_bytes(b"")
    stamp = parse_timestamp_from_filename(str(target), 0.0)
    assert stamp.year == 2026 and stamp.microsecond == 500_000


def test_natural_sort_and_folder_listing(tmp_path):
    folder = tmp_path / "mereni"
    frames = [make_frame() for _ in range(3)]
    os.makedirs(folder, exist_ok=True)
    for i in (10, 2, 1):
        imwrite_unicode(str(folder / f"img{i}.png"), frames[0].astype(np.uint8))

    listed = [os.path.basename(p) for p in list_image_files(str(folder))]
    assert listed == ["img1.png", "img2.png", "img10.png"]

    # kořenová složka se snímky se musí v nabídce objevit také
    found = dict(list_measurement_folders(str(tmp_path)))
    assert str(folder) in found and found[str(folder)] == 3


def test_unicode_path_roundtrip(tmp_path):
    """cv2.imread na Windows neumí diakritiku – čtení musí jít přes imdecode."""
    folder = tmp_path / "Měření žluťoučký kůň"
    folder.mkdir()
    path = str(folder / "snímek_ěščřž.png")
    assert imwrite_unicode(path, make_frame(dots=[(50, 50)]).astype(np.uint8))
    frame = load_frame(path)
    assert frame.data.shape == SHAPE
    assert frame.data.max() > 100


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

def test_median_bias_rejects_stray_particle(tmp_path):
    """Náhodná částice v jednom bias snímku nesmí zkazit referenci."""
    clean = make_frame()
    dirty = make_frame(disks=[(160, 120, 6)])
    paths = write_series(str(tmp_path / "bias"), [dirty, clean, clean, clean])

    median_bias = compute_bias(paths, AnalysisParams(bias_frames=3, bias_method="median", binning=1))
    mean_bias = compute_bias(paths, AnalysisParams(bias_frames=3, bias_method="mean", binning=1))

    assert median_bias.data[120, 160] < 1.0        # částice odfiltrována
    assert mean_bias.data[120, 160] > 30.0         # průměr ji do reference propíše


def test_bias_frames_are_flagged(tmp_path):
    frames = [make_frame(noise=2.0, seed=i) for i in range(6)]
    paths = write_series(str(tmp_path / "flag"), frames)
    result = analyze_series(
        paths, AnalysisParams(bias_frames=2, binning=1, exclude_bias_from_series=False)
    )
    assert [m.is_bias_frame for m in result.metrics][:2] == [True, True]
    assert not any(m.is_bias_frame for m in result.metrics[2:])


# ---------------------------------------------------------------------------
# Odhad šumu
# ---------------------------------------------------------------------------

def test_noise_estimate_matches_known_sigma():
    rng = np.random.default_rng(3)
    field = rng.normal(0.0, 3.0, (400, 400)).astype(np.float32)
    _median, sigma = estimate_noise(field)
    assert 2.4 < sigma < 3.6


def test_noise_estimate_survives_spike_at_zero():
    """Regrese: MAD selhává, když je polovina hodnot přesně nulová.

    Přesně to nastane u snímku, který sám vstoupil do mediánového biasu.
    Původní odhad vracel σ ≈ 0, práh spadl pod šum a analýza „našla“
    desítky tisíc neexistujících částic.
    """
    rng = np.random.default_rng(5)
    field = rng.normal(0.0, 3.0, (400, 400)).astype(np.float32)
    field[rng.random(field.shape) < 0.55] = 0.0     # pík v nule

    mad_only = float(np.median(np.abs(field - np.median(field)))) * 1.4826
    _median, sigma = estimate_noise(field)

    assert mad_only < 0.5          # samotný MAD je rozbitý
    assert sigma > 1.0             # kombinovaný odhad drží (byť konzervativně nižší)


# ---------------------------------------------------------------------------
# Detekce a klasifikace
# ---------------------------------------------------------------------------

def test_counts_isolated_particles():
    """Šum se odečítá proti reálnému biasu, ne proti nule (jinak je jednostranný)."""
    dots = [(30 + 25 * i, 40 + 30 * (i % 5)) for i in range(12)]
    bias = make_frame(haze=6.0, noise=1.0, seed=99)
    image = np.clip(bias + make_frame(dots=dots), 0, 255)
    metrics, _ = analyze(image, bias=bias)
    assert metrics.point_count == 12
    assert metrics.cluster_count == 0
    assert metrics.total_coverage_pct > 0


def test_diagonal_fiber_is_not_classified_as_cluster():
    """Regrese: protáhlost z opsaného obdélníku je u šikmého vlákna ≈ 1."""
    metrics, _ = analyze(make_frame(lines=[(60, 60, 180, 180)]))
    assert metrics.fiber_count == 1
    assert metrics.cluster_count == 0
    assert metrics.fiber_total_length_px > 100


def test_large_disk_is_cluster():
    metrics, _ = analyze(make_frame(disks=[(160, 120, 12)]))
    assert metrics.cluster_count == 1
    assert metrics.fiber_count == 0
    assert metrics.max_particle_area_px > 300


def test_haze_is_separated_from_particles():
    metrics, _ = analyze(make_frame(dots=[(80, 80), (200, 150)], haze=20.0))
    assert metrics.haze_coverage_pct > 50.0
    assert metrics.point_count == 2      # opar nesmí být rozdroben na částice


def test_min_area_filters_single_pixels():
    image = np.zeros(SHAPE, dtype=np.float32)
    rng = np.random.default_rng(11)
    ys = rng.integers(0, SHAPE[0], 200)
    xs = rng.integers(0, SHAPE[1], 200)
    image[ys, xs] = 200.0
    # jednotlivé pixely: s min_area=1 se najdou, s min_area=5 zmizí

    loose = AnalysisParams(binning=1, min_area_px=1)
    strict = AnalysisParams(binning=1, min_area_px=5)
    assert analyze(image, params=loose)[0].total_particle_count > 100
    assert analyze(image, params=strict)[0].total_particle_count == 0


def test_binning_keeps_coverage_comparable():
    image = make_frame(disks=[(160, 120, 20)])
    full, _ = analyze(image, params=AnalysisParams(binning=1), binning=1)
    binned_img = apply_geometry(image, 2, None)
    binned, _ = analyze(binned_img, params=AnalysisParams(binning=2), binning=2)
    assert full.total_coverage_pct == pytest.approx(binned.total_coverage_pct, rel=0.15)
    assert full.total_area_px == pytest.approx(binned.total_area_px, rel=0.2)


# ---------------------------------------------------------------------------
# Regrese na pády
# ---------------------------------------------------------------------------

def test_more_than_65535_objects_does_not_crash():
    """Regrese: ``ltype=cv2.CV_16U`` vyhodí výjimku nad 65 535 objekty."""
    rng = np.random.default_rng(1)
    image = np.zeros((1080, 1920), dtype=np.float32)
    ys = rng.integers(0, 1080, 120_000)
    xs = rng.integers(0, 1920, 120_000)
    image[ys, xs] = 255.0

    with pytest.raises(cv2.error):        # chování původní implementace
        cv2.connectedComponentsWithStats(
            (image > 0).astype(np.uint8), connectivity=8, ltype=cv2.CV_16U
        )

    metrics, _ = analyze(image, params=AnalysisParams(binning=1, min_area_px=1))
    assert metrics.total_particle_count > 50_000


def test_bias_shape_mismatch_is_handled():
    """Regrese: prohlížeč posílal bias v jiném rozlišení než snímek."""
    image = make_frame(dots=[(100, 100)])
    small_bias = np.zeros((SHAPE[0] // 2, SHAPE[1] // 2), dtype=np.float32)
    assert match_shape(small_bias, image.shape).shape == image.shape
    metrics, _ = analyze(image, bias=small_bias)
    assert metrics.point_count == 1


def test_corrupted_file_does_not_stop_series(tmp_path):
    folder = tmp_path / "series"
    frames = [make_frame(dots=[(50, 50)], noise=1.5, seed=i) for i in range(6)]
    write_series(str(folder), frames)
    broken = str(folder / "df_00007_20260907_145625_000.png")
    with open(broken, "wb") as handle:
        handle.write(b"tohle neni obrazek")

    result = analyze_series(list_image_files(str(folder)), AnalysisParams(bias_frames=2, binning=1))
    assert len(result.metrics) == 4        # 6 snímků minus 2 bias (ty se ve výchozím stavu vynechávají)
    assert len(result.failed_files) == 1
    assert os.path.basename(result.failed_files[0][0]) == os.path.basename(broken)
    assert result.warnings


# ---------------------------------------------------------------------------
# Odvozené veličiny
# ---------------------------------------------------------------------------

def test_spatial_heterogeneity_range():
    uniform = np.ones(SHAPE, dtype=np.uint8)
    local = np.zeros(SHAPE, dtype=np.uint8)
    local[:30, :40] = 1

    het_uniform, cx, cy = spatial_distribution(uniform)
    het_local, cx_local, cy_local = spatial_distribution(local)

    assert het_uniform < 1.0
    assert het_local > 30.0
    assert (cx, cy) == pytest.approx((50.0, 50.0), abs=1.0)
    assert cx_local < 15.0 and cy_local < 15.0


def test_cleanliness_score_is_monotonic_and_bounded():
    assert compute_cleanliness_score(0.0, 0.0, 0.0) == 100.0
    assert 0.0 <= compute_cleanliness_score(100.0, 255.0, 100_000.0) <= 100.0
    scores = [compute_cleanliness_score(cov, 0.0, 0.0) for cov in (0.0, 0.5, 1.0, 5.0, 20.0)]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_rates_and_phases_follow_the_process():
    def metrics_with(coverage, haze, index, time_s):
        image = make_frame()
        m, _ = analyze(image)
        m.index = index
        m.time_s = time_s
        m.total_coverage_pct = coverage
        m.haze_coverage_pct = haze
        return m

    coverages = [0.1, 0.1, 0.1, 5.0, 20.0, 40.0, 40.0, 40.0, 25.0, 10.0, 1.0, 1.0]
    series = [metrics_with(cov, cov, i, i * 0.5) for i, cov in enumerate(coverages)]
    compute_rates_and_phases(series)

    phases = [m.phase for m in series]
    assert phases[0] == "stabilní"
    assert "nárůst / zamlžování" in phases[3:6]
    assert "odpařování / ústup" in phases[8:11]
    assert series[4].rate_coverage_pct_per_s > 0
    assert series[9].rate_coverage_pct_per_s < 0


def test_rates_survive_identical_timestamps():
    """Stejná razítka u rychlé série nesmí způsobit dělení nulou."""
    series = []
    for i in range(5):
        m, _ = analyze(make_frame())
        m.index = i
        m.time_s = 0.0
        m.total_coverage_pct = float(i)
        series.append(m)
    compute_rates_and_phases(series)
    assert all(math.isfinite(m.rate_coverage_pct_per_s) for m in series)


# ---------------------------------------------------------------------------
# Série
# ---------------------------------------------------------------------------

def test_parallel_and_sequential_results_match(tmp_path):
    frames = [make_frame(dots=[(40 + 10 * i, 60)], noise=1.5, seed=i) for i in range(8)]
    paths = write_series(str(tmp_path / "par"), frames)

    sequential = analyze_series(paths, AnalysisParams(bias_frames=2, binning=1, workers=1))
    parallel = analyze_series(paths, AnalysisParams(bias_frames=2, binning=1, workers=4))

    assert [m.index for m in sequential.metrics] == [m.index for m in parallel.metrics]
    for a, b in zip(sequential.metrics, parallel.metrics):
        assert a.total_coverage_pct == pytest.approx(b.total_coverage_pct)
        assert a.total_particle_count == b.total_particle_count


def test_cancel_stops_series(tmp_path):
    frames = [make_frame(noise=1.0, seed=i) for i in range(12)]
    paths = write_series(str(tmp_path / "cancel"), frames)
    state = {"calls": 0}

    def should_cancel():
        state["calls"] += 1
        return state["calls"] > 3

    result = analyze_series(paths, AnalysisParams(bias_frames=1, binning=1, workers=1),
                            should_cancel=should_cancel)
    assert result.cancelled
    assert len(result.metrics) < len(paths)


def test_16bit_input_gives_same_metrics(tmp_path):
    """12/16bitová kamera musí dát stejná čísla jako 8bitová (společná škála ADU)."""
    dots = [(40, 40), (120, 80), (200, 160)]
    frame8 = make_frame(dots=dots)
    frame16 = (frame8.astype(np.float32) * 257.0).astype(np.uint16)

    folder8 = tmp_path / "osm"
    folder16 = tmp_path / "sestnact"
    os.makedirs(folder8), os.makedirs(folder16)
    for i in range(3):
        stamp = T0 + timedelta(seconds=i)
        name = f"df_{i:05d}_{stamp.strftime('%Y%m%d_%H%M%S')}_000.png"
        imwrite_unicode(str(folder8 / name), frame8.astype(np.uint8))
        imwrite_unicode(str(folder16 / name), frame16)

    params = AnalysisParams(bias_frames=1, binning=1)
    res8 = analyze_series(list_image_files(str(folder8)), params)
    res16 = analyze_series(list_image_files(str(folder16)), params)
    assert res8.metrics[-1].point_count == res16.metrics[-1].point_count == 0  # shodné s biasem

    # a s částicemi navíc oproti biasu
    extra = make_frame(dots=dots + [(260, 200)])
    imwrite_unicode(str(folder8 / "df_00009_20260907_145630_000.png"), extra.astype(np.uint8))
    imwrite_unicode(str(folder16 / "df_00009_20260907_145630_000.png"), (extra * 257.0).astype(np.uint16))
    res8 = analyze_series(list_image_files(str(folder8)), params)
    res16 = analyze_series(list_image_files(str(folder16)), params)
    assert res8.metrics[-1].point_count == res16.metrics[-1].point_count == 1


def test_time_axis_falls_back_to_file_mtime(tmp_path):
    """Bez razítka v názvu se použije čas modifikace souboru."""
    folder = tmp_path / "bez_casu"
    os.makedirs(folder)
    for i in range(4):
        path = str(folder / f"snimek_{i}.png")
        imwrite_unicode(path, make_frame().astype(np.uint8))
        os.utime(path, (1_800_000_000 + i * 5, 1_800_000_000 + i * 5))
    result = analyze_series(list_image_files(str(folder)), AnalysisParams(bias_frames=1, binning=1))
    assert not result.synthetic_time_axis
    assert result.metrics[-1].time_s == pytest.approx(15.0, abs=0.01)


def test_synthetic_time_axis_when_no_time_information(tmp_path):
    """Když jsou všechna razítka stejná, dopočítá se rovnoměrná osa."""
    folder = tmp_path / "stejny_cas"
    os.makedirs(folder)
    for i in range(4):
        path = str(folder / f"snimek_{i}.png")
        imwrite_unicode(path, make_frame().astype(np.uint8))
        os.utime(path, (1_800_000_000, 1_800_000_000))
    params = AnalysisParams(bias_frames=1, binning=1, assumed_fps=2.0)
    result = analyze_series(list_image_files(str(folder)), params)
    assert result.synthetic_time_axis
    assert result.metrics[-1].time_s == pytest.approx(1.5, abs=0.01)
    assert result.warnings


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def test_roi_limits_analysis_to_selected_area():
    """Částice mimo výřez se nesmí započítat."""
    inside = make_frame(dots=[(60, 60)])
    outside = make_frame(dots=[(60, 60), (280, 200)])

    params_full = AnalysisParams(binning=1)
    params_roi = AnalysisParams(binning=1, roi=(0, 0, 160, 120))

    assert analyze(outside, params=params_full)[0].point_count == 2

    bias_full = np.zeros(SHAPE, dtype=np.float32)
    cropped_image = apply_geometry(outside, 1, params_roi.roi)
    cropped_bias = apply_geometry(bias_full, 1, params_roi.roi)
    metrics, _ = analyze_prepared_frame(
        image=cropped_image, bias=cropped_bias, params=params_roi, index=0,
        filename="roi.png", filepath="roi.png", timestamp=T0, t0=T0, binning=1,
    )
    assert metrics.point_count == 1
    assert cropped_image.shape == (120, 160)
    assert analyze(inside, params=params_full)[0].point_count == 1


# ---------------------------------------------------------------------------
# Rozklad pokrytí podle typů
# ---------------------------------------------------------------------------

def test_composition_components_sum_to_total_coverage():
    """Složky musí být disjunktní – jinak by graf složení lhal.

    Maska oparu a masky částic se překrývají, proto se opar pro rozklad počítá
    bez plochy částic.
    """
    image = make_frame(dots=[(60, 60), (150, 90)], disks=[(200, 150, 11)],
                       lines=[(40, 180, 120, 220)], haze=18.0, noise=1.0, seed=3)
    bias = make_frame(noise=1.0, seed=4)
    metrics, _ = analyze(np.clip(image + bias, 0, 255), bias=bias)

    parts = (metrics.haze_only_coverage_pct + metrics.point_area_pct
             + metrics.cluster_area_pct + metrics.fiber_area_pct)
    assert parts == pytest.approx(metrics.total_coverage_pct, abs=1e-6)
    assert metrics.haze_only_coverage_pct <= metrics.haze_coverage_pct + 1e-9
    assert metrics.total_coverage_pct > 0


def test_haze_only_is_zero_without_haze():
    metrics, _ = analyze(make_frame(dots=[(60, 60), (150, 90)]))
    assert metrics.haze_only_coverage_pct == pytest.approx(0.0, abs=1e-6)
    assert metrics.point_area_pct > 0


# ---------------------------------------------------------------------------
# Pojistky proti cizím souborům ve složce
# ---------------------------------------------------------------------------

def test_exported_charts_are_not_treated_as_frames(tmp_path):
    """Regrese: exportované grafy uložené ve složce se braly jako snímky.

    Řadí se abecedně před ``df_00001_…``, takže se z nich stal dokonce
    referenční bias a celá analýza se počítala proti obrázku grafu.
    """
    folder = tmp_path / "mereni"
    frames = [make_frame(dots=[(50, 50), (150, 100)], noise=1.5, seed=i) for i in range(6)]
    paths = write_series(str(folder), frames)

    chart = np.full((900, 1200), 240, dtype=np.uint8)      # „graf“ jiného rozměru
    imwrite_unicode(str(folder / "analyza_mereni_grafy.png"), chart)
    imwrite_unicode(str(folder / "df_00003_nahled.png"), chart)

    listed = list_image_files(str(folder))
    assert listed == paths
    assert list_measurement_folders(str(tmp_path)) == [(str(folder), len(paths))]

    # bez filtru by se do seznamu dostaly a řadily by se první
    unfiltered = list_image_files(str(folder), skip_outputs=False)
    assert len(unfiltered) == len(paths) + 2
    assert os.path.basename(unfiltered[0]).startswith("analyza_")


def test_foreign_image_is_rejected_and_does_not_shift_bias(tmp_path):
    """Cizí obrázek s nevinným názvem nesmí sérii rozhodit.

    Druhá pojistka je rozlišení: soubor, který neodpovídá zbytku série,
    se vyřadí ještě před sestavením časové osy, takže nezmění ani bias,
    ani časy snímků.
    """
    clean = tmp_path / "cista"
    dirty = tmp_path / "spinava"
    frames = [make_frame(dots=[(50, 50), (150, 100)], noise=1.5, seed=i) for i in range(8)]
    clean_paths = write_series(str(clean), frames)
    write_series(str(dirty), frames)
    # sortuje se před "df_…" a má jiné rozlišení
    imwrite_unicode(str(dirty / "aaa_screenshot.png"), np.zeros((600, 800), dtype=np.uint8))

    params = AnalysisParams(bias_frames=3, binning=1)
    reference = analyze_series(clean_paths, params)
    result = analyze_series(list_image_files(str(dirty)), params)

    assert len(result.metrics) == len(reference.metrics)
    assert not result.synthetic_time_axis
    assert [os.path.basename(p) for p in result.bias.frame_paths] == [
        os.path.basename(p) for p in clean_paths[:3]
    ]
    assert len(result.failed_files) == 1
    rejected_path, reason = result.failed_files[0]
    assert os.path.basename(rejected_path) == "aaa_screenshot.png"
    assert "rozlišení" in reason

    for expected, actual in zip(reference.metrics, result.metrics):
        assert actual.filename == expected.filename
        assert actual.time_s == pytest.approx(expected.time_s)
        assert actual.total_coverage_pct == pytest.approx(expected.total_coverage_pct)


def test_bias_uses_dominant_resolution_even_if_first_file_differs(tmp_path):
    folder = tmp_path / "hlasovani"
    frames = [make_frame(dots=[(60, 60)], noise=1.0, seed=i) for i in range(5)]
    write_series(str(folder), frames)
    imwrite_unicode(str(folder / "aaa_cizi.png"), np.zeros((100, 120), dtype=np.uint8))

    bias = compute_bias(list_image_files(str(folder)), AnalysisParams(bias_frames=3, binning=1))
    assert bias.source_shape == SHAPE
    assert bias.frames_used == 3
    assert all("aaa_cizi" not in path for path in bias.frame_paths)
    assert any("aaa_cizi" in path for path, _reason in bias.rejected_paths)
