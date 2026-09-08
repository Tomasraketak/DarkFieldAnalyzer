"""Testy načítání referenčního pozadí ze složky ``reference`` a barevných snímků.

Reference je dvojice souborů, kterou ukládá záznamový program BMS Cam Control:
``reference_RRRRMMDD_HHMMSS.npz`` (průměr N tmavých snímků) a stejnojmenný
``.json`` s nastavením kamery. Aplikace musí vybrat tu, která vznikla
**naposledy před** začátkem měření.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (  # noqa: E402
    AnalysisParams,
    analyze_series,
    bias_from_reference,
    estimate_reference_offset,
    probe_series_shape,
    resolve_reference,
)
from frameio import (  # noqa: E402
    MONO_MODES,
    imwrite_unicode,
    list_image_files,
    load_frame,
    normalize_mono_mode,
    to_mono,
)
from reference import (  # noqa: E402
    ReferenceError,
    find_reference_dir,
    list_references,
    load_reference,
    read_reference_record,
    scale_to_adu,
    select_reference,
    series_start_time,
)

SHAPE = (120, 160)
MEASUREMENT = datetime(2026, 9, 8, 14, 35, 0)


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def background(level: float = 20.0) -> np.ndarray:
    """Nerovnoměrné pozadí temného pole (vinětace + jasnější roh)."""
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(np.float32)
    gradient = 1.0 - 0.4 * ((xx / SHAPE[1] - 0.5) ** 2 + (yy / SHAPE[0] - 0.5) ** 2)
    return (level * gradient).astype(np.float32)


def write_reference(folder, stamp: datetime, plane: np.ndarray, frames: int = 16,
                    with_json: bool = True, channels: bool = False) -> str:
    """Uloží referenci ve formátu, který zapisuje BMS Cam Control."""
    os.makedirs(folder, exist_ok=True)
    name = f"reference_{stamp:%Y%m%d_%H%M%S}"
    describe = (f"{plane.shape[1]}×{plane.shape[0]} px · {frames} snímků · "
                f"úroveň {plane.mean():.1f} ADU · {stamp:%d.%m.%Y %H:%M:%S}")

    payload = {
        "mean_mono": plane.astype(np.float32),
        "frames_mono": np.int64(frames),
        "created_mono": np.str_(stamp.isoformat()),
        "note_mono": np.str_(describe),
    }
    if channels:
        payload.update({
            "mean_blue": (plane * 1.2).astype(np.float32),
            "mean_green": plane.astype(np.float32),
            "mean_red": (plane * 0.7).astype(np.float32),
        })
    npz_path = os.path.join(folder, name + ".npz")
    np.savez_compressed(npz_path, **payload)

    if with_json:
        meta = {
            "app": "BMS Cam Control",
            "format": 2,
            "saved": (stamp + timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "camera": {"resolution_size": [plane.shape[1], plane.shape[0]]},
            "reference": {"file": name + ".npz", "note": describe, "describe": describe},
        }
        with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
    return npz_path


def scene(index: int, seed: int = 0) -> np.ndarray:
    """Kontaminace: pár částic, jejichž počet s časem roste."""
    img = np.zeros(SHAPE, dtype=np.float32)
    rng = np.random.default_rng(seed + index)
    for _ in range(6 + 2 * index):
        x, y = int(rng.integers(10, SHAPE[1] - 10)), int(rng.integers(10, SHAPE[0] - 10))
        cv2.circle(img, (x, y), 2, 120.0, -1)
    return img


def write_series(folder, base: np.ndarray, count: int = 6, color: bool = False,
                 start: datetime = MEASUREMENT, offset: float = 0.0) -> list:
    """Uloží sérii snímků s pozadím ``base`` a rostoucí kontaminací."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(7)
    paths = []
    for i in range(count):
        stamp = start + timedelta(seconds=5 * i)
        mono = base + offset + scene(i) + rng.normal(0.0, 0.8, SHAPE).astype(np.float32)
        mono = np.clip(mono, 0, 255)
        if color:
            # Kanály se liší, ale vážený jas Rec.601 dá přesně `mono`,
            # takže barevná i mono varianta série musí vyjít stejně.
            blue = np.clip(mono * 1.25, 0, 255)
            green = np.clip(mono * 1.05, 0, 255)
            red = np.clip((mono - 0.114 * blue - 0.587 * green) / 0.299, 0, 255)
            image = np.stack([blue, green, red], axis=2).astype(np.uint8)
        else:
            image = mono.astype(np.uint8)
        name = f"df_{i + 1:05d}_{stamp:%Y%m%d_%H%M%S}_000.png"
        path = os.path.join(folder, name)
        imwrite_unicode(path, image)
        paths.append(path)
    return paths


@pytest.fixture()
def workspace(tmp_path):
    """Složka ve tvaru „BMS fotky“: reference/ vedle složek s měřeními."""
    root = tmp_path / "BMS fotky"
    ref_dir = root / "reference"
    base = background()

    write_reference(str(ref_dir), datetime(2026, 9, 8, 9, 0, 0), base * 0.5)
    write_reference(str(ref_dir), datetime(2026, 9, 8, 14, 29, 10), base)      # ta správná
    write_reference(str(ref_dir), datetime(2026, 9, 8, 16, 0, 0), base * 1.5)

    paths = write_series(str(root / "mereni"), base)
    return {"root": str(root), "reference_dir": str(ref_dir), "folder": str(root / "mereni"),
            "paths": paths, "base": base}


# ---------------------------------------------------------------------------
# Hledání a čtení referencí
# ---------------------------------------------------------------------------

def test_reference_dir_is_found_next_to_measurement(workspace):
    """Složka „reference“ leží o úroveň výš než měření – musí se najít sama."""
    found = find_reference_dir(workspace["folder"])
    assert found is not None
    assert os.path.normpath(found) == os.path.normpath(workspace["reference_dir"])


def test_empty_reference_dir_does_not_stop_the_search(tmp_path):
    """Prázdná složka „reference“ u měření nesmí zastínit tu naplněnou výš."""
    root = tmp_path / "BMS fotky"
    real = root / "reference"
    write_reference(str(real), datetime(2026, 9, 8, 9, 0, 0), background())
    (root / "mereni" / "reference").mkdir(parents=True)

    found = find_reference_dir(str(root / "mereni"))
    assert os.path.normpath(found) == os.path.normpath(str(real))


def test_record_reads_time_and_description_without_decompressing(workspace):
    records = list_references(workspace["reference_dir"])
    assert [r.created.hour for r in records] == [9, 14, 16]      # seřazeno podle času
    middle = records[1]
    assert middle.created == datetime(2026, 9, 8, 14, 29, 10)
    assert middle.frames == 16                                    # z popisu v JSON
    assert middle.resolution == (SHAPE[0], SHAPE[1])              # (výška, šířka)
    assert middle.json_path and os.path.isfile(middle.json_path)


def test_record_falls_back_to_filename_without_json(tmp_path):
    """Bez JSON popisu se čas musí vzít z názvu souboru."""
    path = write_reference(str(tmp_path), datetime(2026, 9, 8, 14, 29, 10),
                           background(), with_json=False)
    record = read_reference_record(path)
    assert record.json_path is None
    assert record.created == datetime(2026, 9, 8, 14, 29, 10)


def test_load_reference_returns_adu_plane(workspace):
    records = list_references(workspace["reference_dir"])
    planes = load_reference(records[1])
    assert planes.mono.dtype == np.float32
    assert planes.shape == SHAPE
    assert planes.frames == 16
    assert np.allclose(planes.mono, workspace["base"], atol=1e-3)


def test_reference_uses_the_scale_of_the_series(tmp_path):
    """Reference z 12bitové kamery se přepočítá škálou zjištěnou ze snímků.

    Bitovou hloubku nelze uhodnout z hodnot v samotné referenci – tmavý bias
    z 12bitové kamery má nízké maximum a vypadal by jako 10bitový. Proto se
    bere škála, kterou aplikace zjistila ze snímků série.
    """
    plane = background(level=20.0) * (4095.0 / 255.0)          # 12bitová škála
    path = write_reference(str(tmp_path), datetime(2026, 9, 8, 9, 0, 0), plane)
    planes = load_reference(read_reference_record(path))

    assert np.allclose(scale_to_adu(planes.mono, 4095.0), background(level=20.0), atol=0.01)

    # Pojistka: kdyby škála série neseděla, nesmí reference přetéct nad 255 ADU.
    assert scale_to_adu(planes.mono, 255.0).max() <= 255.0 + 1e-3


def test_broken_npz_raises_reference_error(tmp_path):
    path = os.path.join(str(tmp_path), "reference_20260908_140000.npz")
    with open(path, "wb") as handle:
        handle.write(b"tohle rozhodne neni npz")
    with pytest.raises(ReferenceError):
        load_reference(read_reference_record(path))


def test_npz_without_mono_plane_raises(tmp_path):
    path = os.path.join(str(tmp_path), "reference_20260908_140000.npz")
    np.savez_compressed(path, neco_jineho=np.zeros(SHAPE, dtype=np.float32))
    with pytest.raises(ReferenceError, match="referenční rovinu"):
        load_reference(read_reference_record(path))


def test_color_reference_without_mono_is_built_from_channels(tmp_path):
    """Barevná reference bez roviny ``mean_mono`` se dopočítá z kanálů."""
    path = os.path.join(str(tmp_path), "reference_20260908_140000.npz")
    plane = background()
    np.savez_compressed(
        path,
        mean_blue=(plane * 1.2).astype(np.float32),
        mean_green=plane.astype(np.float32),
        mean_red=(plane * 0.7).astype(np.float32),
        frames_mono=np.int64(8),
    )
    planes = load_reference(read_reference_record(path))
    assert planes.has_channels
    assert np.allclose(planes.mono, plane * (1.2 + 1.0 + 0.7) / 3.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Výběr podle času
# ---------------------------------------------------------------------------

def test_selects_the_last_reference_taken_before_the_series(workspace):
    """Jádro požadavku: vybírá se poslední reference PŘED začátkem měření."""
    records = list_references(workspace["reference_dir"])
    choice = select_reference(records, series_start_time(workspace["paths"]))
    assert choice.ok
    assert choice.record.created == datetime(2026, 9, 8, 14, 29, 10)
    assert not choice.taken_after
    assert 0 < choice.gap_s < 10 * 60          # necelých 6 minut před měřením


def test_later_reference_never_wins_even_if_closer_in_time(tmp_path):
    """Reference pořízená po měření je časově blíž, ale použít se nesmí."""
    ref_dir = tmp_path / "reference"
    write_reference(str(ref_dir), datetime(2026, 9, 8, 14, 0, 0), background())   # 35 min před
    write_reference(str(ref_dir), datetime(2026, 9, 8, 14, 36, 0), background())  # 1 min po

    choice = select_reference(list_references(str(ref_dir)), MEASUREMENT)
    assert choice.record.created == datetime(2026, 9, 8, 14, 0, 0)
    assert not choice.taken_after


def test_only_later_reference_is_used_but_flagged(tmp_path):
    ref_dir = tmp_path / "reference"
    write_reference(str(ref_dir), datetime(2026, 9, 8, 15, 0, 0), background())
    write_reference(str(ref_dir), datetime(2026, 9, 8, 18, 0, 0), background())

    choice = select_reference(list_references(str(ref_dir)), MEASUREMENT)
    assert choice.ok
    assert choice.taken_after
    assert choice.record.created == datetime(2026, 9, 8, 15, 0, 0)   # ta nejbližší z pozdějších


def test_stale_reference_is_reported(tmp_path):
    ref_dir = tmp_path / "reference"
    write_reference(str(ref_dir), datetime(2026, 9, 1, 8, 0, 0), background())
    choice = select_reference(list_references(str(ref_dir)), MEASUREMENT, stale_hours=6.0)
    assert choice.ok
    assert "Odstup" in choice.reason


def test_series_start_uses_the_earliest_frame(workspace):
    assert series_start_time(workspace["paths"]) == MEASUREMENT
    assert series_start_time(list(reversed(workspace["paths"]))) == MEASUREMENT


# ---------------------------------------------------------------------------
# Napojení na analýzu
# ---------------------------------------------------------------------------

def test_series_uses_reference_and_keeps_every_frame(workspace):
    """S externí referencí se nespotřebují žádné snímky série na bias."""
    params = AnalysisParams(binning=1)
    result = analyze_series(workspace["paths"], params)

    assert result.bias.is_external
    assert result.bias.frames_used == 16
    assert os.path.basename(result.bias.reference_path) == "reference_20260908_142910.npz"
    assert result.frame_count == len(workspace["paths"])     # nic neubylo
    assert not any(m.is_bias_frame for m in result.metrics)
    assert result.notes and "14:29:10" in result.notes[0]
    assert not result.warnings                                # nic k řešení


def test_missing_reference_folder_is_a_note_not_a_warning(tmp_path):
    """Bez složky s referencemi analýza normálně doběhne na biasu ze série.

    Regrese: dokud byla tato informace mezi ``warnings``, vyskočil po každé
    běžné analýze modální dialog „Upozornění k analýze“.
    """
    folder = tmp_path / "mereni"
    paths = write_series(str(folder), background())
    result = analyze_series(paths, AnalysisParams(binning=1, bias_frames=2))

    assert not result.bias.is_external
    assert not result.warnings
    assert any("bias z prvních snímků" in note for note in result.notes)


def test_strict_mode_fails_without_reference(tmp_path):
    folder = tmp_path / "mereni"
    paths = write_series(str(folder), background())
    params = AnalysisParams(binning=1, reference_mode="reference")
    with pytest.raises(ReferenceError):
        analyze_series(paths, params)


def test_series_mode_ignores_the_reference_folder(workspace):
    params = AnalysisParams(binning=1, bias_frames=2, reference_mode="serie")
    result = analyze_series(workspace["paths"], params)
    assert not result.bias.is_external
    assert result.frame_count == len(workspace["paths"]) - 2      # bias snímky vypadly


def test_explicit_reference_dir_wins(workspace, tmp_path):
    """Ručně zadaná složka má přednost před automatickým hledáním."""
    other = tmp_path / "jinde"
    write_reference(str(other), datetime(2026, 9, 8, 10, 0, 0), background() * 0.9)

    params = AnalysisParams(binning=1, reference_dir=str(other))
    choice, directory = resolve_reference(workspace["paths"], params)
    assert os.path.normpath(directory) == os.path.normpath(str(other))
    assert choice.record.created == datetime(2026, 9, 8, 10, 0, 0)


def test_reference_is_rescaled_to_series_resolution(workspace):
    """Reference v jiném rozlišení se přeškáluje a analýza to ohlásí."""
    ref_dir = workspace["root"] + os.sep + "reference"
    small = cv2.resize(workspace["base"], (SHAPE[1] // 2, SHAPE[0] // 2),
                       interpolation=cv2.INTER_AREA)
    write_reference(ref_dir, datetime(2026, 9, 8, 14, 30, 0), small)

    result = analyze_series(workspace["paths"], AnalysisParams(binning=1))
    assert result.bias.is_external
    assert result.bias.shape == SHAPE
    assert any("přeškálováno" in w for w in result.warnings)


def test_foreign_image_is_still_rejected_with_external_reference(workspace):
    """Cizí obrázek ve složce musí vypadnout i tehdy, když bias přijde odjinud."""
    intruder = os.path.join(workspace["folder"], "aaa_screenshot.png")
    imwrite_unicode(intruder, np.zeros((60, 80), dtype=np.uint8))

    paths = list_image_files(workspace["folder"])
    assert intruder in paths                       # název filtr neodchytí

    result = analyze_series(paths, AnalysisParams(binning=1))
    assert result.frame_count == len(workspace["paths"])
    assert any(os.path.basename(p) == "aaa_screenshot.png" for p, _ in result.failed_files)
    assert not result.synthetic_time_axis          # časová osa zůstala použitelná


# ---------------------------------------------------------------------------
# Srovnání úrovně reference
# ---------------------------------------------------------------------------

def test_level_offset_is_zero_for_a_matching_reference(workspace):
    result = analyze_series(workspace["paths"], AnalysisParams(binning=1))
    assert abs(result.bias.level_offset_adu) < 1.0


@pytest.mark.parametrize("shift", [-6.0, 6.0])
def test_constant_level_shift_is_removed(workspace, shift):
    """Posunutá reference by se bez srovnání projevila jako plošné zamlžení."""
    records = list_references(workspace["reference_dir"])
    record = [r for r in records if r.created.hour == 14][0]
    probe = probe_series_shape(workspace["paths"], AnalysisParams(binning=1), 255.0)

    measured = {}
    for match in (False, True):
        params = AnalysisParams(binning=1, match_reference_level=match)
        bias = bias_from_reference(record, params, probe.source_shape)
        bias.data = np.ascontiguousarray(bias.data + np.float32(shift))
        if match:
            bias.level_offset_adu = estimate_reference_offset(workspace["paths"], bias, params)
        measured[match] = analyze_series(workspace["paths"], params, bias=bias)

    assert abs(measured[True].bias.level_offset_adu + shift) < 1.5    # posun se trefil
    haze_matched = max(m.haze_coverage_pct for m in measured[True].metrics)
    assert haze_matched < 1.0                                        # čisté sklíčko zůstalo čisté
    if shift < 0:
        # Reference tmavší než snímky: bez srovnání by celá plocha byla „zamlžená“.
        assert max(m.haze_coverage_pct for m in measured[False].metrics) > 90.0


def test_real_haze_survives_the_level_correction(tmp_path):
    """Srovnání úrovně nesmí spolknout skutečné zamlžení, které v čase ustupuje."""
    root = tmp_path / "BMS fotky"
    base = background()
    write_reference(str(root / "reference"), datetime(2026, 9, 8, 14, 0, 0), base)

    folder = root / "mereni"
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(3)
    paths = []
    for i in range(6):
        fog = max(0.0, 12.0 - 3.0 * i)                # opar postupně mizí
        frame = base + fog + scene(i) + rng.normal(0.0, 0.8, SHAPE).astype(np.float32)
        stamp = MEASUREMENT + timedelta(seconds=5 * i)
        path = os.path.join(str(folder), f"df_{i + 1:05d}_{stamp:%Y%m%d_%H%M%S}_000.png")
        imwrite_unicode(path, np.clip(frame, 0, 255).astype(np.uint8))
        paths.append(path)

    result = analyze_series(paths, AnalysisParams(binning=1))
    assert result.bias.is_external
    assert result.metrics[0].haze_coverage_pct > 90.0     # opar na začátku zůstal
    assert result.metrics[-1].haze_coverage_pct < 5.0     # a na konci je pryč


# ---------------------------------------------------------------------------
# Barevné snímky
# ---------------------------------------------------------------------------

def test_mono_mode_names_are_normalized():
    assert normalize_mono_mode("PRŮMĚR") == "prumer"
    assert normalize_mono_mode("mean") == "prumer"
    assert normalize_mono_mode("blue") == "b"
    assert normalize_mono_mode(None) == "luma"
    assert normalize_mono_mode("nesmysl") == "luma"
    assert all(normalize_mono_mode(mode) == mode for mode in MONO_MODES)


def test_to_mono_projections():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 100, 50, 10      # B, G, R
    assert np.allclose(to_mono(image, "prumer"), (100 + 50 + 10) / 3.0)
    assert np.allclose(to_mono(image, "maximum"), 100.0)
    assert np.allclose(to_mono(image, "b"), 100.0)
    assert np.allclose(to_mono(image, "r"), 10.0)
    expected = 0.114 * 100 + 0.587 * 50 + 0.299 * 10
    assert np.allclose(to_mono(image, "luma"), expected, atol=1e-3)


def test_color_frame_loads_as_single_channel(tmp_path):
    path = os.path.join(str(tmp_path), "barevny.png")
    imwrite_unicode(path, np.full((10, 12, 3), 40, dtype=np.uint8))
    frame = load_frame(path)
    assert frame.data.ndim == 2
    assert frame.is_color and frame.source_channels == 3
    assert np.allclose(frame.data, 40.0, atol=1e-3)


def test_color_and_mono_series_give_the_same_result(tmp_path):
    """Barevná série musí projít stejně jako její černobílý ekvivalent."""
    root = tmp_path / "BMS fotky"
    base = background()
    write_reference(str(root / "reference"), datetime(2026, 9, 8, 14, 0, 0), base)

    mono_paths = write_series(str(root / "mono"), base, color=False)
    color_paths = write_series(str(root / "barevne"), base, color=True)

    params = AnalysisParams(binning=1)
    mono_result = analyze_series(mono_paths, params)
    color_result = analyze_series(color_paths, params)

    assert mono_result.bias.is_external and color_result.bias.is_external
    assert color_result.frame_count == mono_result.frame_count
    for a, b in zip(mono_result.metrics, color_result.metrics):
        assert a.total_coverage_pct == pytest.approx(b.total_coverage_pct, abs=0.35)
        assert abs(a.total_particle_count - b.total_particle_count) <= 2


def test_channel_mode_changes_what_is_measured(tmp_path):
    """Volba kanálu se skutečně projeví: v modrém kanálu je částice jasnější."""
    folder = tmp_path / "mereni"
    os.makedirs(folder, exist_ok=True)
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    image[..., 0] = 30                                     # modré pozadí
    cv2.circle(image, (80, 60), 4, (220, 60, 20), -1)      # modravá částice
    path = os.path.join(str(folder), "df_00001_20260908_143500_000.png")
    imwrite_unicode(path, image)

    blue = load_frame(path, mono_mode="b").data
    red = load_frame(path, mono_mode="r").data
    assert blue.max() > red.max()
    assert load_frame(path, mono_mode="maximum").data.max() >= blue.max()
