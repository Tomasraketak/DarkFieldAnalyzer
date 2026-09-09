"""Testy zarovnání snímků podle souhvězdí prachových částic.

Scéna se generuje s **předem známým posunem**, takže se dá ověřit nejen to, že
kód něco vrátí, ale i jak přesně. Součástí je regrese na dva konkrétní problémy
zjištěné při vývoji:

* horké pixely, které se nepohybují se scénou, přehlasovaly skutečné částice
  a posun vyšel nulový,
* bilineární interpolace při srovnání rozmazala snímek natolik, že kolem každé
  statické částice vznikla falešná detekce.
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

import alignment as alignment_module  # noqa: E402
from alignment import (  # noqa: E402
    FrameShift,
    detect_stars,
    estimate_shift,
    find_hot_pixels,
    intersect_roi,
    repair_hot_pixels,
    safe_crop_rect,
    warp_to_anchor,
)
from analyzer import (  # noqa: E402
    AnalysisParams,
    analyze_series,
    apply_binning,
    build_alignment,
    crop_to_roi,
)
from frameio import imwrite_unicode, list_image_files  # noqa: E402
from imageops import estimate_noise, separate_haze  # noqa: E402

H, W = 420, 640
START = datetime(2026, 9, 9, 10, 0, 0)


# ---------------------------------------------------------------------------
# Generátor scény
# ---------------------------------------------------------------------------

def _blob(canvas, cx, cy, radius, amplitude):
    y0, y1 = max(0, int(cy - 4 * radius)), min(canvas.shape[0], int(cy + 4 * radius) + 1)
    x0, x1 = max(0, int(cx - 4 * radius)), min(canvas.shape[1], int(cx + 4 * radius) + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] += amplitude * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius * radius))


def particles(count: int, seed: int = 1):
    """Pevný seznam částic (pozice, poloměr, jas)."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(50, W - 50), rng.uniform(50, H - 50),
             rng.uniform(1.8, 3.0), rng.uniform(70, 190)) for _ in range(count)]


def scene(specs, dx=0.0, dy=0.0, hot=(), noise=1.0, seed=0) -> np.ndarray:
    """Snímek s částicemi posunutými o (dx, dy); horké pixely se NEPOSOUVAJÍ."""
    img = np.full((H, W), 16.0, dtype=np.float32)
    for cx, cy, radius, amplitude in specs:
        _blob(img, cx + dx, cy + dy, radius, amplitude)
    for hx, hy in hot:
        img[hy, hx] += 210.0
    if noise:
        img += np.random.default_rng(seed).normal(0.0, noise, (H, W)).astype(np.float32)
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------------------
# Detekce částic
# ---------------------------------------------------------------------------

def test_detects_particles_and_limits_their_count():
    field = detect_stars(scene(particles(150)), star_count=40)
    assert field.count == 40                       # respektuje zadaný limit
    assert field.points.shape == (40, 2)
    assert np.all(field.flux[:-1] >= field.flux[1:])   # seřazeno od nejjasnější


def test_centroids_are_subpixel_accurate():
    """Těžiště musí sedět na zlomek pixelu – z toho žije celá přesnost posunu."""
    specs = [(200.0, 150.0, 2.5, 180.0), (400.25, 300.75, 2.5, 180.0)]
    field = detect_stars(scene(specs, noise=0.4), star_count=10)
    found = sorted((tuple(p) for p in field.points), key=lambda p: p[0])
    assert found[0] == pytest.approx((200.0, 150.0), abs=0.15)
    assert found[1] == pytest.approx((400.25, 300.75), abs=0.15)


# ---------------------------------------------------------------------------
# Odhad posunu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("truth", [(0.0, 0.0), (3.0, -2.0), (7.4, 11.2), (-15.5, 6.25)])
def test_shift_is_measured_to_a_tenth_of_a_pixel(truth):
    specs = particles(120)
    anchor = detect_stars(scene(specs))
    moved = detect_stars(scene(specs, *truth, seed=3))

    shift = estimate_shift(anchor, moved)
    assert shift.ok
    assert math.hypot(shift.dx - truth[0], shift.dy - truth[1]) < 0.1


def test_shift_survives_new_particles_appearing():
    """Kontaminace během měření přibývá – nové částice nesmí odhad rozhodit."""
    specs = particles(80)
    anchor = detect_stars(scene(specs))
    grown = detect_stars(scene(specs + particles(60, seed=99), 5.0, -4.0, seed=3))

    shift = estimate_shift(anchor, grown)
    assert shift.ok
    assert (shift.dx, shift.dy) == pytest.approx((5.0, -4.0), abs=0.2)


def test_refuses_when_there_are_too_few_particles():
    anchor = detect_stars(scene(particles(80)))
    poor = detect_stars(scene(particles(3, seed=42), 4.0, 4.0, seed=3))

    shift = estimate_shift(anchor, poor, min_matches=8)
    assert not shift.ok
    assert "částic" in shift.reason


def test_rotation_is_recovered():
    specs = particles(120)
    anchor = detect_stars(scene(specs))

    angle = 0.6
    rad = math.radians(angle)
    rotated_specs = []
    for cx, cy, radius, amplitude in specs:
        ox, oy = cx - W / 2, cy - H / 2
        rotated_specs.append((W / 2 + ox * math.cos(rad) - oy * math.sin(rad),
                              H / 2 + ox * math.sin(rad) + oy * math.cos(rad),
                              radius, amplitude))
    shift = estimate_shift(anchor, detect_stars(scene(rotated_specs, seed=3)))
    assert shift.ok
    assert shift.angle_deg == pytest.approx(-angle, abs=0.15)


# ---------------------------------------------------------------------------
# Horké pixely (regrese)
# ---------------------------------------------------------------------------

def test_hot_pixels_are_found_and_particles_are_not():
    """Maska smí obsahovat vadné pixely, ale ne skutečné částice."""
    specs = particles(40)
    hot = [(90, 70), (300, 200), (500, 330), (150, 380), (610, 90)]
    mask = find_hot_pixels(scene(specs, hot=hot))

    for hx, hy in hot:
        assert mask[hy, hx], f"horký pixel ({hx},{hy}) nebyl nalezen"
    for cx, cy, _r, _a in specs:
        assert not mask[int(round(cy)), int(round(cx))], "částice označena jako vadný pixel"


def test_hot_pixels_do_not_outvote_real_particles():
    """Regrese: bez pojistek přehlasuje 60 vadných pixelů 20 skutečných částic.

    Vadné pixely se s driftem nepohybují, takže všechny hlasují pro nulový
    posun. Když je jich víc než skutečných částic, vyhrálo by hlasování „scéna
    se nehnula“ a zarovnání by neudělalo nic.
    """
    specs = particles(20)
    rng = np.random.default_rng(11)
    hot = [(int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20))) for _ in range(60)]
    truth = (8.0, -5.0)

    anchor_img = scene(specs, hot=hot)
    moved_img = scene(specs, *truth, hot=hot, seed=3)

    # a) tvarový test vypnutý a povolené jednopixelové objekty = naivní detekce
    original_ratio = alignment_module.HOT_PEAK_RATIO
    alignment_module.HOT_PEAK_RATIO = 1e9
    try:
        naive = estimate_shift(detect_stars(anchor_img, min_area=1),
                               detect_stars(moved_img, min_area=1))
    finally:
        alignment_module.HOT_PEAK_RATIO = original_ratio

    assert naive.ok
    assert naive.magnitude < 1.0, "bez pojistky měl vyhrát nulový posun (to je ta chyba)"

    # b) s pojistkami vyhraje skutečný posun
    mask = find_hot_pixels(anchor_img)
    guarded = estimate_shift(detect_stars(anchor_img, hot_mask=mask),
                             detect_stars(moved_img, hot_mask=mask))
    assert guarded.ok
    assert (guarded.dx, guarded.dy) == pytest.approx(truth, abs=0.2)


def test_repair_replaces_only_the_masked_pixels():
    specs = particles(30)
    hot = [(100, 100), (400, 250)]
    image = scene(specs, hot=hot, noise=0.0)
    mask = find_hot_pixels(image)
    repaired = repair_hot_pixels(image, mask)

    for hx, hy in hot:
        assert repaired[hy, hx] < image[hy, hx] - 100    # špička odstraněna
    untouched = ~mask
    assert np.array_equal(repaired[untouched], image[untouched])


# ---------------------------------------------------------------------------
# Srovnání obrazu (regrese na interpolaci)
# ---------------------------------------------------------------------------

def test_warp_moves_the_image_back():
    image = scene(particles(40), noise=0.0)
    truth = (6.0, -4.0)
    moved = cv2.warpAffine(image, np.float32([[1, 0, truth[0]], [0, 1, truth[1]]]), (W, H),
                           borderMode=cv2.BORDER_REPLICATE)
    back = warp_to_anchor(moved, FrameShift(dx=truth[0], dy=truth[1], ok=True))

    inner = (slice(30, H - 30), slice(30, W - 30))
    assert np.abs(back[inner] - image[inner]).max() < np.abs(moved[inner] - image[inner]).max() / 20


def test_interpolation_does_not_invent_particles():
    """Regrese: bilineární interpolace vyrobila kolem každé částice detekci.

    Srovnaný snímek se porovnává s referencí, která interpolací neprošla –
    každé rozmazání se proto projeví jako rozdíl. Lanczos rozmazání nedělá.
    """
    specs = particles(120)
    image = scene(specs, noise=0.0)
    shift = FrameShift(dx=9.02, dy=-6.53, ok=True)

    def false_detections(warped):
        inner = (slice(40, H - 40), slice(40, W - 40))
        diff = cv2.subtract(warped[inner], image[inner])
        haze, _small = separate_haze(diff)
        sharp = cv2.subtract(diff, haze)
        median, sigma = estimate_noise(sharp)
        mask = (sharp > max(median + 4.0 * sigma, 1.5)).astype(np.uint8)
        count, _labels, stats, _c = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        return int((stats[1:, cv2.CC_STAT_AREA] >= 3).sum()) if count > 1 else 0

    # Posunutá scéna se vykreslí ANALYTICKY – skutečný drift je posun optiky,
    # ne interpolace. Interpolaci do řetězce vnáší až samotné srovnání.
    moved = scene(specs, dx=shift.dx, dy=shift.dy, noise=0.0)
    bilinear = cv2.warpAffine(moved, shift.matrix(), (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    unaligned = false_detections(moved)
    assert unaligned > 80, "bez srovnání má drift vyrábět detekce u většiny částic"
    assert false_detections(bilinear) > 20, "bilineární interpolace jich má vyrábět taky dost"

    lanczos = false_detections(warp_to_anchor(moved, shift))
    assert lanczos <= 5
    assert lanczos < unaligned / 20


# ---------------------------------------------------------------------------
# Ořez
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(2160, 3840), (1080, 1920), (720, 1280)])
def test_crop_keeps_the_aspect_ratio(shape):
    rect, fraction = safe_crop_rect(shape, drift_px=12.0)
    assert rect[2] / rect[3] == pytest.approx(shape[1] / shape[0], rel=0.005)
    assert 0.5 <= fraction <= 1.0
    assert rect[0] > 0 and rect[1] > 0                    # výřez je uprostřed
    assert rect[0] + rect[2] <= shape[1]
    assert rect[1] + rect[3] <= shape[0]


def test_auto_crop_follows_the_measured_drift():
    small, _f = safe_crop_rect((2160, 3840), drift_px=2.0)
    large, _f = safe_crop_rect((2160, 3840), drift_px=60.0)
    assert small[2] > large[2], "větší drift musí ořezat víc"

    fixed, fraction = safe_crop_rect((2160, 3840), drift_px=2.0, mode="fixed", fraction=0.90)
    assert fraction == pytest.approx(0.90)
    assert fixed[2] == pytest.approx(3840 * 0.9, abs=2)


def test_crop_excludes_the_invalid_border():
    """Do výřezu nesmí zasáhnout okraj, který po srovnání zůstal prázdný."""
    drift = 9.0
    rect, _fraction = safe_crop_rect((H, W), drift_px=drift)
    image = scene(particles(60), noise=0.0)
    warped = warp_to_anchor(image, FrameShift(dx=drift, dy=-drift, ok=True), fill=-999.0)
    assert (crop_to_roi(warped, rect, 1) > -900).all()


def test_roi_intersection():
    assert intersect_roi(None, (10, 10, 100, 100)) == (10, 10, 100, 100)
    assert intersect_roi((0, 0, 50, 50), None) == (0, 0, 50, 50)
    assert intersect_roi((0, 0, 50, 50), (25, 25, 50, 50)) == (25, 25, 25, 25)
    with pytest.raises(alignment_module.AlignmentError):
        intersect_roi((0, 0, 10, 10), (500, 500, 10, 10))


# ---------------------------------------------------------------------------
# Napojení na analýzu
# ---------------------------------------------------------------------------

def _write_drifting_series(folder, frames=10, drift=(9.0, -6.5), new_particles=12):
    """Série se statickými částicemi, driftem a přibývající kontaminací."""
    os.makedirs(folder, exist_ok=True)
    static = particles(90)
    fresh = particles(new_particles, seed=77)
    paths = []
    for index in range(frames):
        ratio = index / max(1, frames - 1)
        dx, dy = drift[0] * ratio, drift[1] * ratio
        count = int(round(new_particles * ratio))
        img = scene(static + fresh[:count], dx, dy, seed=index)
        stamp = START + timedelta(seconds=20 * index)
        path = os.path.join(folder, f"df_{index + 1:05d}_{stamp:%Y%m%d_%H%M%S}_000.png")
        imwrite_unicode(path, img.astype(np.uint8))
        paths.append(path)
    return paths


def test_alignment_model_measures_the_drift(tmp_path):
    paths = _write_drifting_series(str(tmp_path / "mereni"))
    params = AnalysisParams(binning=1)
    model = build_alignment(paths, params, (H, W), 255.0)

    assert model is not None and model.usable
    assert model.anchor.count >= 20
    assert model.crop is not None
    assert model.sampled_drift_px == pytest.approx(math.hypot(9.0, 6.5), abs=0.5)

    last = model.measure(apply_binning(
        __import__("frameio").load_frame(paths[-1]).data, 1))
    assert last.ok
    assert (last.dx, last.dy) == pytest.approx((9.0, -6.5), abs=0.3)


def test_drift_no_longer_inflates_the_particle_count(tmp_path):
    """Hlavní důvod, proč zarovnání existuje.

    Statické částice se po driftu přestanou krýt s referencí a zůstanou po nich
    světlé půlměsíce, které analýza počítá jako novou kontaminaci. Stejná série
    bez driftu slouží jako správná odpověď.
    """
    still = _write_drifting_series(str(tmp_path / "bez_driftu"), drift=(0.0, 0.0))
    moving = _write_drifting_series(str(tmp_path / "s_driftem"), drift=(9.0, -6.5))

    reference = analyze_series(still, AnalysisParams(binning=1, align_frames=False))
    broken = analyze_series(moving, AnalysisParams(binning=1, align_frames=False))
    fixed = analyze_series(moving, AnalysisParams(binning=1, align_frames=True))

    truth = reference.metrics[-1].total_particle_count
    assert broken.metrics[-1].total_particle_count > 2 * truth, "drift má škodit"
    assert fixed.metrics[-1].total_particle_count < broken.metrics[-1].total_particle_count / 2
    assert fixed.metrics[-1].total_particle_count == pytest.approx(truth, rel=0.5)
    assert fixed.alignment is not None
    assert all(m.align_ok for m in fixed.metrics)


def test_alignment_can_be_switched_off(tmp_path):
    paths = _write_drifting_series(str(tmp_path / "mereni"), frames=6)
    result = analyze_series(paths, AnalysisParams(binning=1, align_frames=False))
    assert result.alignment is None
    assert all(m.align_dx_px == 0.0 for m in result.metrics)


def test_series_without_particles_falls_back_quietly(tmp_path):
    """Na prázdném sklíčku není co sledovat – analýza musí přesto doběhnout."""
    folder = tmp_path / "prazdne"
    os.makedirs(folder, exist_ok=True)
    paths = []
    for index in range(5):
        img = np.full((H, W), 16.0, dtype=np.float32)
        img += np.random.default_rng(index).normal(0, 1.0, (H, W)).astype(np.float32)
        stamp = START + timedelta(seconds=20 * index)
        path = os.path.join(str(folder), f"df_{index + 1:05d}_{stamp:%Y%m%d_%H%M%S}_000.png")
        imwrite_unicode(path, np.clip(img, 0, 255).astype(np.uint8))
        paths.append(path)

    result = analyze_series(paths, AnalysisParams(binning=1, bias_frames=2))
    assert result.metrics                                   # analýza doběhla
    assert result.alignment is None                         # zarovnání se vzdalo
    assert any("částic" in note for note in result.notes)   # a řeklo proč


def test_crop_shrinks_the_analysed_area(tmp_path):
    paths = _write_drifting_series(str(tmp_path / "mereni"), frames=6)
    cropped = analyze_series(paths, AnalysisParams(binning=1, align_crop_mode="fixed",
                                                   align_crop_fraction=0.90))
    whole = analyze_series(paths, AnalysisParams(binning=1, align_crop_mode="none"))

    assert cropped.alignment.crop is not None
    assert whole.alignment.crop is None
    assert cropped.bias.data.shape[0] < whole.bias.data.shape[0]
    assert cropped.bias.data.shape[1] < whole.bias.data.shape[1]
    # Poměr stran zůstal zachovaný
    ratio_before = whole.bias.data.shape[1] / whole.bias.data.shape[0]
    ratio_after = cropped.bias.data.shape[1] / cropped.bias.data.shape[0]
    assert ratio_after == pytest.approx(ratio_before, rel=0.02)
