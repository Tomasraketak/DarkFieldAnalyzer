"""Jádro vědecké analýzy kontaminace witness sklíčka v temném poli.

Zpracovatelský řetězec jednoho snímku
------------------------------------
1. **Načtení a normalizace** – snímek se převede na float32 v jednotné škále
   0–255 ADU (funguje tedy stejně pro 8bit i 16bit kamery).
2. **Binning** – volitelné zmenšení (INTER_AREA, tj. průměrování 2×2 nebo 4×4).
   Na rozdíl od původního podvzorkování ``img[::2, ::2]`` se nic neztrácí
   náhodně – průměrování navíc zlepšuje poměr signál/šum.
3. **Odečet biasu** – ``diff = snímek − bias`` ve float32. Rozdíl **není**
   oříznut na nulu; záporná část je jediný nezkreslený odhad šumu pozadí.
4. **Separace oparu** – nízkofrekvenční složka (kondenzace, zamlžení) se
   odhadne na silně zmenšeném obraze morfologickým otevřením + Gaussem,
   takže ji bodové částice neznečistí. ``sharp = diff − haze``.
5. **Prahování** – práh ``medián + k·σ`` z robustního odhadu (MAD) přímo ve
   float32, bez zaokrouhlování na celé ADU.
6. **Segmentace a klasifikace** – ``connectedComponentsWithStats`` (CV_32S) a
   plně vektorizovaná klasifikace přes momenty druhého řádu. Protáhlost se
   počítá z ekvivalentní elipsy, ne z opsaného obdélníku – šikmé vlákno pod
   45° tak už není klasifikováno jako shluk.

Oproti původní verzi jsou opraveny tři příčiny pádů:

* ``ltype=cv2.CV_16U`` v ``connectedComponentsWithStats`` vyhodí výjimku,
  jakmile snímek obsahuje více než 65 535 objektů (u zašuměného 4K snímku
  zcela běžné). Nyní se používá ``CV_32S``.
* Nesoulad rozměrů snímku a biasu (``cv2.subtract`` vyhodí výjimku) je
  ošetřen automatickým přeškálováním biasu.
* Nenačtený snímek už není tiše přeskočen – je nahlášen do seznamu chyb.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from frameio import (
    FrameReadError,
    build_time_axis,
    load_frame,
    parse_timestamp_from_filename,
    probe_full_scale,
)

#: Nad tento počet kontaminovaných pixelů se přeskočí výpočet momentů
#: (ochrana paměti u extrémně zašuměných sérií) a použije se opsaný obdélník.
MOMENT_PIXEL_LIMIT = 6_000_000

#: Cílová výška obrazu pro automatickou volbu binningu.
AUTO_BINNING_TARGET_HEIGHT = 1200


# ---------------------------------------------------------------------------
# Parametry
# ---------------------------------------------------------------------------

@dataclass
class AnalysisParams:
    """Konfigurace analýzy. Všechny prahy jsou v ADU na škále 0–255."""

    bias_frames: int = 3               # Počet snímků pro referenční pozadí
    bias_method: str = "median"        # "median" (odolný vůči částici) nebo "mean"
    threshold_mode: str = "sigma"      # "sigma" nebo "absolute"
    sigma: float = 4.0                 # Násobek směrodatné odchylky šumu
    absolute_threshold: float = 12.0   # Absolutní práh v ADU nad bias
    min_threshold_adu: float = 1.5     # Dolní mez prahu (pod ní nemá 8bit signál smysl)
    haze_threshold: float = 4.0        # Práh difuzního zamlžení (ADU nad bias)
    min_area_px: int = 3               # Minimální plocha objektu (v px plného rozlišení)
    cluster_min_area_px: int = 100     # Hranice velkého shluku (v px plného rozlišení)
    fiber_aspect_ratio: float = 2.8    # Minimální protáhlost pro vlákno/škrábanec
    fiber_min_length_px: int = 12      # Minimální délka vlákna (v px plného rozlišení)
    saturation_adu: float = 250.0      # Hranice hotspotu / přesyceného pixelu
    um_per_px: float = 1.0             # Kalibrace mikroskopu (0 = nekalibrováno)
    binning: int = 0                   # 0 = auto, jinak 1 / 2 / 4
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) v plném rozlišení
    workers: int = 0                   # 0 = auto (počet vláken pro sérii)
    assumed_fps: float = 1.0           # Náhradní osa, pokud snímky nemají čas
    exclude_bias_from_series: bool = True   # Bias snímky nemají vlastní referenci – viz README

    def resolve_binning(self, image_height: int) -> int:
        """Vrátí skutečný binning – při ``binning=0`` ho odvodí z rozlišení."""
        if self.binning and self.binning > 0:
            return max(1, int(self.binning))
        factor = 1
        while image_height // factor > AUTO_BINNING_TARGET_HEIGHT and factor < 4:
            factor *= 2
        return factor

    @property
    def resolution_label(self) -> str:
        return {0: "auto", 1: "plné rozlišení", 2: "1/2 (binning 2×2)", 4: "1/4 (binning 4×4)"}.get(
            int(self.binning), f"binning {self.binning}×{self.binning}"
        )


@dataclass
class BiasModel:
    """Referenční pozadí (master bias) připravené v geometrii analýzy."""

    data: np.ndarray            # float32, po binningu a ořezu ROI
    frames_used: int
    binning: int
    full_scale: float
    source_shape: Tuple[int, int]  # rozměr původního (nebinovaného) snímku

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Výsledné metriky
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    """Kompletní metriky jednoho snímku."""

    index: int
    filename: str
    filepath: str
    timestamp: datetime
    time_s: float

    # Celkové znečištění
    total_coverage_pct: float
    total_area_px: int                 # přepočteno na px plného rozlišení
    total_area_um2: float
    integrated_signal_adu: float
    mean_signal_adu: float
    max_signal_adu: float
    bg_level_adu: float                # medián diference (drift osvětlení)
    bg_noise_sigma: float
    applied_threshold: float
    snr: float                         # průměrný signál / šum pozadí

    # Difuzní zamlžení / kondenzace
    haze_coverage_pct: float           # celá plocha oparu (i tam, kde leží částice)
    haze_only_coverage_pct: float      # plocha oparu bez částic – složky se sčítají na celkové pokrytí
    haze_mean_adu: float
    haze_max_adu: float

    # Drobné částice
    point_count: int
    point_area_px: int
    point_area_pct: float

    # Velké shluky
    cluster_count: int
    cluster_area_px: int
    cluster_area_pct: float

    # Vlákna a škrábance
    fiber_count: int
    fiber_area_px: int
    fiber_area_pct: float
    fiber_total_length_px: float

    # Souhrn částic
    total_particle_count: int
    particle_density_per_mpx: float     # částic na megapixel
    mean_particle_area_px: float
    median_particle_area_px: float
    p90_particle_area_px: float
    max_particle_area_px: int
    mean_particle_diameter_um: float    # ekvivalentní průměr (0 = nekalibrováno)

    # Optická kvalita a rozložení
    saturated_pixels_count: int
    spatial_heterogeneity_pct: float
    centroid_x_pct: float
    centroid_y_pct: float
    focus_score: float                  # variance Laplaciánu (ostrost/rozostření)
    cleanliness_score: float

    is_bias_frame: bool = False
    note: str = ""

    # Dopočítává se přes celou sérii
    rate_coverage_pct_per_s: float = 0.0
    rate_haze_pct_per_s: float = 0.0
    rate_signal_adu_per_s: float = 0.0
    rate_particles_per_s: float = 0.0
    phase: str = "stabilní"


@dataclass
class SeriesResult:
    """Výsledek analýzy celé série snímků."""

    metrics: List[FrameMetrics] = field(default_factory=list)
    bias: Optional[BiasModel] = None
    params: Optional[AnalysisParams] = None
    elapsed_s: float = 0.0
    failed_files: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    synthetic_time_axis: bool = False
    cancelled: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.metrics)


# ---------------------------------------------------------------------------
# Geometrie: binning, ROI, sladění biasu
# ---------------------------------------------------------------------------

def apply_geometry(image: np.ndarray, binning: int, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Aplikuje binning (INTER_AREA) a ořez ROI na float32 snímek."""
    out = image
    if binning > 1:
        h, w = out.shape[:2]
        new_w = max(1, w // binning)
        new_h = max(1, h // binning)
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if roi is not None:
        rx, ry, rw, rh = (int(v) for v in roi)
        if binning > 1:
            rx, ry, rw, rh = rx // binning, ry // binning, rw // binning, rh // binning
        h, w = out.shape[:2]
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))
        out = out[ry : ry + rh, rx : rx + rw]

    return np.ascontiguousarray(out, dtype=np.float32)


def match_shape(reference: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Přizpůsobí bias tvaru snímku.

    Pojistka proti pádu ``cv2.subtract``/``numpy`` při nesouladu rozměrů
    (např. když prohlížeč zobrazuje náhled v jiném zmenšení, než v jakém
    proběhla analýza).
    """
    if reference.shape[:2] == tuple(shape):
        return reference
    return cv2.resize(reference, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

def compute_bias(
    image_paths: Sequence[str],
    params: AnalysisParams,
    full_scale: Optional[float] = None,
) -> BiasModel:
    """Sestaví master bias z prvních N snímků.

    Výchozí metoda je **medián** – jediná náhodná částice, která se v bias
    snímku objeví, tak neznehodnotí referenci pro celé měření (u průměru by
    se propsala do všech následujících snímků jako záporný artefakt).
    """
    if not image_paths:
        raise ValueError("Seznam souborů pro výpočet biasu je prázdný.")

    n = max(1, min(int(params.bias_frames), len(image_paths)))
    if full_scale is None:
        full_scale = probe_full_scale(image_paths, sample=min(3, n))

    first = load_frame(image_paths[0], full_scale=full_scale)
    source_shape = first.data.shape
    binning = params.resolve_binning(source_shape[0])

    prepared = [apply_geometry(first.data, binning, params.roi)]
    for path in image_paths[1:n]:
        try:
            frame = load_frame(path, full_scale=full_scale)
        except FrameReadError:
            continue
        if frame.data.shape != source_shape:
            continue
        prepared.append(apply_geometry(frame.data, binning, params.roi))

    if len(prepared) == 1:
        bias = prepared[0]
    elif params.bias_method == "mean" or len(prepared) > 8:
        # U mnoha snímků je průměr paměťově výhodnější než medián.
        acc = np.zeros_like(prepared[0], dtype=np.float32)
        for item in prepared:
            acc += item
        bias = acc / float(len(prepared))
    else:
        bias = np.median(np.stack(prepared, axis=0), axis=0).astype(np.float32)

    return BiasModel(
        data=np.ascontiguousarray(bias, dtype=np.float32),
        frames_used=len(prepared),
        binning=binning,
        full_scale=float(full_scale),
        source_shape=source_shape,
    )


# ---------------------------------------------------------------------------
# Statistika pozadí a separace oparu
# ---------------------------------------------------------------------------

def estimate_noise(image: np.ndarray, sample_limit: int = 250_000) -> Tuple[float, float]:
    """Robustní odhad (medián, σ) šumu pozadí na podvzorku.

    Šum se odhaduje z **rozdílů sousedních pixelů** (MAD × 1.4826 / √2). Tento
    vysokofrekvenční odhad má oproti odhadu z celkového rozdělení dvě zásadní
    výhody:

    * nezkreslí ho struktura scény – kontaminace je prostorově souvislá,
      zatímco šum se mění od pixelu k pixelu;
    * funguje i u snímků, které samy vstoupily do výpočtu biasu. U mediánového
      biasu je u nich přes polovinu pixelů rozdílu přesně nulová, klasický MAD
      vyjde téměř nulový, práh spadne pod úroveň šumu a analýza „najde“
      desítky tisíc neexistujících částic.

    Teprve když je i tento odhad degenerovaný (dokonale hladké pozadí), sáhne
    se po MAD a šířce dolní poloviny rozdělení.

    Odhad se počítá z *neořezané* diference včetně záporné části – původní
    verze měřila šum až po saturačním odečtu v uint8, kde je polovina
    rozdělení uříznutá, a šum tím systematicky podhodnocovala.
    """
    if image.size == 0:
        return 0.0, 1.0

    step = max(1, int(math.sqrt(image.size / max(1, sample_limit))))
    patch = image[::step, ::step].astype(np.float32, copy=False)
    sample = patch.ravel()
    if sample.size == 0:
        return 0.0, 1.0

    median = float(np.median(sample))

    # Primární odhad: rozdíly sousedních pixelů (vysokofrekvenční složka).
    # Měří skutečný šum senzoru, nikoliv strukturu scény – na reálném snímku
    # z temného pole je totiž velká část plochy pokrytá texturou (zaschlý film,
    # rozostřené halo kolem kapek) a odhad z celkového rozdělení by tuto
    # strukturu započítal jako „šum“, práh by vyletěl a jemné částice by zmizely.
    sigma = 0.0
    if patch.ndim == 2 and patch.shape[1] > 1:
        deltas = (patch[:, 1:] - patch[:, :-1]).ravel()
        sigma = float(np.median(np.abs(deltas - np.median(deltas)))) * 1.4826 / math.sqrt(2.0)

    if not np.isfinite(sigma) or sigma <= 0.25:
        # Záložní odhady pro degenerované případy (dokonale hladké pozadí).
        mad_sigma = float(np.median(np.abs(sample - median))) * 1.4826
        lower_sigma = median - float(np.percentile(sample, 15.87))
        sigma = max(mad_sigma, lower_sigma)
    if not np.isfinite(sigma) or sigma <= 0.25:
        # Naprosto ploché pozadí (typicky dokonale černé 8bit pole). Nepočítá se
        # náhradní odhad ze směrodatné odchylky celého snímku – ta zahrnuje i
        # samotnou kontaminaci a práh by vyšel tak vysoko, že by se nenašlo nic.
        # Práh v takovém případě určuje mez ``min_threshold_adu``.
        sigma = 0.25
    return median, sigma


def separate_haze(diff: np.ndarray, downscale: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Rozdělí diferenci na nízkofrekvenční opar a jeho zmenšenou verzi.

    Postup: zmenšení (INTER_AREA) → morfologické otevření (odstraní bodové
    částice, aby nezvyšovaly odhad oparu) → Gaussovo rozostření → zpět na
    plné rozlišení. Vrací ``(haze_full, haze_small)``.
    """
    h, w = diff.shape[:2]
    small_w = max(16, w // downscale)
    small_h = max(16, h // downscale)

    small = cv2.resize(diff, (small_w, small_h), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=2.0, sigmaY=2.0)

    haze_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return haze_full, small


# ---------------------------------------------------------------------------
# Segmentace a klasifikace objektů
# ---------------------------------------------------------------------------

@dataclass
class ParticleStats:
    """Výsledek klasifikace objektů v jednom snímku."""

    category_lut: np.ndarray           # uint8 pro každý label: 0/1/2/3
    point_count: int = 0
    point_area_px: int = 0
    cluster_count: int = 0
    cluster_area_px: int = 0
    fiber_count: int = 0
    fiber_area_px: int = 0
    fiber_total_length_px: float = 0.0
    areas: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


def _shape_descriptors(
    labels: np.ndarray,
    n_labels: int,
    stats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vrátí (protáhlost, délku hlavní osy) pro každý label.

    Počítá se z momentů druhého řádu (ekvivalentní elipsa), takže hodnota
    nezávisí na natočení objektu. U extrémně zašuměných snímků se přepne na
    levnější odhad z opsaného obdélníku, aby nedošlo k vyčerpání paměti.
    """
    n = n_labels
    elongation = np.ones(n, dtype=np.float64)
    major_axis = np.zeros(n, dtype=np.float64)

    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)

    flat = labels.ravel()
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return elongation, major_axis

    if nz.size > MOMENT_PIXEL_LIMIT:
        max_dim = np.maximum(widths, heights)
        min_dim = np.maximum(np.minimum(widths, heights), 1.0)
        return max_dim / min_dim, max_dim

    lab = flat[nz].astype(np.int64, copy=False)
    width = labels.shape[1]
    ys, xs = np.divmod(nz, width)
    ys = ys.astype(np.float64, copy=False)
    xs = xs.astype(np.float64, copy=False)

    counts = np.bincount(lab, minlength=n).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    sx = np.bincount(lab, weights=xs, minlength=n)
    sy = np.bincount(lab, weights=ys, minlength=n)
    sxx = np.bincount(lab, weights=xs * xs, minlength=n)
    syy = np.bincount(lab, weights=ys * ys, minlength=n)
    sxy = np.bincount(lab, weights=xs * ys, minlength=n)

    mx = sx / counts_safe
    my = sy / counts_safe
    # +1/12 = rozptyl rovnoměrného rozdělení uvnitř pixelu (diskrétní korekce)
    cxx = sxx / counts_safe - mx * mx + 1.0 / 12.0
    cyy = syy / counts_safe - my * my + 1.0 / 12.0
    cxy = sxy / counts_safe - mx * my

    tmp = np.sqrt(np.maximum(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    lam_major = 0.5 * (cxx + cyy + tmp)
    lam_minor = np.maximum(0.0, 0.5 * (cxx + cyy - tmp))

    major_axis = 4.0 * np.sqrt(np.maximum(lam_major, 0.0))
    minor_axis = 4.0 * np.sqrt(lam_minor)
    elongation = major_axis / np.maximum(minor_axis, 1.0)

    return elongation, major_axis


def classify_components(
    labels: np.ndarray,
    n_labels: int,
    stats: np.ndarray,
    params: AnalysisParams,
    binning: int,
) -> ParticleStats:
    """Vektorizovaně roztřídí objekty na částice / shluky / vlákna."""
    lut = np.zeros(max(n_labels, 1), dtype=np.uint8)
    result = ParticleStats(category_lut=lut)
    if n_labels <= 1:
        return result

    area_factor = 1.0 / float(binning * binning)     # px plného rozlišení → px po binningu
    length_factor = 1.0 / float(binning)
    min_area = max(1.0, params.min_area_px * area_factor)
    cluster_min = max(min_area + 1.0, params.cluster_min_area_px * area_factor)
    min_fiber_len = max(3.0, params.fiber_min_length_px * length_factor)
    aspect_limit = max(1.2, float(params.fiber_aspect_ratio))

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    elongation, major_axis = _shape_descriptors(labels, n_labels, stats)

    valid = areas >= min_area
    valid[0] = False  # pozadí

    is_fiber = valid & (elongation >= aspect_limit) & (major_axis >= min_fiber_len)
    is_cluster = valid & ~is_fiber & (areas >= cluster_min)
    is_point = valid & ~is_fiber & ~is_cluster

    lut[is_point] = 1
    lut[is_cluster] = 2
    lut[is_fiber] = 3

    result.point_count = int(is_point.sum())
    result.point_area_px = int(areas[is_point].sum())
    result.cluster_count = int(is_cluster.sum())
    result.cluster_area_px = int(areas[is_cluster].sum())
    result.fiber_count = int(is_fiber.sum())
    result.fiber_area_px = int(areas[is_fiber].sum())
    result.fiber_total_length_px = float(major_axis[is_fiber].sum())
    result.areas = areas[valid]
    return result


# ---------------------------------------------------------------------------
# Skóre čistoty
# ---------------------------------------------------------------------------

def compute_cleanliness_score(
    coverage_pct: float,
    haze_mean_adu: float,
    particle_density_per_mpx: float,
) -> float:
    """Souhrnný index čistoty 0–100 % (100 % = dokonale čisté sklíčko).

    Každá složka je saturující exponenciála, takže skóre je spojité, monotónní
    a nikdy neskočí skokem na nulu. Hustota částic se počítá na megapixel, aby
    hodnota nezávisela na rozlišení kamery ani na zvoleném binningu.

    Rozpočet penalizací: pokrytí 45 b., opar 30 b., hustota částic 25 b.
    """
    p_cov = 45.0 * (1.0 - math.exp(-max(0.0, coverage_pct) / 2.0))
    p_haze = 30.0 * (1.0 - math.exp(-max(0.0, haze_mean_adu) / 8.0))
    p_part = 25.0 * (1.0 - math.exp(-max(0.0, particle_density_per_mpx) / 120.0))
    return float(max(0.0, min(100.0, round(100.0 - (p_cov + p_haze + p_part), 1))))


# ---------------------------------------------------------------------------
# Analýza jednoho snímku
# ---------------------------------------------------------------------------

def analyze_prepared_frame(
    image: np.ndarray,
    bias: np.ndarray,
    params: AnalysisParams,
    index: int,
    filename: str,
    filepath: str,
    timestamp: datetime,
    t0: datetime,
    binning: int,
    generate_masks: bool = False,
) -> Tuple[FrameMetrics, Optional[Dict[str, np.ndarray]]]:
    """Analyzuje snímek, který je již ve float32 ADU a v geometrii analýzy."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    bias = match_shape(np.asarray(bias, dtype=np.float32), image.shape[:2])

    h, w = image.shape[:2]
    total_pixels = float(h * w)
    px_scale = float(binning * binning)        # px po binningu → px plného rozlišení

    diff = cv2.subtract(image, bias)
    bg_level, bg_sigma = estimate_noise(diff)

    haze_full, haze_small = separate_haze(diff)
    sharp = cv2.subtract(diff, haze_full)

    # --- práh pro ostrou složku -------------------------------------------
    sharp_median, sharp_sigma = estimate_noise(sharp)
    if params.threshold_mode == "absolute":
        applied_threshold = float(params.absolute_threshold)
    else:
        applied_threshold = sharp_median + float(params.sigma) * sharp_sigma
    applied_threshold = max(applied_threshold, float(params.min_threshold_adu), 0.5)

    mask_sharp = (sharp > applied_threshold).astype(np.uint8)

    # --- segmentace (CV_32S: bez přetečení počtu labelů) -------------------
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_sharp, connectivity=8, ltype=cv2.CV_32S
    )
    particles = classify_components(labels, n_labels, stats, params, binning)

    classified = particles.category_lut[labels]        # uint8, 0–3
    mask_particles = classified > 0

    # --- opar --------------------------------------------------------------
    haze_thr = float(params.haze_threshold)
    haze_small_mask = haze_small > haze_thr
    haze_pixels = int(haze_small_mask.sum())
    haze_coverage_pct = 100.0 * haze_pixels / float(haze_small.size) if haze_small.size else 0.0
    haze_mean_adu = float(haze_small[haze_small_mask].mean()) if haze_pixels else 0.0
    haze_max_adu = float(haze_small.max()) if haze_small.size else 0.0

    mask_haze = haze_full > haze_thr
    mask_total = mask_particles | mask_haze
    total_area_px_binned = int(mask_total.sum())
    total_coverage_pct = 100.0 * total_area_px_binned / total_pixels if total_pixels else 0.0

    # Plocha oparu bez částic. Masky se překrývají (částice leží i uvnitř oparu),
    # takže pro grafy složení je potřeba disjunktní rozklad: součet
    # „opar bez částic + mikročástice + shluky + vlákna“ dá přesně celkové pokrytí.
    particle_area_binned = (
        particles.point_area_px + particles.cluster_area_px + particles.fiber_area_px
    )
    haze_only_px = max(0, total_area_px_binned - particle_area_binned)
    haze_only_coverage_pct = 100.0 * haze_only_px / total_pixels if total_pixels else 0.0

    # --- signál ------------------------------------------------------------
    if total_area_px_binned:
        integrated = float(np.sum(diff, where=mask_total, dtype=np.float64))
        mean_signal = integrated / total_area_px_binned
    else:
        integrated = 0.0
        mean_signal = 0.0
    max_signal = float(diff.max()) if diff.size else 0.0
    snr = mean_signal / bg_sigma if bg_sigma > 0 else 0.0

    # --- rozložení v ploše -------------------------------------------------
    heterogeneity, cx_pct, cy_pct = spatial_distribution(mask_total)

    # --- optická kvalita ---------------------------------------------------
    saturated = int((image >= float(params.saturation_adu)).sum())
    focus = focus_score(image)

    # --- statistika velikosti částic ---------------------------------------
    areas_full = particles.areas * px_scale
    particle_count = int(areas_full.size)
    if particle_count:
        mean_area = float(areas_full.mean())
        median_area = float(np.median(areas_full))
        p90_area = float(np.percentile(areas_full, 90))
        max_area = int(areas_full.max())
        mean_diameter_px = float(np.mean(2.0 * np.sqrt(areas_full / math.pi)))
    else:
        mean_area = median_area = p90_area = 0.0
        max_area = 0
        mean_diameter_px = 0.0

    scale_um = float(params.um_per_px)
    mean_diameter_um = mean_diameter_px * scale_um if scale_um > 0 else 0.0

    megapixels = total_pixels * px_scale / 1e6
    density = particle_count / megapixels if megapixels > 0 else 0.0
    cleanliness = compute_cleanliness_score(total_coverage_pct, haze_mean_adu, density)

    total_area_px_full = int(round(total_area_px_binned * px_scale))
    total_area_um2 = total_area_px_full * scale_um * scale_um if scale_um > 0 else 0.0

    metrics = FrameMetrics(
        index=index,
        filename=filename,
        filepath=filepath,
        timestamp=timestamp,
        time_s=(timestamp - t0).total_seconds(),
        total_coverage_pct=total_coverage_pct,
        total_area_px=total_area_px_full,
        total_area_um2=total_area_um2,
        integrated_signal_adu=integrated * px_scale,
        mean_signal_adu=mean_signal,
        max_signal_adu=max_signal,
        bg_level_adu=bg_level,
        bg_noise_sigma=bg_sigma,
        applied_threshold=applied_threshold,
        snr=snr,
        haze_coverage_pct=haze_coverage_pct,
        haze_only_coverage_pct=haze_only_coverage_pct,
        haze_mean_adu=haze_mean_adu,
        haze_max_adu=haze_max_adu,
        point_count=particles.point_count,
        point_area_px=int(particles.point_area_px * px_scale),
        point_area_pct=100.0 * particles.point_area_px / total_pixels if total_pixels else 0.0,
        cluster_count=particles.cluster_count,
        cluster_area_px=int(particles.cluster_area_px * px_scale),
        cluster_area_pct=100.0 * particles.cluster_area_px / total_pixels if total_pixels else 0.0,
        fiber_count=particles.fiber_count,
        fiber_area_px=int(particles.fiber_area_px * px_scale),
        fiber_area_pct=100.0 * particles.fiber_area_px / total_pixels if total_pixels else 0.0,
        fiber_total_length_px=particles.fiber_total_length_px * binning,
        total_particle_count=particle_count,
        particle_density_per_mpx=density,
        mean_particle_area_px=mean_area,
        median_particle_area_px=median_area,
        p90_particle_area_px=p90_area,
        max_particle_area_px=max_area,
        mean_particle_diameter_um=mean_diameter_um,
        saturated_pixels_count=int(round(saturated * px_scale)),
        spatial_heterogeneity_pct=heterogeneity,
        centroid_x_pct=cx_pct,
        centroid_y_pct=cy_pct,
        focus_score=focus,
        cleanliness_score=cleanliness,
    )

    masks: Optional[Dict[str, np.ndarray]] = None
    if generate_masks:
        masks = {
            "diff": diff,
            "haze": haze_full,
            "sharp": sharp,
            "mask_haze": mask_haze.astype(np.uint8) * 255,
            "mask_points": (classified == 1).astype(np.uint8) * 255,
            "mask_clusters": (classified == 2).astype(np.uint8) * 255,
            "mask_fibers": (classified == 3).astype(np.uint8) * 255,
            "mask_total": mask_total.astype(np.uint8) * 255,
        }
    else:
        # Velká mezipole už nejsou potřeba – uvolníme je dřív, než se načte
        # další snímek (u 4K jde o desítky MB na snímek).
        del labels, classified, mask_sharp, sharp, haze_full, diff
        del mask_total, mask_particles, mask_haze

    return metrics, masks


def analyze_frame_file(
    path: str,
    bias: BiasModel,
    params: AnalysisParams,
    index: int,
    timestamp: datetime,
    t0: datetime,
    generate_masks: bool = False,
) -> Tuple[FrameMetrics, Optional[Dict[str, np.ndarray]]]:
    """Načte snímek ze souboru a zanalyzuje ho."""
    frame = load_frame(path, full_scale=bias.full_scale)
    prepared = apply_geometry(frame.data, bias.binning, params.roi)
    return analyze_prepared_frame(
        image=prepared,
        bias=bias.data,
        params=params,
        index=index,
        filename=os.path.basename(path),
        filepath=path,
        timestamp=timestamp,
        t0=t0,
        binning=bias.binning,
        generate_masks=generate_masks,
    )


# ---------------------------------------------------------------------------
# Pomocné metriky
# ---------------------------------------------------------------------------

def spatial_distribution(mask: np.ndarray, zones: int = 8) -> Tuple[float, float, float]:
    """Index nehomogenity (0–100 %) a těžiště kontaminace (0–100 %).

    Nehomogenita je variační koeficient pokrytí v mřížce ``zones × zones``
    normovaný svým teoretickým maximem ``sqrt(N−1)`` (veškerá kontaminace
    v jediné zóně) – hodnota je tak skutečně v rozsahu 0–100 %.
    """
    mask_u8 = mask.astype(np.uint8, copy=False) if mask.dtype != np.uint8 else mask
    h, w = mask_u8.shape[:2]

    moments = cv2.moments(mask_u8, binaryImage=True)
    if moments["m00"] <= 0:
        return 0.0, 50.0, 50.0

    cx_pct = float(moments["m10"] / moments["m00"] / max(w, 1) * 100.0)
    cy_pct = float(moments["m01"] / moments["m00"] / max(h, 1) * 100.0)

    grid = cv2.resize(
        (mask_u8 > 0).astype(np.float32), (zones, zones), interpolation=cv2.INTER_AREA
    )
    mean_zone = float(grid.mean())
    if mean_zone <= 1e-9:
        return 0.0, cx_pct, cy_pct

    cv_score = float(grid.std()) / mean_zone
    max_cv = math.sqrt(zones * zones - 1)
    heterogeneity = 100.0 * min(1.0, cv_score / max_cv)
    return heterogeneity, cx_pct, cy_pct


def focus_score(image: np.ndarray, max_width: int = 960) -> float:
    """Ostrost obrazu jako variance Laplaciánu (detekce rozostření/vibrací)."""
    h, w = image.shape[:2]
    if w > max_width:
        step = max(1, w // max_width)
        small = image[::step, ::step]
    else:
        small = image
    lap = cv2.Laplacian(small, cv2.CV_32F, ksize=3)
    return float(lap.var())


# ---------------------------------------------------------------------------
# Rychlosti změn a fáze děje
# ---------------------------------------------------------------------------

def _local_slope(times: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """Derivace z lokální lineární regrese (Savitzky–Golay 1. řádu).

    Odolnější než diference sousedních snímků, které u vysokých snímkových
    frekvencí zesilují šum (dělí se velmi malým dt).
    """
    n = len(times)
    slopes = np.zeros(n, dtype=np.float64)
    if n < 2:
        return slopes

    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        t = times[lo:hi]
        v = values[lo:hi]
        t_mean = t.mean()
        denom = float(((t - t_mean) ** 2).sum())
        if denom <= 1e-12:
            slopes[i] = 0.0
        else:
            slopes[i] = float(((t - t_mean) * (v - v.mean())).sum() / denom)
    return slopes


def compute_rates_and_phases(
    metrics_list: List[FrameMetrics], window: int = 5
) -> List[FrameMetrics]:
    """Dopočítá časové derivace a klasifikuje fáze děje.

    Prahy pro klasifikaci fáze se odvozují z rozptylu samotných rychlostí
    (robustní MAD), takže fungují stejně dobře pro pomalé usazování prachu
    i pro prudké zapaření dechem.
    """
    n = len(metrics_list)
    if n == 0:
        return metrics_list
    if n == 1:
        metrics_list[0].phase = "výchozí"
        return metrics_list

    times = np.array([m.time_s for m in metrics_list], dtype=np.float64)
    # Ochrana proti nulovému nebo klesajícímu dt (stejná razítka u rychlé série)
    for i in range(1, n):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 1e-3

    coverage = np.array([m.total_coverage_pct for m in metrics_list], dtype=np.float64)
    haze = np.array([m.haze_coverage_pct for m in metrics_list], dtype=np.float64)
    signal = np.array([m.mean_signal_adu for m in metrics_list], dtype=np.float64)
    counts = np.array([m.total_particle_count for m in metrics_list], dtype=np.float64)

    d_cov = _local_slope(times, coverage, window)
    d_haze = _local_slope(times, haze, window)
    d_sig = _local_slope(times, signal, window)
    d_cnt = _local_slope(times, counts, window)

    cov_thresh = _phase_threshold(d_cov, floor=0.02)
    haze_thresh = _phase_threshold(d_haze, floor=0.02)

    for i, m in enumerate(metrics_list):
        m.rate_coverage_pct_per_s = float(d_cov[i])
        m.rate_haze_pct_per_s = float(d_haze[i])
        m.rate_signal_adu_per_s = float(d_sig[i])
        m.rate_particles_per_s = float(d_cnt[i])
        m.phase = _classify_phase(d_cov[i], d_haze[i], cov_thresh, haze_thresh)

    return metrics_list


def _classify_phase(d_cov: float, d_haze: float, cov_thresh: float, haze_thresh: float) -> str:
    """Určí fázi děje z rychlosti pokrytí, případně z rychlosti zamlžení.

    Dvě opravy proti původní verzi:

    * Původní podmínka ``if d_cov > práh or d_signál > práh`` označila klesající
      pokrytí za nárůst, jakmile zároveň rostl průměrný jas. Znaménko teď určuje
      vždy jen jedna veličina.
    * Jako doplňkové kritérium slouží zamlžení, nikoliv průměrný jas. Průměrný jas
      se totiž počítá jen přes kontaminované pixely – ve chvíli, kdy opar zmizí a
      zůstanou jen jasné částice, mechanicky vyskočí nahoru, i když kontaminace
      ubývá.
    """
    if abs(d_cov) > cov_thresh:
        return "nárůst / zamlžování" if d_cov > 0 else "odpařování / ústup"
    if abs(d_haze) > haze_thresh:
        return "nárůst / zamlžování" if d_haze > 0 else "odpařování / ústup"
    return "stabilní"


def _phase_threshold(rates: np.ndarray, floor: float, fraction: float = 0.05) -> float:
    """Práh „už je to změna“ pro klasifikaci fáze.

    Odvozuje se z dynamiky konkrétního měření (5 % 95. percentilu rychlosti),
    nikdy však neklesne pod pevnou dolní mez. Pevný práh by u pomalého usazování
    prachu neoznačil nic a u prudkého zapaření dechem naopak úplně vše.
    """
    if rates.size == 0:
        return floor
    peak = float(np.percentile(np.abs(rates), 95))
    return max(floor, fraction * peak)


def _robust_scale(values: np.ndarray) -> float:
    """Robustní odhad rozptylu (MAD × 1.4826)."""
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return mad * 1.4826


# ---------------------------------------------------------------------------
# Analýza celé série
# ---------------------------------------------------------------------------

#: Odhad paměti na jeden megapixel zpracovávaného snímku [MB].
MEMORY_PER_MPX_MB = 30.0

#: Strop celkové paměti pro paralelní zpracování [MB].
MEMORY_BUDGET_MB = 800.0


def resolve_workers(params: AnalysisParams, megapixels: float = 0.0) -> int:
    """Určí počet paralelních vláken (0 = auto).

    Kromě počtu jader se hlídá i paměť: u 4K v plném rozlišení zabere jeden
    snímek zhruba 250 MB mezivýsledků, takže se počet vláken automaticky sníží,
    aby analýza nevytlačila notebook do odkládacího souboru.
    """
    cpus = os.cpu_count() or 2
    requested = int(params.workers) if params.workers and params.workers > 0 else max(1, min(4, cpus - 1))

    if megapixels > 0:
        per_worker = max(1.0, megapixels * MEMORY_PER_MPX_MB)
        allowed = int(MEMORY_BUDGET_MB // per_worker)
        requested = max(1, min(requested, max(1, allowed)))
    return requested


def analyze_series(
    image_paths: Sequence[str],
    params: AnalysisParams,
    bias: Optional[BiasModel] = None,
    progress: Optional[Callable[[int, int, FrameMetrics], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> SeriesResult:
    """Zanalyzuje celou sérii snímků.

    Snímky se zpracovávají v malém okně paralelních úloh – čtení z disku se
    tak překrývá s výpočtem, ale v paměti nikdy není víc než několik snímků.
    Chyba u jednoho souboru sérii nezastaví; skončí v ``failed_files``.
    """
    result = SeriesResult(params=params)
    paths = list(image_paths)
    if not paths:
        result.warnings.append("Nebyl nalezen žádný snímek k analýze.")
        return result

    started = time.perf_counter()
    full_scale = probe_full_scale(paths)
    if bias is None:
        bias = compute_bias(paths, params, full_scale=full_scale)
    result.bias = bias

    if bias.frames_used < params.bias_frames:
        result.warnings.append(
            f"Pro bias bylo použito jen {bias.frames_used} z požadovaných {params.bias_frames} snímků."
        )

    timestamps, synthetic = build_time_axis(paths, assumed_fps=params.assumed_fps)
    result.synthetic_time_axis = synthetic
    if synthetic:
        result.warnings.append(
            "Názvy souborů neobsahují použitelná časová razítka – časová osa "
            f"byla dopočítána podle předpokládané frekvence {params.assumed_fps:g} sn./s."
        )
    t0 = timestamps[0] if timestamps else datetime.now()

    start_index = bias.frames_used if params.exclude_bias_from_series else 0
    work = [(i, paths[i], timestamps[i]) for i in range(start_index, len(paths))]

    frame_megapixels = float(bias.data.size) / 1e6
    workers = min(resolve_workers(params, frame_megapixels), max(1, len(work)))
    previous_cv_threads = cv2.getNumThreads()
    if workers > 1:
        # Zabráníme přeplnění CPU: paralelizujeme na úrovni snímků, ne uvnitř OpenCV.
        cv2.setNumThreads(1)

    def task(item: Tuple[int, str, datetime]):
        idx, path, stamp = item
        try:
            metrics, _ = analyze_frame_file(
                path=path,
                bias=bias,
                params=params,
                index=idx,
                timestamp=stamp,
                t0=t0,
                generate_masks=False,
            )
            if idx < bias.frames_used:
                metrics.is_bias_frame = True
                metrics.note = "bias"
            return idx, metrics, None
        except (FrameReadError, cv2.error, ValueError, MemoryError) as exc:
            return idx, None, f"{type(exc).__name__}: {exc}"

    collected: List[FrameMetrics] = []
    total = len(work)
    try:
        if workers <= 1:
            for done, item in enumerate(work, start=1):
                if should_cancel and should_cancel():
                    result.cancelled = True
                    break
                _idx, metrics, error = task(item)
                if metrics is None:
                    result.failed_files.append((item[1], error or "neznámá chyba"))
                    continue
                collected.append(metrics)
                if progress:
                    progress(done, total, metrics)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending: deque[Future] = deque()
                next_index = 0
                done = 0
                while next_index < total or pending:
                    if should_cancel and should_cancel():
                        result.cancelled = True
                        for fut in pending:
                            fut.cancel()
                        break
                    while next_index < total and len(pending) < workers * 2:
                        pending.append(pool.submit(task, work[next_index]))
                        next_index += 1
                    future = pending.popleft()
                    _idx, metrics, error = future.result()
                    done += 1
                    if metrics is None:
                        result.failed_files.append((paths[_idx], error or "neznámá chyba"))
                        continue
                    collected.append(metrics)
                    if progress:
                        progress(done, total, metrics)
    finally:
        cv2.setNumThreads(previous_cv_threads)

    collected.sort(key=lambda m: m.index)
    result.metrics = compute_rates_and_phases(collected)
    result.elapsed_s = time.perf_counter() - started

    if result.failed_files:
        count = len(result.failed_files)
        noun = "snímek" if count == 1 else ("snímky" if count < 5 else "snímků")
        result.warnings.append(f"Nepodařilo se zpracovat {count} {noun} (viz seznam chyb).")
    return result


__all__ = [
    "AnalysisParams",
    "BiasModel",
    "FrameMetrics",
    "SeriesResult",
    "analyze_frame_file",
    "analyze_prepared_frame",
    "analyze_series",
    "apply_geometry",
    "compute_bias",
    "compute_cleanliness_score",
    "compute_rates_and_phases",
    "estimate_noise",
    "focus_score",
    "match_shape",
    "parse_timestamp_from_filename",
    "separate_haze",
    "spatial_distribution",
]
