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
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from alignment import (
    AlignmentError,
    AlignmentModel,
    DEFAULT_MAX_SHIFT,
    DEFAULT_MIN_MATCHES,
    DEFAULT_STAR_COUNT,
    DEFAULT_STAR_SIGMA,
    FrameShift,
    detect_stars,
    estimate_shift,
    find_hot_pixels,
    repair_hot_pixels,
    intersect_roi,
    safe_crop_rect,
    warp_to_anchor,
)
from imageops import estimate_noise, separate_haze
from frameio import (
    DEFAULT_MONO_MODE,
    FrameReadError,
    FrameShapeError,
    build_time_axis,
    load_frame,
    normalize_mono_mode,
    parse_timestamp_from_filename,
    probe_full_scale,
)
from reference import (
    ReferenceChoice,
    ReferenceError,
    ReferenceRecord,
    find_reference_dir,
    list_references,
    load_reference,
    scale_to_adu,
    select_reference,
    series_start_time,
)

#: Nad tento počet kontaminovaných pixelů se přeskočí výpočet momentů
#: (ochrana paměti u extrémně zašuměných sérií) a použije se opsaný obdélník.
MOMENT_PIXEL_LIMIT = 6_000_000

#: Cílová výška obrazu pro automatickou volbu binningu.
AUTO_BINNING_TARGET_HEIGHT = 1200

#: Kolik souborů na začátku série se prohlédne, než se určí rozlišení měření.
SHAPE_PROBE_COUNT = 8

#: Percentil nízkofrekvenční složky, který se bere jako „úroveň pozadí“ při
#: srovnávání externí reference se snímky. Nízký percentil je dominantně
#: pozadí i tehdy, když je většina plochy kontaminovaná.
REFERENCE_LEVEL_PERCENTILE = 10.0

#: Nad tímto posunem úrovně (ADU) se do protokolu zapíše upozornění, že
#: reference nesedí na snímky (jiná expozice, jiný převod barvy na mono).
REFERENCE_LEVEL_WARN_ADU = 3.0

#: Menší posun než tento se ignoruje – dobře sedící reference se nemá „opravovat“.
REFERENCE_LEVEL_DEADBAND_ADU = 0.5

#: Kolik snímků rovnoměrně rozložených po sérii se prohlédne při odhadu posunu.
REFERENCE_LEVEL_SAMPLES = 6


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

    # --- reference (bias) z externí složky --------------------------------
    reference_mode: str = "auto"       # "auto" | "reference" | "serie"
    reference_dir: Optional[str] = None    # None = hledat složku "reference" automaticky
    match_reference_level: bool = True     # Srovnat úroveň externí reference se snímky

    # --- barevné snímky ----------------------------------------------------
    mono_mode: str = DEFAULT_MONO_MODE     # Převod barvy na intenzitu (viz frameio.MONO_MODES)

    # --- zarovnání snímků (drift sklíčka nebo kamery) ----------------------
    align_frames: bool = True              # Srovnat snímky podle souhvězdí částic
    align_star_count: int = DEFAULT_STAR_COUNT      # Kolik částic tvoří souhvězdí
    align_star_sigma: float = DEFAULT_STAR_SIGMA    # Práh detekce částice [× σ]
    align_max_shift_px: int = DEFAULT_MAX_SHIFT     # Největší uvažovaný posun [px]
    align_min_matches: int = DEFAULT_MIN_MATCHES    # Minimum spárovaných částic
    align_crop_mode: str = "auto"          # "auto" (podle driftu) | "fixed" | "none"
    align_crop_fraction: float = 0.90      # Podíl plochy při "fixed"
    align_rotation: bool = True            # Odhadovat i pootočení scény

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

    @property
    def mono_label(self) -> str:
        return {
            "luma": "vážený jas (Rec.601)",
            "prumer": "průměr kanálů",
            "maximum": "maximum kanálů",
            "r": "jen červený kanál",
            "g": "jen zelený kanál",
            "b": "jen modrý kanál",
        }.get(normalize_mono_mode(self.mono_mode), self.mono_mode)


@dataclass
class BiasModel:
    """Referenční pozadí (master bias) připravené v geometrii analýzy."""

    data: np.ndarray            # float32, po binningu a ořezu ROI
    frames_used: int
    binning: int
    full_scale: float
    source_shape: Tuple[int, int]  # rozměr původního (nebinovaného) snímku
    frame_paths: List[str] = field(default_factory=list)     # soubory tvořící bias
    rejected_paths: List[Tuple[str, str]] = field(default_factory=list)  # (cesta, důvod)

    # Externí reference ze složky "reference" (soubory .npz z BMS Cam Control)
    origin: str = "serie"                     # "serie" | "reference"
    reference_path: Optional[str] = None
    reference_created: Optional[datetime] = None
    reference_label: str = ""
    level_offset_adu: float = 0.0             # konstantní posun úrovně reference

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def is_external(self) -> bool:
        """True, pokud pozadí nepochází ze snímků samotné série."""
        return self.origin == "reference"

    @property
    def origin_label(self) -> str:
        if self.is_external:
            return f"reference {os.path.basename(self.reference_path or '')}"
        return f"prvních {self.frames_used} snímků série"


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

    #: Posun úrovně, o který se srovnala externí reference se snímkem [ADU].
    #: U biasu počítaného ze série je vždy 0.
    reference_offset_adu: float = 0.0

    #: Naměřený drift snímku vůči kotvě (v px plného rozlišení) a jeho kvalita.
    align_dx_px: float = 0.0
    align_dy_px: float = 0.0
    align_rotation_deg: float = 0.0
    align_stars: int = 0
    align_rms_px: float = 0.0
    align_ok: bool = False

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
    #: Informativní poznámky k průběhu (která reference se vzala, odkud pozadí).
    #: Na rozdíl od ``warnings`` neznamenají problém a GUI je nevyskakuje v dialogu.
    notes: List[str] = field(default_factory=list)
    synthetic_time_axis: bool = False
    cancelled: bool = False

    #: Model zarovnání driftu (``None`` = snímky se nesrovnávaly).
    alignment: Optional[AlignmentModel] = None

    #: Zvolená externí reference (``None`` = bias se počítal ze série).
    reference: Optional[ReferenceRecord] = None
    reference_dir: Optional[str] = None
    reference_note: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.metrics)


# ---------------------------------------------------------------------------
# Geometrie: binning, ROI, sladění biasu
# ---------------------------------------------------------------------------

def apply_binning(image: np.ndarray, binning: int) -> np.ndarray:
    """Zmenší obraz průměrováním bloků ``binning × binning`` (INTER_AREA)."""
    if binning <= 1:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, w // binning), max(1, h // binning)),
                      interpolation=cv2.INTER_AREA)


def crop_to_roi(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]],
                binning: int) -> np.ndarray:
    """Ořízne obraz na ROI zadaný v pixelech **plného rozlišení**."""
    if roi is None:
        return np.ascontiguousarray(image, dtype=np.float32)

    rx, ry, rw, rh = (int(v) for v in roi)
    if binning > 1:
        rx, ry, rw, rh = rx // binning, ry // binning, rw // binning, rh // binning
    h, w = image.shape[:2]
    rx = max(0, min(rx, w - 1))
    ry = max(0, min(ry, h - 1))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))
    return np.ascontiguousarray(image[ry : ry + rh, rx : rx + rw], dtype=np.float32)


def apply_geometry(
    image: np.ndarray,
    binning: int,
    roi: Optional[Tuple[int, int, int, int]],
    shift: Optional[FrameShift] = None,
) -> np.ndarray:
    """Aplikuje binning, volitelné srovnání driftu a ořez ROI na float32 snímek.

    Pořadí je závazné: **binning → srovnání → ořez**. Srovnání musí proběhnout
    nad celým (neořezaným) obrazem, jinak by se do výřezu natáhl neplatný okraj;
    ořez naopak až po srovnání, protože právě on ten neplatný okraj odřízne.
    """
    out = apply_binning(image, binning)
    if shift is not None:
        out = warp_to_anchor(out, shift)
    return crop_to_roi(out, roi, binning)


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
# Rozlišení série (pojistka proti cizím souborům ve složce)
# ---------------------------------------------------------------------------

@dataclass
class SeriesProbe:
    """Co se zjistilo z prvních souborů série."""

    source_shape: Tuple[int, int]
    frames: List[Tuple[str, np.ndarray]] = field(default_factory=list)   # jen shodné rozlišení
    rejected: List[Tuple[str, str]] = field(default_factory=list)        # (cesta, důvod)


def probe_series_shape(
    image_paths: Sequence[str],
    params: AnalysisParams,
    full_scale: float,
    probe_count: int = SHAPE_PROBE_COUNT,
) -> SeriesProbe:
    """Určí rozlišení série hlasováním z několika prvních souborů.

    Rozlišení se **nebere z prvního souboru v pořadí**: kdyby se do složky
    dostal cizí obrázek (typicky exportovaný graf, který se řadí abecedně před
    ``df_00001_…``), stal by se jinak měřítkem pro celé měření – a při
    interním biasu dokonce referenčním pozadím. Soubory s jiným rozlišením
    se vrací v ``rejected`` a z analýzy se vyřazují.
    """
    candidates: List[Tuple[str, np.ndarray]] = []
    rejected: List[Tuple[str, str]] = []
    mono_mode = normalize_mono_mode(params.mono_mode)

    for path in list(image_paths)[: max(1, probe_count)]:
        try:
            frame = load_frame(path, full_scale=full_scale, mono_mode=mono_mode)
        except FrameReadError as exc:
            rejected.append((path, str(exc)))
            continue
        candidates.append((path, frame.data))

    if not candidates:
        return SeriesProbe(source_shape=(0, 0), frames=[], rejected=rejected)

    shape_counts: Dict[Tuple[int, int], int] = {}
    for _path, data in candidates:
        shape_counts[data.shape] = shape_counts.get(data.shape, 0) + 1
    # Nejčastější rozlišení; při shodě vyhraje to, které se objevilo dřív.
    source_shape = max(shape_counts, key=lambda shape: shape_counts[shape])

    usable: List[Tuple[str, np.ndarray]] = []
    for path, data in candidates:
        if data.shape == source_shape:
            usable.append((path, data))
        else:
            rejected.append((
                path,
                f"rozlišení {data.shape[1]}×{data.shape[0]} px neodpovídá sérii "
                f"({source_shape[1]}×{source_shape[0]} px)",
            ))

    return SeriesProbe(source_shape=source_shape, frames=usable, rejected=rejected)


# ---------------------------------------------------------------------------
# Zarovnání série (drift sklíčka nebo kamery)
# ---------------------------------------------------------------------------

#: Kolik snímků rovnoměrně po sérii se prohlédne při odhadu celkového driftu.
ALIGN_PROBE_FRAMES = 10


def _load_binned(path: str, params: AnalysisParams, full_scale: float, binning: int) -> np.ndarray:
    """Načte snímek a zmenší ho do geometrie analýzy (bez ořezu)."""
    frame = load_frame(path, full_scale=full_scale, mono_mode=params.mono_mode)
    return apply_binning(frame.data, binning)


def build_alignment(
    image_paths: Sequence[str],
    params: AnalysisParams,
    source_shape: Tuple[int, int],
    full_scale: float,
    probe_frames: int = ALIGN_PROBE_FRAMES,
) -> Optional[AlignmentModel]:
    """Sestaví model zarovnání: kotvu, masku horkých pixelů a bezpečný ořez.

    Kotvou je **první snímek série** – k němu se srovnává všechno ostatní,
    takže se chyby nesčítají tak, jako kdyby se každý snímek srovnával
    k předchozímu.

    Velikost ořezu se odvozuje z driftu naměřeného na vzorku snímků rozložených
    po celé sérii. Drift bývá jednosměrný, takže krajní snímky zachytí jeho
    maximum; k naměřené hodnotě se ještě přidává rezerva.
    """
    paths = list(image_paths)
    if not params.align_frames or not paths:
        return None

    binning = params.resolve_binning(source_shape[0])
    try:
        anchor_image = _load_binned(paths[0], params, full_scale, binning)
    except FrameReadError as exc:
        raise AlignmentError(f"Kotevní snímek nelze načíst: {exc}") from exc

    # Horké pixely se hledají v samotné kotvě: je to test ostrosti jednotlivého
    # bodu, žádnou další referenci k tomu není potřeba.
    hot_mask = find_hot_pixels(anchor_image)
    anchor_image = repair_hot_pixels(anchor_image, hot_mask)
    anchor_stars = detect_stars(
        anchor_image, hot_mask=hot_mask,
        star_count=params.align_star_count, sigma=params.align_star_sigma,
    )

    model = AlignmentModel(
        anchor=anchor_stars,
        anchor_path=paths[0],
        binning=binning,
        max_shift_px=int(params.align_max_shift_px),
        min_matches=int(params.align_min_matches),
        star_count=int(params.align_star_count),
        star_sigma=float(params.align_star_sigma),
        hot_mask=hot_mask,
        hot_pixel_count=int(hot_mask.sum()) if hot_mask is not None else 0,
        estimate_rotation=bool(params.align_rotation),
    )

    if anchor_stars.count < params.align_min_matches:
        model.notes.append(
            f"Na prvním snímku se našlo jen {anchor_stars.count} zřetelných částic "
            f"(potřeba {params.align_min_matches}) – zarovnání se nepoužije."
        )
        model.usable = False
        return model

    # --- drift na vzorku snímků -------------------------------------------
    picks = _spread_sample(paths, probe_frames)
    for path in picks:
        if path == paths[0]:
            model.sampled.append((path, FrameShift(matched=anchor_stars.count, ok=True)))
            continue
        try:
            binned = _load_binned(path, params, full_scale, binning)
        except FrameReadError:
            continue
        if binned.shape != anchor_image.shape:
            continue
        model.sampled.append((path, model.measure(repair_hot_pixels(binned, hot_mask))))

    # Když se nepodařilo srovnat ani jeden ze vzorkovaných snímků, nemá smysl
    # nic ořezávat ani zdržovat analýzu měřením u každého snímku – souhvězdí
    # v téhle sérii prostě není (prázdné sklíčko, málo výrazných částic).
    others = [shift for path, shift in model.sampled if path != paths[0]]
    if others and not any(shift.ok for shift in others):
        reason = next((s.reason for s in others if s.reason), "částice se nepodařilo spárovat")
        model.notes.append(
            f"Snímky nejde zarovnat podle souhvězdí částic ({reason}) – "
            "zarovnání se nepoužije."
        )
        model.usable = False
        return model

    drift = model.sampled_drift_px
    mode = (params.align_crop_mode or "auto").strip().lower()
    if mode in ("none", "zadny", "žádný", "vypnuto"):
        model.crop, model.crop_fraction = None, 1.0
    else:
        model.crop, model.crop_fraction = safe_crop_rect(
            source_shape, drift_px=drift, mode=mode, fraction=params.align_crop_fraction,
        )
    return model


def _spread_sample(paths: Sequence[str], count: int) -> List[str]:
    """Vybere ``count`` souborů rovnoměrně rozložených po sérii (včetně krajních)."""
    items = list(paths)
    count = max(1, min(int(count), len(items)))
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / float(count - 1)
    return [items[int(round(i * step))] for i in range(count)]


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

def compute_bias(
    image_paths: Sequence[str],
    params: AnalysisParams,
    full_scale: Optional[float] = None,
    alignment: Optional[AlignmentModel] = None,
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

    probe = probe_series_shape(image_paths, params, full_scale, probe_count=max(n, SHAPE_PROBE_COUNT))
    if not probe.frames:
        raise ValueError("Ze začátku série se nepodařilo načíst žádný snímek.")

    source_shape = probe.source_shape
    rejected = list(probe.rejected)
    binning = params.resolve_binning(source_shape[0])
    usable = probe.frames[:n]

    # Bias snímky se srovnávají na kotvu ještě před mediánem – jinak by se
    # drift mezi nimi propsal do reference jako rozmazání částic.
    prepared = []
    for _path, data in usable:
        binned = apply_binning(data, binning)
        if alignment is not None and alignment.usable:
            binned = repair_hot_pixels(binned, alignment.hot_mask)
            shift = alignment.measure(binned)
            if shift.ok:
                binned = warp_to_anchor(binned, shift)
        prepared.append(crop_to_roi(binned, params.roi, binning))
    bias_paths = [path for path, _data in usable]

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
        frame_paths=bias_paths,
        rejected_paths=rejected,
    )


# ---------------------------------------------------------------------------
# Bias z externí reference (.npz z BMS Cam Control)
# ---------------------------------------------------------------------------

def bias_from_reference(
    record: ReferenceRecord,
    params: AnalysisParams,
    source_shape: Tuple[int, int],
    full_scale: float = 255.0,
    rejected_paths: Optional[Sequence[Tuple[str, str]]] = None,
    alignment: Optional[AlignmentModel] = None,
) -> BiasModel:
    """Postaví :class:`BiasModel` z uložené reference.

    Reference se pořizuje na stejné rozlišení jako měření, ale kdyby se
    lišila (jiný režim kamery), přeškáluje se – pozadí je nízkofrekvenční,
    takže interpolace ho nezkreslí. Rozdílné rozlišení se hlásí jako
    upozornění, protože obvykle znamená, že reference patří k jinému měření.

    Raises:
        ReferenceError: referenci nelze načíst.
    """
    planes = load_reference(record)
    # Reference je uložená v jednotkách senzoru (8bit 0–255, 12bit 0–4095),
    # stejně jako snímky – převede se proto stejnou škálou jako ony.
    reference = scale_to_adu(planes.mono, full_scale)

    if reference.shape != tuple(source_shape):
        reference = cv2.resize(
            reference, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_AREA
        )

    binning = params.resolve_binning(source_shape[0])
    binned = apply_binning(reference, binning)

    # Reference vznikla dřív, takže scéna na ní může být posunutá vůči kotvě.
    # Bez srovnání by se statické částice nekryly a zůstaly by po nich dipóly.
    bias_shift: Optional[FrameShift] = None
    if alignment is not None and alignment.usable:
        binned = repair_hot_pixels(binned, alignment.hot_mask)
        bias_shift = alignment.measure(binned)
        if bias_shift.ok:
            binned = warp_to_anchor(binned, bias_shift)
        alignment.bias_shift = bias_shift
    data = crop_to_roi(binned, params.roi, binning)

    return BiasModel(
        data=np.ascontiguousarray(data, dtype=np.float32),
        frames_used=int(planes.frames or record.frames or 1),
        binning=binning,
        full_scale=float(full_scale),
        source_shape=tuple(int(v) for v in source_shape),  # type: ignore[arg-type]
        frame_paths=[],                      # žádný snímek série se nespotřebuje
        rejected_paths=list(rejected_paths or []),
        origin="reference",
        reference_path=record.npz_path,
        reference_created=record.created,
        reference_label=record.label,
    )


def estimate_reference_offset(
    image_paths: Sequence[str],
    bias: BiasModel,
    params: AnalysisParams,
    samples: int = REFERENCE_LEVEL_SAMPLES,
    alignment: Optional[AlignmentModel] = None,
) -> float:
    """Odhadne konstantní posun úrovně mezi externí referencí a sérií [ADU].

    Postup: u několika snímků rovnoměrně rozložených po sérii se změří úroveň
    pozadí jako nízký percentil nízkofrekvenční složky diference (tedy jas
    nejtmavšího místa pole, kde kontaminace není). Z těchto hodnot se vezme
    **minimum přes celou sérii**.

    Proč minimum: v temném poli kontaminace jen *přidává* světlo, nikdy ho
    neubírá. Konstantní rozdíl přístrojového původu je proto v každém snímku
    stejný, kdežto zamlžení se v čase mění – nejnižší naměřená úroveň je tak
    nejlepší odhad skutečné nuly. Kdyby se posun počítal pro každý snímek
    zvlášť, odečetlo by se i plošné zamlžení, které má analýza naopak měřit.

    Cenou je, že se odečte i kontaminace, která je po celou sérii konstantní.
    Proto se korekce používá jen u externí reference a jen nad mezí
    :data:`REFERENCE_LEVEL_DEADBAND_ADU`.

    Snímky se sem berou **už srovnané na kotvu**. Bez toho by drift sklíčka
    nechal po každé statické částici dvojici světlý/tmavý půlměsíc, záporné
    půlky by stáhly nízký percentil hluboko pod nulu a výsledkem by byl
    několikaADUový „posun úrovně“, který ve skutečnosti neexistuje.
    """
    paths = list(image_paths)
    if not paths:
        return 0.0

    count = max(1, min(int(samples), len(paths)))
    if count == 1:
        picks = [paths[0]]
    else:
        step = (len(paths) - 1) / float(count - 1)
        picks = [paths[int(round(i * step))] for i in range(count)]

    levels: List[float] = []
    for path in picks:
        try:
            frame = load_frame(path, full_scale=bias.full_scale, mono_mode=params.mono_mode)
        except FrameReadError:
            continue
        if frame.data.shape != bias.source_shape:
            continue
        binned = apply_binning(frame.data, bias.binning)
        if alignment is not None and alignment.usable:
            binned = repair_hot_pixels(binned, alignment.hot_mask)
            shift = alignment.measure(binned)
            if shift.ok:
                binned = warp_to_anchor(binned, shift)
        prepared = crop_to_roi(binned, params.roi, bias.binning)
        diff = cv2.subtract(prepared, match_shape(bias.data, prepared.shape[:2]))
        _haze_full, haze_small = separate_haze(diff)
        if haze_small.size:
            levels.append(float(np.percentile(haze_small, REFERENCE_LEVEL_PERCENTILE)))

    if not levels:
        return 0.0
    offset = min(levels)
    return offset if abs(offset) >= REFERENCE_LEVEL_DEADBAND_ADU else 0.0


def resolve_reference(
    image_paths: Sequence[str],
    params: AnalysisParams,
    folder: Optional[str] = None,
) -> Tuple[Optional[ReferenceChoice], Optional[str]]:
    """Najde a vybere referenci pro sérii. Vrací ``(volba, složka)``.

    Reference se vybírá podle času **prvního snímku série** – hledá se ta
    poslední pořízená ještě před ním, protože jen taková popisuje pozadí
    platné v okamžiku měření.
    """
    if not image_paths:
        return None, None

    base = folder or os.path.dirname(os.path.abspath(image_paths[0]))
    reference_dir = params.reference_dir
    if reference_dir and os.path.isdir(reference_dir):
        directory: Optional[str] = reference_dir
    else:
        directory = find_reference_dir(base, extra_roots=[reference_dir] if reference_dir else ())

    if not directory:
        return None, None

    records = list_references(directory)
    if not records:
        return ReferenceChoice(record=None, reason="Složka s referencemi je prázdná."), directory

    return select_reference(records, series_start_time(image_paths)), directory


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
    level_offset: float = 0.0,
) -> Tuple[FrameMetrics, Optional[Dict[str, np.ndarray]]]:
    """Analyzuje snímek, který je již ve float32 ADU a v geometrii analýzy.

    ``level_offset`` je konstantní posun úrovně (v ADU), který se od diference
    odečte. Používá se jen u **externí reference**: ta vznikla v jiném
    okamžiku, takže se od snímků může lišit konstantou (jiná teplota senzoru,
    jiný jas zdroje, u barevné kamery jiný převod na mono) a bez srovnání by
    taková konstanta prošla jako plošné zamlžení přes celý snímek. Posun se
    odhaduje **jednou pro celou sérii** (viz :func:`estimate_reference_offset`),
    ne pro každý snímek zvlášť – jinak by se odečetlo i skutečné zamlžení,
    které je v každém snímku jiné.
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    bias = match_shape(np.asarray(bias, dtype=np.float32), image.shape[:2])

    h, w = image.shape[:2]
    total_pixels = float(h * w)
    px_scale = float(binning * binning)        # px po binningu → px plného rozlišení

    diff = cv2.subtract(image, bias)
    reference_offset = float(level_offset)
    if abs(reference_offset) > 1e-3:
        diff = diff - np.float32(reference_offset)

    haze_full, haze_small = separate_haze(diff)
    bg_level, bg_sigma = estimate_noise(diff)
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
        reference_offset_adu=reference_offset,
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
    alignment: Optional[AlignmentModel] = None,
) -> Tuple[FrameMetrics, Optional[Dict[str, np.ndarray]]]:
    """Načte snímek ze souboru, srovná ho na kotvu a zanalyzuje."""
    frame = load_frame(path, full_scale=bias.full_scale, mono_mode=params.mono_mode)
    if frame.data.shape != bias.source_shape:
        # Druhá pojistka proti cizím obrázkům ve složce (exportované grafy,
        # náhledy, snímky z jiného měření): rozměr musí sedět na zbytek série.
        raise FrameShapeError(
            f"rozlišení {frame.data.shape[1]}×{frame.data.shape[0]} px neodpovídá sérii "
            f"({bias.source_shape[1]}×{bias.source_shape[0]} px)"
        )

    binned = apply_binning(frame.data, bias.binning)
    shift: Optional[FrameShift] = None
    if alignment is not None and alignment.usable:
        binned = repair_hot_pixels(binned, alignment.hot_mask)
        shift = alignment.measure(binned)
        if shift.ok:
            binned = warp_to_anchor(binned, shift)
    prepared = crop_to_roi(binned, params.roi, bias.binning)

    metrics, masks = analyze_prepared_frame(
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
        level_offset=bias.level_offset_adu,
    )

    if shift is not None:
        # Posun se hlásí v pixelech plného rozlišení, aby se dal převést na µm.
        metrics.align_dx_px = shift.dx * bias.binning
        metrics.align_dy_px = shift.dy * bias.binning
        metrics.align_rotation_deg = shift.angle_deg
        metrics.align_stars = shift.matched
        metrics.align_rms_px = shift.rms_px * bias.binning
        metrics.align_ok = shift.ok
        if not shift.ok and shift.reason:
            metrics.note = (metrics.note + "; " if metrics.note else "") + \
                f"nezarovnáno ({shift.reason})"
    return metrics, masks


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


def _report_alignment(
    result: SeriesResult,
    alignment: AlignmentModel,
    metrics_list: Sequence[FrameMetrics],
    params: AnalysisParams,
) -> None:
    """Doplní do výsledku poznámky a upozornění k zarovnání driftu."""
    aligned = [m for m in metrics_list if m.align_ok]
    failed = len(metrics_list) - len(aligned)

    if not aligned:
        # Není to chyba měření – na sérii bez výrazných částic prostě není co
        # sledovat. Analýza doběhla se snímky v původní poloze.
        result.notes.append(
            "Snímky se nepodařilo zarovnat – souhvězdí částic nebylo nalezeno. "
            "Pokud se scéna během měření hýbala, výsledky to zkreslí."
        )
        return

    drift = max(math.hypot(m.align_dx_px, m.align_dy_px) for m in aligned)
    stars = sum(m.align_stars for m in aligned) / len(aligned)
    rotation = max(abs(m.align_rotation_deg) for m in aligned)
    alignment.measured_drift_px = drift

    crop_note = ""
    if alignment.crop is not None:
        crop_note = (f", ořezáno na {alignment.crop[2]}×{alignment.crop[3]} px "
                     f"({alignment.crop_fraction * 100:.1f} % plochy)")
    result.notes.append(
        f"Snímky srovnány podle souhvězdí částic: největší drift {drift:.1f} px"
        f" ({drift * params.um_per_px:.1f} µm), průměrně {stars:.0f} spárovaných částic"
        f"{crop_note}."
    )

    if alignment.bias_shift is not None and alignment.bias_shift.ok:
        shift = alignment.bias_shift
        result.notes.append(
            f"Referenční pozadí bylo posunuto o "
            f"({shift.dx * alignment.binning:+.1f}, {shift.dy * alignment.binning:+.1f}) px, "
            f"aby sedělo na sérii."
        )

    if failed:
        result.warnings.append(
            f"{failed} snímků se nepodařilo zarovnat (málo zřetelných částic) – "
            "zůstaly v původní poloze, jejich hodnoty mohou být nadhodnocené."
        )
    if rotation > 0.05:
        result.notes.append(f"Scéna se během měření pootočila až o {rotation:.2f}°.")

    margin = alignment.crop_margin_px
    if margin is not None and drift > margin:
        result.warnings.append(
            f"Drift {drift:.1f} px přesáhl rezervu ořezu ({margin:.1f} px). "
            "Zvětšete ořez (režim „pevný podíl“) nebo zkontrolujte upevnění vzorku."
        )


def _build_bias(
    paths: Sequence[str],
    params: AnalysisParams,
    full_scale: float,
    result: SeriesResult,
    alignment: Optional[AlignmentModel] = None,
) -> BiasModel:
    """Vybere zdroj referenčního pozadí a sestaví :class:`BiasModel`.

    Pořadí:

    1. ``reference_mode="serie"`` – bias se počítá z prvních snímků série.
    2. jinak se hledá složka s referencemi a vybere se ta pořízená naposledy
       **před** začátkem měření;
    3. ``reference_mode="auto"`` navíc při jakémkoli problému (chybí složka,
       poškozený .npz) tiše spadne zpět na bias ze série, ať analýza doběhne.
       ``reference_mode="reference"`` naopak selže, aby se výsledek nepočítal
       proti jinému pozadí, než uživatel čekal.
    """
    mode = (params.reference_mode or "auto").strip().lower()
    if mode in ("serie", "série", "series", "vypnuto", "off"):
        return compute_bias(paths, params, full_scale=full_scale, alignment=alignment)

    strict = mode in ("reference", "vzdy", "vždy")
    choice, directory = resolve_reference(paths, params)
    result.reference_dir = directory

    if choice is None or not choice.ok:
        message = (
            choice.reason if choice is not None
            else "Složka s referencemi (…/reference) nebyla nalezena."
        )
        if strict:
            raise ReferenceError(message)
        # V automatickém režimu je návrat k biasu ze série normální provoz,
        # ne chyba – proto jen poznámka, ne varovný dialog.
        result.notes.append(f"{message} Použit bias z prvních snímků série.")
        return compute_bias(paths, params, full_scale=full_scale, alignment=alignment)

    record = choice.record
    assert record is not None

    # Rozlišení série se určuje vždy ze snímků, i když pozadí přijde odjinud –
    # jinak by se do měření dostaly cizí soubory ležící ve složce.
    probe = probe_series_shape(paths, params, full_scale)
    if not probe.frames:
        raise ValueError("Ze začátku série se nepodařilo načíst žádný snímek.")

    try:
        bias = bias_from_reference(
            record,
            params,
            source_shape=probe.source_shape,
            full_scale=full_scale,
            rejected_paths=probe.rejected,
            alignment=alignment,
        )
    except ReferenceError as exc:
        if strict:
            raise
        result.warnings.append(f"{exc} Použit bias z prvních snímků série.")
        return compute_bias(paths, params, full_scale=full_scale, alignment=alignment)

    result.reference = record
    result.reference_note = choice.reason
    result.notes.append(choice.reason)
    if choice.taken_after:
        result.warnings.append(
            "Použitá reference je novější než měřené snímky – výsledky ověřte,"
            " pozadí nemuselo v době měření ještě platit."
        )
    if record.resolution and tuple(record.resolution) != tuple(probe.source_shape):
        result.warnings.append(
            f"Reference má rozlišení {record.resolution[1]}×{record.resolution[0]} px,"
            f" série {probe.source_shape[1]}×{probe.source_shape[0]} px – pozadí bylo přeškálováno."
        )

    if params.match_reference_level:
        rejected = {path for path, _reason in probe.rejected}
        usable = [path for path in paths if path not in rejected]
        bias.level_offset_adu = estimate_reference_offset(
            usable, bias, params, alignment=alignment)
        if abs(bias.level_offset_adu) > REFERENCE_LEVEL_WARN_ADU:
            direction = "světlejší" if bias.level_offset_adu > 0 else "tmavší"
            result.warnings.append(
                f"Snímky jsou proti referenci systematicky o {abs(bias.level_offset_adu):.1f} ADU"
                f" {direction}; úroveň byla srovnána. Ověřte, že reference patří k tomuto"
                f" měření a že sedí režim převodu barvy ({params.mono_label})."
            )
    return bias


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

    # --- zarovnání driftu --------------------------------------------------
    # Musí se rozhodnout dřív než cokoli jiného: určuje totiž bezpečný ořez,
    # a ten už musí platit pro bias i pro každý snímek stejně.
    alignment: Optional[AlignmentModel] = None
    if bias is None and params.align_frames:
        probe = probe_series_shape(paths, params, full_scale)
        if probe.frames:
            usable_paths = [p for p in paths if p not in dict(probe.rejected)]
            try:
                alignment = build_alignment(usable_paths, params, probe.source_shape, full_scale)
            except (AlignmentError, cv2.error, MemoryError) as exc:
                result.warnings.append(f"Zarovnání snímků se nezdařilo: {exc}")
                alignment = None

    if alignment is not None and not alignment.usable:
        result.notes.extend(alignment.notes)
        alignment = None

    if alignment is not None:
        try:
            effective_roi = intersect_roi(params.roi, alignment.crop)
        except AlignmentError as exc:
            result.warnings.append(str(exc))
            effective_roi = params.roi
            alignment.crop = None
        params = replace(params, roi=effective_roi)
        result.params = params
        result.alignment = alignment

    if bias is None:
        bias = _build_bias(paths, params, full_scale, result, alignment=alignment)
    result.bias = bias

    if not bias.is_external and bias.frames_used < params.bias_frames:
        result.warnings.append(
            f"Pro bias bylo použito jen {bias.frames_used} z požadovaných {params.bias_frames} snímků."
        )

    # Soubory, které do měření nepatří (jiné rozlišení – typicky exportovaný graf
    # nebo snímek z jiné série), se vyřadí ještě před sestavením časové osy.
    # Kdyby v seznamu zůstaly, rozhodí razítka i číslování bias snímků.
    rejected = dict(bias.rejected_paths)
    if rejected:
        result.failed_files.extend(bias.rejected_paths)
        paths = [path for path in paths if path not in rejected]
        if not paths:
            result.warnings.append("Po vyřazení cizích souborů nezbyl žádný snímek k analýze.")
            return result

    timestamps, synthetic = build_time_axis(paths, assumed_fps=params.assumed_fps)
    result.synthetic_time_axis = synthetic
    if synthetic:
        result.warnings.append(
            "Názvy souborů neobsahují použitelná časová razítka – časová osa "
            f"byla dopočítána podle předpokládané frekvence {params.assumed_fps:g} sn./s."
        )
    t0 = timestamps[0] if timestamps else datetime.now()

    # Bias snímky se poznají podle cesty, ne podle pořadí – v seznamu totiž
    # nemusí být první, pokud se před ně abecedně vloudil jiný soubor.
    bias_paths = set(bias.frame_paths)
    work = [
        (i, paths[i], timestamps[i])
        for i in range(len(paths))
        if not (params.exclude_bias_from_series and paths[i] in bias_paths)
    ]

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
                alignment=alignment,
            )
            if path in bias_paths:
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

    if alignment is not None and collected:
        _report_alignment(result, alignment, collected, params)

    if result.failed_files:
        count = len(result.failed_files)
        if count == 1:
            phrase = "1 soubor nebyl zahrnut"
        elif count < 5:
            phrase = f"{count} soubory nebyly zahrnuty"
        else:
            phrase = f"{count} souborů nebylo zahrnuto"
        result.warnings.append(
            f"{phrase} do analýzy – nečitelný soubor nebo jiné rozlišení "
            "než zbytek série (viz seznam chyb)."
        )
    return result


__all__ = [
    "AnalysisParams",
    "BiasModel",
    "FrameMetrics",
    "SeriesProbe",
    "SeriesResult",
    "analyze_frame_file",
    "bias_from_reference",
    "estimate_reference_offset",
    "probe_series_shape",
    "resolve_reference",
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
