"""Obecné operace nad obrazem, které nezávisí na parametrech analýzy.

Samostatný modul má jediný důvod: tyhle dvě funkce potřebuje jak analytické
jádro (``analyzer.py``), tak zarovnávání snímků (``alignment.py``), a bez
společného modulu by mezi nimi vznikl cyklický import.

``analyzer`` obojí re-exportuje, takže ``from analyzer import estimate_noise``
funguje dál.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


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


__all__ = ["estimate_noise", "separate_haze"]
