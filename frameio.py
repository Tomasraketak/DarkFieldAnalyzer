"""Vstupně-výstupní vrstva: bezpečné načítání snímků, řazení a časová razítka.

Proč samostatný modul:

* ``cv2.imread()`` na Windows **selže** (vrátí ``None``) u každé cesty s diakritikou
  nebo jiným ne-ASCII znakem (``C:\\Users\\Jiří\\Měření``). Původní verze aplikace
  takový snímek tiše přeskočila, takže analýza skončila s nula snímky a GUI
  zůstalo viset. Zde se čte přes ``np.fromfile`` + ``cv2.imdecode``, což je
  na Unicode cestách spolehlivé.
* Podpora 16bitových snímků (vědecké kamery ukládají 10/12/16 bitů do PNG/TIFF).
  Data se převádějí na jednotnou float32 škálu 0–255 ADU, aby všechny prahy
  v GUI měly stále stejný význam bez ohledu na bitovou hloubku kamery.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: Přípony, které aplikace považuje za snímky měření.
IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")

#: Předpony a přípony souborů, které aplikace sama vytváří.
#:
#: Export ukládá grafy jako PNG přímo do složky s měřením, takže při druhém
#: spuštění by je hledání snímků našlo jako běžné snímky. Protože se řadí
#: abecedně před ``df_00001_…``, staly by se z nich dokonce **bias snímky**
#: a celá analýza by se počítala proti obrázku grafu.
OUTPUT_NAME_PREFIXES: Tuple[str, ...] = ("analyza_", "analýza_")
OUTPUT_NAME_MARKERS: Tuple[str, ...] = (
    "_grafy", "_nahled", "_náhled", "_souhrn", "_overlay", "_maska", "_slozeni", "_složení",
)

#: Referenční rozsah, na který se všechny snímky normalizují (8bitová škála ADU).
DISPLAY_FULL_SCALE = 255.0

#: Jak se barevný snímek převede na jednokanálovou intenzitu.
#:
#: Kamera v BMS Cam Control umí ukládat mono i barevně a referenční pozadí
#: (``mean_mono`` v .npz) je vždy jednokanálové. Aby odečet reference dával
#: smysl, musí se barevný snímek promítnout do intenzity stejným způsobem,
#: jakým vznikl mono kanál v záznamové aplikaci – proto je volba nastavitelná.
#:
#: * ``luma``    – vážený jas Rec.601 (0,299 R + 0,587 G + 0,114 B). Standardní
#:                 „černobílý“ převod, který používá i sama kamera.
#: * ``prumer``  – prostý průměr kanálů. Nepodceňuje modrou, takže modravé
#:                 rozptylové halo částic v temném poli má plnou váhu.
#: * ``maximum`` – maximum kanálů. Nejcitlivější na částice, které svítí jen
#:                 v jednom kanálu, ale zvyšuje i šum pozadí (√3× u nekorelovaného šumu).
#: * ``r`` / ``g`` / ``b`` – jediný kanál.
MONO_MODES: Tuple[str, ...] = ("luma", "prumer", "maximum", "r", "g", "b")
DEFAULT_MONO_MODE = "luma"

#: Váhy Rec.601 v pořadí kanálů OpenCV (B, G, R).
_LUMA_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)


class FrameReadError(RuntimeError):
    """Snímek se nepodařilo načíst nebo dekódovat."""


class FrameShapeError(FrameReadError):
    """Snímek má jiné rozlišení než zbytek série – do měření nepatří."""


@dataclass(frozen=True)
class FrameData:
    """Načtený snímek převedený na jednotnou škálu 0–255 ADU."""

    data: np.ndarray          # float32, rozsah 0–255 ADU (vždy jednokanálový)
    native_dtype: str         # "uint8" / "uint16" / ...
    native_full_scale: float  # plný rozsah v původních jednotkách (255, 4095, 65535, ...)
    path: str
    source_channels: int = 1  # 1 = mono soubor, 3 = barevný soubor
    mono_mode: str = DEFAULT_MONO_MODE  # jak se barva převedla na intenzitu

    @property
    def is_color(self) -> bool:
        return self.source_channels >= 3

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Čtení a zápis
# ---------------------------------------------------------------------------

def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Načte obrázek i z cesty s diakritikou (náhrada za ``cv2.imread``)."""
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, flags)


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """Uloží obrázek i do cesty s diakritikou (náhrada za ``cv2.imwrite``)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


def _guess_full_scale(image: np.ndarray) -> float:
    """Odhadne plný rozsah senzoru (8/10/12/14/16 bitů) podle datového typu a maxima."""
    if image.dtype == np.uint8:
        return 255.0
    if image.dtype == np.uint16:
        peak = int(image.max()) if image.size else 0
        for bits in (10, 12, 14):
            full = float((1 << bits) - 1)
            if peak <= full:
                return full
        return 65535.0
    # float / int32 vstupy: normalizujeme podle maxima
    peak = float(image.max()) if image.size else 1.0
    if peak <= 1.0:
        return 1.0
    if peak <= 255.0:
        return 255.0
    return peak


def normalize_mono_mode(mode: Optional[str]) -> str:
    """Ošetří uživatelský vstup názvu převodu barvy na intenzitu."""
    text = (mode or DEFAULT_MONO_MODE).strip().lower()
    aliases = {
        "": DEFAULT_MONO_MODE,
        "auto": DEFAULT_MONO_MODE,
        "jas": "luma",
        "gray": "luma",
        "grey": "luma",
        "mean": "prumer",
        "průměr": "prumer",
        "avg": "prumer",
        "max": "maximum",
        "red": "r",
        "green": "g",
        "blue": "b",
        "červená": "r",
        "zelená": "g",
        "modrá": "b",
    }
    text = aliases.get(text, text)
    return text if text in MONO_MODES else DEFAULT_MONO_MODE


def to_mono(image: np.ndarray, mono_mode: str = DEFAULT_MONO_MODE) -> np.ndarray:
    """Převede barevný (BGR) snímek na jednokanálovou intenzitu.

    Mono snímek vrátí beze změny. Výsledek má vždy stejný datový typ jako
    vstup u celočíselných dat by zaokrouhlení posunulo úroveň pozadí, proto
    se počítá ve float32 a na původní typ se **nevrací** – volající pracuje
    dál ve float.
    """
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise FrameReadError(f"Nepodporovaný tvar snímku {image.shape}")

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] != 3:
        raise FrameReadError(f"Nepodporovaný počet kanálů: {image.shape[2]}")

    mode = normalize_mono_mode(mono_mode)
    if mode in ("b", "g", "r"):
        return image[:, :, {"b": 0, "g": 1, "r": 2}[mode]]

    planes = image.astype(np.float32, copy=False)
    if mode == "maximum":
        return planes.max(axis=2)
    if mode == "prumer":
        return planes.mean(axis=2, dtype=np.float32)
    return planes @ _LUMA_WEIGHTS          # luma (Rec.601, pořadí B, G, R)


def load_frame(
    path: str,
    full_scale: Optional[float] = None,
    mono_mode: str = DEFAULT_MONO_MODE,
) -> FrameData:
    """Načte snímek jako float32 v jednotné škále 0–255 ADU.

    Args:
        path: cesta ke snímku.
        full_scale: známý plný rozsah senzoru (např. 4095 pro 12bitovou kameru).
            Pokud je ``None``, odhadne se automaticky. Při analýze série se
            předává hodnota zjištěná z bias snímků, aby byla škála všech
            snímků v sérii identická.
        mono_mode: převod barevného snímku na intenzitu (viz :data:`MONO_MODES`).
            Mono snímků se netýká.

    Raises:
        FrameReadError: pokud soubor nelze načíst nebo dekódovat.
    """
    image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FrameReadError(f"Snímek nelze načíst nebo dekódovat: {path}")

    channels = 1 if image.ndim == 2 else (image.shape[2] if image.ndim == 3 else 0)
    if image.ndim not in (2, 3):
        raise FrameReadError(f"Nepodporovaný tvar snímku {image.shape}: {path}")

    native_dtype = str(image.dtype)
    scale = float(full_scale) if full_scale else _guess_full_scale(image)
    scale = max(scale, 1.0)

    mode = normalize_mono_mode(mono_mode)
    data = np.asarray(to_mono(image, mode), dtype=np.float32)
    if abs(scale - DISPLAY_FULL_SCALE) > 1e-6:
        data = data * np.float32(DISPLAY_FULL_SCALE / scale)

    return FrameData(
        data=np.ascontiguousarray(data, dtype=np.float32),
        native_dtype=native_dtype,
        native_full_scale=scale,
        path=path,
        source_channels=int(channels),
        mono_mode=mode,
    )


def probe_full_scale(paths: Sequence[str], sample: int = 3) -> float:
    """Zjistí plný rozsah senzoru z několika prvních snímků série."""
    best = 255.0
    for path in list(paths)[: max(1, sample)]:
        image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3:
            image = image[:, :, 0]
        best = max(best, _guess_full_scale(image))
    return best


# ---------------------------------------------------------------------------
# Vyhledávání souborů
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(\d+)")


def is_analysis_output(path: str) -> bool:
    """Pozná soubor, který vytvořila samotná aplikace (graf, náhled, souhrn).

    Slouží jako pojistka, aby se exportované grafy uložené ve složce s měřením
    nedostaly do analýzy jako snímky – a hlavně ne jako referenční bias.
    """
    name = os.path.basename(path).lower()
    stem = os.path.splitext(name)[0]
    if any(name.startswith(prefix) for prefix in OUTPUT_NAME_PREFIXES):
        return True
    return any(marker in stem for marker in OUTPUT_NAME_MARKERS)


def natural_sort_key(text: str):
    """Přirozené řazení (``img2`` před ``img10``)."""
    return [int(part) if part.isdigit() else part.lower() for part in _NUM_RE.split(text)]


def list_image_files(
    folder: str,
    recursive: bool = False,
    skip_outputs: bool = True,
) -> List[str]:
    """Vrátí seřazený seznam snímků ve složce.

    ``skip_outputs`` vynechá soubory, které vytvořila sama aplikace
    (exportované grafy, uložené náhledy) – viz :func:`is_analysis_output`.
    """
    if not os.path.isdir(folder):
        return []

    def accept(name: str, full: str) -> bool:
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            return False
        if skip_outputs and is_analysis_output(name):
            return False
        return os.path.isfile(full)

    found: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                full = os.path.join(root, name)
                if accept(name, full):
                    found.append(full)
    else:
        try:
            entries = os.listdir(folder)
        except OSError:
            return []
        for name in entries:
            full = os.path.join(folder, name)
            if accept(name, full):
                found.append(full)

    found.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return found


def list_measurement_folders(base_dir: str) -> List[Tuple[str, int]]:
    """Najde podsložky s měřeními (a případně i samotnou kořenovou složku).

    Vrací dvojice ``(cesta, počet snímků)``. Oproti původní verzi je do seznamu
    zahrnuta i samotná zvolená složka, pokud přímo obsahuje snímky – uživatel
    tak nemusí hádat, že musí vybrat o úroveň výš.
    """
    results: List[Tuple[str, int]] = []
    if not os.path.isdir(base_dir):
        return results

    own = len(list_image_files(base_dir))
    if own > 0:
        results.append((base_dir, own))

    try:
        entries = sorted(os.listdir(base_dir), key=natural_sort_key)
    except OSError:
        return results

    for name in entries:
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            count = len(list_image_files(full))
            if count > 0:
                results.append((full, count))
    return results


# ---------------------------------------------------------------------------
# Časová razítka
# ---------------------------------------------------------------------------

_TS_MS_RE = re.compile(r"(\d{8})_(\d{6})[_-](\d{1,6})")
_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_timestamp_from_filename(filename: str, fallback_mtime: Optional[float] = None) -> datetime:
    """Extrahuje časové razítko z názvu souboru (``df_00001_20260907_145622_018.png``).

    Bere se pouze *název* souboru, nikoliv celá cesta – jinak by čísla v cestě
    (např. ``BMS_20260101``) přebila skutečný čas snímku.
    """
    name = os.path.basename(filename)

    match = _TS_MS_RE.search(name)
    if match:
        frac = match.group(3)
        micro = int(frac.ljust(6, "0")[:6])
        try:
            base = datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
            return base.replace(microsecond=micro)
        except ValueError:
            pass

    match = _TS_RE.search(name)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    if fallback_mtime is None:
        try:
            fallback_mtime = os.path.getmtime(filename)
        except OSError:
            fallback_mtime = 0.0
    return datetime.fromtimestamp(fallback_mtime)


def build_time_axis(paths: Sequence[str], assumed_fps: float = 1.0) -> Tuple[List[datetime], bool]:
    """Vytvoří časovou osu série.

    Vrací ``(časová razítka, je_odhadnutá)``. Pokud jsou razítka nepoužitelná
    (všechna stejná nebo klesající – typicky když názvy souborů čas neobsahují
    a soubory byly hromadně zkopírovány), vygeneruje se náhradní rovnoměrná osa
    podle ``assumed_fps`` a druhá hodnota je ``True``.
    """
    stamps: List[datetime] = []
    for path in paths:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        stamps.append(parse_timestamp_from_filename(path, mtime))

    if len(stamps) < 2:
        return stamps, False

    span = (stamps[-1] - stamps[0]).total_seconds()
    monotonic = all(
        (stamps[i + 1] - stamps[i]).total_seconds() >= -1e-6 for i in range(len(stamps) - 1)
    )
    if span > 1e-6 and monotonic:
        return stamps, False

    step = 1.0 / max(assumed_fps, 1e-6)
    t0 = stamps[0]
    synthetic = [t0 + _seconds(i * step) for i in range(len(stamps))]
    return synthetic, True


def _seconds(value: float):
    from datetime import timedelta

    return timedelta(seconds=float(value))
