r"""Referenční pozadí (bias) uložené aplikací BMS Cam Control.

Záznamový program ukládá referenci mimo složku s měřením – do společné složky
``…\BMS fotky\reference`` – jako dvojici souborů:

``reference_20260908_142910.npz``
    Komprimovaný NumPy archiv. Klíč ``mean_mono`` obsahuje průměr N tmavých
    snímků jako ``float32`` na 8bitové škále ADU, ``frames_mono`` počet
    zprůměrovaných snímků a ``created_mono`` čas pořízení. Pokud kamera
    snímala barevně, mohou být přítomny i kanálové roviny (``mean_red`` …).

``reference_20260908_142910.json``
    Popis nastavení kamery, osvětlení a prahů v okamžiku pořízení reference.
    Slouží jen k informaci – analýza z něj nic nepřebírá.

Proč vlastní modul: výběr reference je samostatná úloha s vlastními pravidly
(najít složku, seřadit podle času, vybrat tu **poslední pořízenou před**
začátkem měření) a nemá co dělat ani ve vstupně-výstupní vrstvě, ani
v analytickém jádru.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from frameio import DISPLAY_FULL_SCALE, natural_sort_key, parse_timestamp_from_filename

#: Názvy podsložek, ve kterých se reference hledá (v tomto pořadí).
REFERENCE_DIR_NAMES: Tuple[str, ...] = ("reference", "reference_bias", "bias", "reference_npz")

#: O kolik úrovní nad složkou s měřením se ještě hledá složka s referencemi.
REFERENCE_SEARCH_LEVELS = 3

#: Nad tímto odstupem (v hodinách) se v protokolu objeví upozornění.
#: Reference by měla vzniknout ve stejném sezení jako měření – po několika
#: hodinách se mění teplota senzoru i zaschlý film na optice.
STALE_REFERENCE_HOURS = 6.0

#: Klíč s mono rovinou a předpony kanálových rovin v .npz.
MONO_KEYS: Tuple[str, ...] = ("mean_mono", "mean", "bias_mono", "master_bias")
CHANNEL_KEYS: Dict[str, Tuple[str, ...]] = {
    "r": ("mean_red", "mean_r"),
    "g": ("mean_green", "mean_g"),
    "b": ("mean_blue", "mean_b"),
}

_FRAME_KEYS: Tuple[str, ...] = ("frames_mono", "frames", "n_frames")
_CREATED_KEYS: Tuple[str, ...] = ("created_mono", "created", "timestamp")
_NOTE_KEYS: Tuple[str, ...] = ("note_mono", "note", "popis")


class ReferenceError(RuntimeError):
    """Referenci se nepodařilo načíst nebo nesedí na sérii."""


# ---------------------------------------------------------------------------
# Popis jedné reference
# ---------------------------------------------------------------------------

@dataclass
class ReferenceRecord:
    """Jedna reference nalezená ve složce – zatím jen metadata, bez dat."""

    npz_path: str
    created: datetime
    json_path: Optional[str] = None
    frames: int = 0
    describe: str = ""
    note: str = ""
    resolution: Optional[Tuple[int, int]] = None   # (výška, šířka) podle JSON
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.npz_path)

    @property
    def label(self) -> str:
        """Krátký popis do GUI a protokolu."""
        stamp = self.created.strftime("%d.%m.%Y %H:%M:%S")
        parts = [stamp]
        if self.resolution:
            parts.append(f"{self.resolution[1]}×{self.resolution[0]} px")
        if self.frames:
            parts.append(f"{self.frames} snímků")
        return " · ".join(parts)


@dataclass
class ReferencePlanes:
    """Načtená data reference."""

    mono: np.ndarray                                  # float32, v jednotkách senzoru
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    frames: int = 0
    record: Optional[ReferenceRecord] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mono.shape  # type: ignore[return-value]

    @property
    def has_channels(self) -> bool:
        return len(self.channels) >= 3


# ---------------------------------------------------------------------------
# Hledání složky s referencemi
# ---------------------------------------------------------------------------

def find_reference_dir(
    measurement_dir: str,
    extra_roots: Sequence[str] = (),
    levels: int = REFERENCE_SEARCH_LEVELS,
) -> Optional[str]:
    """Najde složku s referencemi pro dané měření.

    Hledá se podsložka ``reference`` nejdřív přímo v měření, pak postupně
    v nadřazených složkách (typicky ``…\\BMS fotky\\reference``) a nakonec
    v ručně zadaných kořenech. Vrací první složku, která opravdu nějakou
    referenci obsahuje – prázdná složka ``reference`` tedy hledání nezastaví.
    """
    candidates: List[str] = []

    current = os.path.abspath(measurement_dir) if measurement_dir else ""
    for _level in range(max(1, levels) + 1):
        if not current:
            break
        for name in REFERENCE_DIR_NAMES:
            candidates.append(os.path.join(current, name))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for root in extra_roots:
        if not root:
            continue
        candidates.append(root)
        for name in REFERENCE_DIR_NAMES:
            candidates.append(os.path.join(root, name))

    seen = set()
    for path in candidates:
        key = os.path.normcase(os.path.normpath(path))
        if key in seen:
            continue
        seen.add(key)
        if os.path.isdir(path) and list_reference_files(path):
            return path
    return None


def list_reference_files(folder: str) -> List[str]:
    """Vrátí .npz soubory ve složce (bez rekurze), seřazené podle názvu."""
    if not os.path.isdir(folder):
        return []
    try:
        entries = os.listdir(folder)
    except OSError:
        return []
    found = [
        os.path.join(folder, name)
        for name in entries
        if name.lower().endswith(".npz") and os.path.isfile(os.path.join(folder, name))
    ]
    found.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return found


# ---------------------------------------------------------------------------
# Čtení metadat
# ---------------------------------------------------------------------------

_JSON_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M:%S")


def _parse_iso(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in _JSON_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _sidecar_json(npz_path: str) -> Optional[str]:
    """Najde JSON se stejným základem názvu jako .npz."""
    candidate = os.path.splitext(npz_path)[0] + ".json"
    return candidate if os.path.isfile(candidate) else None


def read_reference_record(npz_path: str) -> ReferenceRecord:
    """Přečte metadata jedné reference **bez** dekomprese obrazových dat.

    Rozbalit 8megapixelovou rovinu jen kvůli času pořízení by u složky
    s desítkami referencí trvalo déle než celá analýza, proto se čas bere
    z názvu souboru a z JSON popisu; do .npz se sahá až při skutečném načtení.
    """
    json_path = _sidecar_json(npz_path)
    meta: Dict[str, object] = {}
    created: Optional[datetime] = None
    describe = ""
    note = ""
    resolution: Optional[Tuple[int, int]] = None
    frames = 0

    if json_path:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError, UnicodeDecodeError):
            meta = {}

    reference_meta = meta.get("reference") if isinstance(meta.get("reference"), dict) else {}
    if isinstance(reference_meta, dict):
        describe = str(reference_meta.get("describe", "") or "")
        note = str(reference_meta.get("note", "") or "")

    camera = meta.get("camera") if isinstance(meta.get("camera"), dict) else {}
    if isinstance(camera, dict):
        size = camera.get("resolution_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                resolution = (int(size[1]), int(size[0]))   # (výška, šířka)
            except (TypeError, ValueError):
                resolution = None

    # Čas pořízení: název souboru (reference_RRRRMMDD_HHMMSS) je nejspolehlivější,
    # protože ho zapisuje sám záznamový program v okamžiku snímání. JSON pole
    # "saved" je čas *uložení*, tedy o pár sekund později.
    name_stamp = _timestamp_from_name(npz_path)
    if name_stamp is not None:
        created = name_stamp
    if created is None and describe:
        created = _timestamp_from_text(describe)
    if created is None:
        created = _parse_iso(str(meta.get("saved", "")))
    if created is None:
        try:
            created = datetime.fromtimestamp(os.path.getmtime(npz_path))
        except OSError:
            created = datetime.fromtimestamp(0)

    # Počet zprůměrovaných snímků se dá vyčíst z popisu („16 snímků“) levně.
    match = re.search(r"(\d+)\s*sn[íi]m", describe)
    if match:
        frames = int(match.group(1))

    return ReferenceRecord(
        npz_path=npz_path,
        created=created,
        json_path=json_path,
        frames=frames,
        describe=describe,
        note=note,
        resolution=resolution,
        meta=meta if isinstance(meta, dict) else {},
    )


_NAME_TS_RE = re.compile(r"(\d{8})[_-](\d{6})")
_TEXT_TS_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})")


def _timestamp_from_name(path: str) -> Optional[datetime]:
    match = _NAME_TS_RE.search(os.path.basename(path))
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _timestamp_from_text(text: str) -> Optional[datetime]:
    match = _TEXT_TS_RE.search(text or "")
    if not match:
        return None
    day, month, year, hour, minute, second = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def list_references(folder: str) -> List[ReferenceRecord]:
    """Vrátí všechny reference ve složce seřazené od nejstarší po nejnovější."""
    records = [read_reference_record(path) for path in list_reference_files(folder)]
    records.sort(key=lambda rec: (rec.created, rec.name))
    return records


# ---------------------------------------------------------------------------
# Výběr reference podle času měření
# ---------------------------------------------------------------------------

@dataclass
class ReferenceChoice:
    """Výsledek výběru reference pro konkrétní měření."""

    record: Optional[ReferenceRecord]
    gap_s: float = 0.0            # kladné = reference je starší než měření
    taken_after: bool = False     # True = žádná starší nebyla, použila se pozdější
    reason: str = ""              # věta do protokolu

    @property
    def ok(self) -> bool:
        return self.record is not None


def select_reference(
    records: Sequence[ReferenceRecord],
    when: datetime,
    stale_hours: float = STALE_REFERENCE_HOURS,
) -> ReferenceChoice:
    """Vybere referenci pořízenou **co nejtěsněji před** začátkem měření.

    Reference musí vzniknout dřív než měřené snímky – jinak by popisovala
    pozadí, které v době měření ještě neplatilo (mezitím se mohla změnit
    expozice nebo se vyměnilo sklíčko). Když je k dispozici jen reference
    pořízená později, použije se, ale výsledek to hlásí jako upozornění;
    bez reference by analýza neběžela vůbec.
    """
    usable = [rec for rec in records if rec is not None]
    if not usable:
        return ReferenceChoice(record=None, reason="Ve složce s referencemi nebyl nalezen žádný .npz soubor.")

    before = [rec for rec in usable if rec.created <= when]
    if before:
        chosen = max(before, key=lambda rec: rec.created)
        gap = (when - chosen.created).total_seconds()
        reason = f"Reference {chosen.name} ({chosen.label}), pořízená {_human_gap(gap)} před měřením."
        if gap > stale_hours * 3600.0:
            reason += (
                f" Odstup je větší než {stale_hours:g} h – ověřte, že mezitím"
                " nedošlo ke změně expozice ani osvětlení."
            )
        return ReferenceChoice(record=chosen, gap_s=gap, taken_after=False, reason=reason)

    chosen = min(usable, key=lambda rec: rec.created)
    gap = (chosen.created - when).total_seconds()
    reason = (
        f"Žádná reference nebyla pořízena před měřením; použita nejbližší pozdější "
        f"{chosen.name} ({chosen.label}), o {_human_gap(gap)} novější."
    )
    return ReferenceChoice(record=chosen, gap_s=-gap, taken_after=True, reason=reason)


def _human_gap(seconds: float) -> str:
    seconds = abs(float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60.0:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600.0:.1f} h"
    return f"{seconds / 86400.0:.1f} dne"


# ---------------------------------------------------------------------------
# Načtení dat
# ---------------------------------------------------------------------------

def _first_key(archive, keys: Sequence[str]):
    for key in keys:
        if key in archive:
            return archive[key]
    return None


def _as_plane(array: np.ndarray) -> np.ndarray:
    """Ověří tvar roviny a převede ji na float32 **v uložených jednotkách**."""
    plane = np.asarray(array, dtype=np.float32)
    if plane.ndim == 3 and plane.shape[2] == 1:
        plane = plane[:, :, 0]
    if plane.ndim != 2:
        raise ReferenceError(f"Referenční rovina má nepodporovaný tvar {plane.shape}.")
    return plane


def scale_to_adu(plane: np.ndarray, full_scale: float) -> np.ndarray:
    """Převede rovinu reference na jednotnou škálu 0–255 ADU.

    Reference vzniká z týchž snímků jako měření, takže je uložená v **týchž
    jednotkách senzoru** – u 8bitové kamery 0–255, u 12bitové 0–4095. Škála se
    proto nehádá z hodnot v rovině (tmavý bias má nízké maximum, takže by
    12bitový záznam vypadal jako 10bitový), ale bere se ta, kterou aplikace
    zjistila ze snímků série.

    Pojistkou zůstává jen kontrola, že po přepočtu nic nepřeteče – kdyby
    reference přece jen přišla v jiné škále, normalizuje se podle maxima.
    """
    plane = _as_plane(plane)
    scale = max(float(full_scale), 1.0)
    if abs(scale - DISPLAY_FULL_SCALE) > 1e-6:
        plane = plane * np.float32(DISPLAY_FULL_SCALE / scale)

    peak = float(plane.max()) if plane.size else 0.0
    if peak > DISPLAY_FULL_SCALE * 1.05:
        plane = plane * np.float32(DISPLAY_FULL_SCALE / peak)
    return plane


def load_reference(record: ReferenceRecord) -> ReferencePlanes:
    """Načte data reference z .npz.

    Raises:
        ReferenceError: soubor chybí, je poškozený nebo neobsahuje mono rovinu.
    """
    try:
        with np.load(record.npz_path, allow_pickle=False) as archive:
            mono_raw = _first_key(archive, MONO_KEYS)
            channels: Dict[str, np.ndarray] = {}
            for short, keys in CHANNEL_KEYS.items():
                plane = _first_key(archive, keys)
                if plane is not None:
                    channels[short] = _as_plane(plane)

            if mono_raw is None and channels:
                # Barevná reference bez mono roviny: mono se dopočítá z kanálů.
                stack = np.stack([channels[key] for key in ("b", "g", "r") if key in channels], axis=2)
                mono_raw = stack.mean(axis=2, dtype=np.float32)
            if mono_raw is None:
                raise ReferenceError(
                    f"Soubor {record.name} neobsahuje referenční rovinu "
                    f"(hledá se {', '.join(MONO_KEYS)})."
                )

            mono = _as_plane(mono_raw)
            frames_raw = _first_key(archive, _FRAME_KEYS)
            frames = int(frames_raw) if frames_raw is not None else record.frames

            created_raw = _first_key(archive, _CREATED_KEYS)
            if created_raw is not None:
                stamp = _parse_iso(str(created_raw))
                if stamp is not None:
                    record.created = stamp

            note_raw = _first_key(archive, _NOTE_KEYS)
            if note_raw is not None and not record.note:
                record.note = str(note_raw)
    except ReferenceError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ReferenceError(f"Referenci {record.name} nelze načíst: {exc}") from exc

    record.frames = frames or record.frames
    record.resolution = (int(mono.shape[0]), int(mono.shape[1]))
    return ReferencePlanes(
        mono=np.ascontiguousarray(mono, dtype=np.float32),
        channels=channels,
        frames=record.frames,
        record=record,
    )


def series_start_time(paths: Sequence[str]) -> datetime:
    """Čas pořízení prvního snímku série (podle názvu, jinak podle mtime)."""
    stamps: List[datetime] = []
    for path in paths:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        stamps.append(parse_timestamp_from_filename(path, mtime))
    return min(stamps) if stamps else datetime.now()


__all__ = [
    "REFERENCE_DIR_NAMES",
    "STALE_REFERENCE_HOURS",
    "ReferenceChoice",
    "ReferenceError",
    "ReferencePlanes",
    "ReferenceRecord",
    "find_reference_dir",
    "list_reference_files",
    "list_references",
    "load_reference",
    "read_reference_record",
    "scale_to_adu",
    "select_reference",
    "series_start_time",
]
