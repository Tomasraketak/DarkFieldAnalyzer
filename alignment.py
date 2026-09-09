r"""Zarovnání snímků podle „souhvězdí“ prachových částic na sklíčku.

Proč to vůbec je potřeba
------------------------

Během dlouhého měření se sklíčko nebo kamera posune o jednotky až desítky
pixelů. Pro analýzu je to zásadní problém, i když posun vypadá zanedbatelně:

Referenční pozadí (bias) obsahuje **statické částice** – prach, který na
sklíčku a na optice leží od začátku. Když se scéna posune, tyto částice se
v odečtu ``snímek − bias`` přestanou krýt. Místo nuly po nich zůstane
**dipól**: kladný půlměsíc tam, kde částice je teď, a záporný tam, kde byla
v referenci. Kladná půlka projde prahem a analýza ji započítá jako *novou*
kontaminaci. Stačí posun o dva pixely a na snímku se „objeví“ dvojnásobek
částic, které tam ve skutečnosti celou dobu byly.

Jak se posun měří
-----------------

Prach usazený na sklíčku se chová jako **hvězdné pole**: je ho hodně, je
jasný, drží pevnou vzájemnou polohu a s posunem sklíčka se posouvá celý
najednou. Postup je proto stejný jako v astrometrii:

1. **Detekce hvězd** – na snímku bez nízkofrekvenčního pozadí se najdou
   jasné kompaktní objekty a spočítá se jejich těžiště s přesností na
   desetiny pixelu (:func:`detect_stars`).
2. **Hlasování o posunu** – pro každou dvojici (hvězda v kotvě, hvězda ve
   snímku) se hlasuje pro jejich rozdíl. Skutečný posun dostane tolik hlasů,
   kolik je společných hvězd; náhodné dvojice se rozptýlí (:func:`estimate_shift`).
3. **Zpřesnění** – ke každé hvězdě kotvy se přiřadí nejbližší hvězda snímku
   a posun se dopočítá jako medián přes tyto páry. Z nich se volitelně určí
   i pootočení (Umeyama bez změny měřítka).

Hlasovací schéma je záměrně zvolené místo párování „nejbližší soused“:
funguje i tehdy, když je posun větší než typická vzdálenost mezi částicemi,
a nevadí mu, že část hvězd mezitím přibyla nebo zmizela.

Pojistka proti horkým pixelům
-----------------------------

Vadné pixely senzoru vypadají jako velmi jasné hvězdy, ale **nepohybují se**
se scénou. Kdyby se dostaly do souhvězdí, hlasovaly by pro nulový posun a při
malém počtu skutečných částic by mohly přebít správný výsledek. Brání se jim
dvakrát:

* **Tvarem.** Horký pixel je ostrý bod, kdežto skutečná částice je rozmazaná
  bodovou rozptylovou funkcí optiky přes několik pixelů. Kandidát, jehož okolí
  není zvednuté (viz ``peak_ratio`` v :func:`detect_stars`), se zahodí.
* **Polohou.** Z referenčního pozadí se předem sestaví maska horkých pixelů
  (:func:`find_hot_pixels`) a ta se z hledání vyloučí i s okolím.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from imageops import estimate_noise, separate_haze

#: Kolik nejjasnějších hvězd se nechá v souhvězdí.
DEFAULT_STAR_COUNT = 100

#: Práh detekce hvězdy jako násobek šumu. Vyšší než u vlastní analýzy –
#: pro zarovnání chceme jen jednoznačné částice, ne všechno na hranici šumu.
DEFAULT_STAR_SIGMA = 6.0

#: Největší uvažovaný posun v pixelech (po binningu). Nad tuto mez se
#: nehledá, aby se hlasování nezaplnilo náhodnými dvojicemi.
DEFAULT_MAX_SHIFT = 60

#: Kolik hvězd se musí spárovat, aby byl posun považovaný za spolehlivý.
DEFAULT_MIN_MATCHES = 8

#: Minimální plocha hvězdy v pixelech. Menší objekt je šum nebo vadný pixel.
MIN_STAR_AREA_PX = 4

#: Maximální plocha hvězdy. Rozlehlé skvrny (kapky, opar) mají nepřesné
#: těžiště a při odpařování mění tvar – na zarovnání se nehodí.
MAX_STAR_AREA_PX = 4000

#: Poměr jasu jádra k okolnímu prstenci, nad kterým je objekt považovaný
#: za horký pixel (skutečná částice je rozmazaná optikou).
HOT_PEAK_RATIO = 3.0

#: Jak nízko musí být nejjasnější soused vůči pixelu, aby šlo o vadný pixel.
#: Nejostřejší zobrazitelná částice (σ ≈ 1 px) má souseda na 0,61 své výšky,
#: takže 0,35 je bezpečně pod ní a zároveň vysoko nad nulou vadného pixelu.
HOT_NEIGHBOUR_RATIO = 0.35

#: Tolerance párování hvězd při zpřesnění posunu [px].
MATCH_TOLERANCE_PX = 1.5

#: Rezerva k naměřenému driftu při automatickém ořezu [px].
CROP_MARGIN_PX = 3.0

#: Nejmenší podíl plochy, na který se smí ořezat.
MIN_CROP_FRACTION = 0.5


class AlignmentError(RuntimeError):
    """Zarovnání nelze provést (málo hvězd, nečitelná kotva…)."""


# ---------------------------------------------------------------------------
# Datové struktury
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarField:
    """Souhvězdí nalezené na jednom snímku."""

    points: np.ndarray                 # (N, 2) float32, souřadnice (x, y)
    flux: np.ndarray                   # (N,) integrovaný jas nad pozadím
    shape: Tuple[int, int]             # rozměr obrazu, ve kterém se hledalo

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


@dataclass
class FrameShift:
    """Posun jednoho snímku vůči kotvě (v pixelech geometrie analýzy)."""

    dx: float = 0.0
    dy: float = 0.0
    angle_deg: float = 0.0
    matched: int = 0                   # počet spárovaných hvězd
    rms_px: float = 0.0                # zbytková odchylka párů
    ok: bool = False
    reason: str = ""

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def is_identity(self) -> bool:
        return abs(self.dx) < 1e-3 and abs(self.dy) < 1e-3 and abs(self.angle_deg) < 1e-4

    def matrix(self) -> np.ndarray:
        """Afinní matice 2×3, která snímek srovná na kotvu."""
        angle = math.radians(self.angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Otočení kolem počátku + posun; při nulovém úhlu jde o čistou translaci.
        return np.array([[cos_a, -sin_a, -self.dx],
                         [sin_a, cos_a, -self.dy]], dtype=np.float32)


@dataclass
class AlignmentModel:
    """Vše, co je potřeba k srovnání celé série na společnou soustavu."""

    anchor: StarField
    anchor_path: str
    binning: int
    max_shift_px: int
    min_matches: int
    star_count: int
    star_sigma: float
    hot_mask: Optional[np.ndarray] = None
    hot_pixel_count: int = 0
    #: Bezpečná oblast v pixelech **plného rozlišení** jako (x, y, š, v).
    crop: Optional[Tuple[int, int, int, int]] = None
    crop_fraction: float = 1.0
    #: Posuny naměřené na vzorku snímků (pro odhad driftu a ořezu).
    sampled: List[Tuple[str, FrameShift]] = field(default_factory=list)
    #: Posun samotného referenčního pozadí vůči kotvě.
    bias_shift: Optional[FrameShift] = None
    estimate_rotation: bool = True
    #: ``False`` = model se sestavil, ale zarovnat nejde (málo částic).
    usable: bool = True
    #: Skutečně naměřený největší drift přes celou sérii [px plného rozlišení].
    measured_drift_px: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def crop_margin_px(self) -> Optional[float]:
        """Kolik pixelů driftu ořez ještě snese (v px plného rozlišení)."""
        if self.crop is None:
            return None
        return min(self.crop[0], self.crop[1])

    @property
    def sampled_drift_px(self) -> float:
        """Největší naměřený posun na vzorku, v px plného rozlišení."""
        usable = [s.magnitude for _p, s in self.sampled if s.ok]
        return max(usable) * self.binning if usable else 0.0

    @property
    def sampled_ok(self) -> int:
        return sum(1 for _p, s in self.sampled if s.ok)

    def shift_for(self, stars: Optional[StarField]) -> FrameShift:
        """Spočítá posun daného souhvězdí vůči kotvě."""
        if not self.usable:
            return FrameShift(reason="zarovnání není k dispozici")
        if stars is None or stars.count == 0:
            return FrameShift(reason="na snímku se nenašly žádné použitelné částice")
        return estimate_shift(
            self.anchor, stars, max_shift=self.max_shift_px,
            min_matches=self.min_matches, estimate_rotation=self.estimate_rotation,
        )

    def measure(self, image: np.ndarray) -> FrameShift:
        """Najde souhvězdí v obraze (už po binningu) a vrátí jeho posun vůči kotvě."""
        if not self.usable:
            return FrameShift(reason="zarovnání není k dispozici")
        stars = detect_stars(
            image, hot_mask=self.hot_mask,
            star_count=self.star_count, sigma=self.star_sigma,
        )
        return self.shift_for(stars)


# ---------------------------------------------------------------------------
# Horké pixely
# ---------------------------------------------------------------------------

#: Jádro pro maximum z osmi sousedů (střed je vynechaný).
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def find_hot_pixels(image: np.ndarray, sigma_mult: float = 8.0,
                    neighbour_ratio: float = HOT_NEIGHBOUR_RATIO) -> np.ndarray:
    """Najde vadné (horké) pixely senzoru.

    Kritérium je **poměrové, ne absolutní**: rozhoduje, jak vysoko nad pozadím
    je nejjasnější z osmi sousedů *v poměru* k samotnému pixelu.

    * Vadný pixel svítí sám za sebe, jeho sousedé jsou na úrovni pozadí,
      takže poměr je blízký nule.
    * Skutečná částice je rozmazaná optikou. I ta nejostřejší, jakou přístroj
      dokáže zobrazit (Gaussova stopa se σ ≈ 1 px), má souseda na
      ``exp(−1/2) ≈ 0,61`` své výšky.

    Absolutní práh („pixel je o 8σ jasnější než soused“) by nestačil: ostrá
    částice o jasu 190 ADU má souseda o 27 ADU níž, což je při σ ≈ 1 ADU
    hluboko nad prahem – a částice by se označila za vadný pixel.
    """
    data = np.ascontiguousarray(image, dtype=np.float32)
    if data.size == 0:
        return np.zeros(data.shape, dtype=bool)

    haze, _small = separate_haze(data)
    excess = cv2.subtract(data, haze)              # jas nad lokálním pozadím
    neighbour_max = cv2.dilate(excess, _NEIGHBOUR_KERNEL)

    _median, sigma = estimate_noise(excess)
    bright = excess > max(sigma_mult * sigma, 1.0)
    isolated = neighbour_max < neighbour_ratio * excess
    return bright & isolated


def repair_hot_pixels(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Nahradí vadné pixely mediánem okolí.

    Dělá se to **před** srovnáním driftu, tedy ještě v soustavě senzoru.
    Horké pixely jsou totiž pevné vůči kameře, kdežto srovnání posouvá obsah
    sklíčka – po srovnání by se s referencí přestaly krýt a zůstala by po
    každém z nich dvojice světlý/tmavý bod. Vadný pixel navíc není
    kontaminace, takže jeho odstranění nic naměřeného neubírá.
    """
    if mask is None or not mask.any() or mask.shape != image.shape[:2]:
        return image
    data = np.ascontiguousarray(image, dtype=np.float32)
    return np.where(mask, cv2.medianBlur(data, 3), data).astype(np.float32)


# ---------------------------------------------------------------------------
# Detekce hvězd
# ---------------------------------------------------------------------------

def detect_stars(
    image: np.ndarray,
    hot_mask: Optional[np.ndarray] = None,
    star_count: int = DEFAULT_STAR_COUNT,
    sigma: float = DEFAULT_STAR_SIGMA,
    min_area: int = MIN_STAR_AREA_PX,
    max_area: int = MAX_STAR_AREA_PX,
) -> StarField:
    """Najde nejjasnější kompaktní částice a vrátí jejich subpixelová těžiště.

    Hledá se v obraze **bez nízkofrekvenční složky** – vinětace ani difuzní
    zamlžení tak nemají na detekci vliv a stejné částice se najdou i na
    snímku, který mezitím zamlžil dech.
    """
    data = np.ascontiguousarray(image, dtype=np.float32)
    if data.ndim != 2 or data.size == 0:
        return StarField(np.zeros((0, 2), np.float32), np.zeros(0, np.float32), (0, 0))

    haze, _small = separate_haze(data)
    sharp = cv2.subtract(data, haze)

    median, noise = estimate_noise(sharp)
    threshold = max(median + sigma * noise, 0.5)
    mask = (sharp > threshold).astype(np.uint8)
    if hot_mask is not None and hot_mask.shape == mask.shape:
        mask[hot_mask] = 0

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    if n_labels <= 1:
        return StarField(np.zeros((0, 2), np.float32), np.zeros(0, np.float32), data.shape)

    areas = stats[:, cv2.CC_STAT_AREA]
    candidates = np.flatnonzero((areas >= min_area) & (areas <= max_area))
    candidates = candidates[candidates != 0]
    if candidates.size == 0:
        return StarField(np.zeros((0, 2), np.float32), np.zeros(0, np.float32), data.shape)

    # Těžiště vážené jasem – přesnější než geometrický střed komponenty.
    flat_labels = labels.ravel()
    nz = np.flatnonzero(flat_labels)
    lab = flat_labels[nz]
    weights = np.maximum(sharp.ravel()[nz], 0.0).astype(np.float64)
    ys, xs = np.divmod(nz, data.shape[1])

    total = np.bincount(lab, weights=weights, minlength=n_labels)
    total_safe = np.maximum(total, 1e-6)
    cx = np.bincount(lab, weights=weights * xs, minlength=n_labels) / total_safe
    cy = np.bincount(lab, weights=weights * ys, minlength=n_labels) / total_safe

    keep = [index for index in candidates if _is_resolved(sharp, cx[index], cy[index])]
    if not keep:
        return StarField(np.zeros((0, 2), np.float32), np.zeros(0, np.float32), data.shape)

    keep_array = np.asarray(keep, dtype=np.int64)
    order = np.argsort(total[keep_array])[::-1][: max(1, int(star_count))]
    chosen = keep_array[order]

    points = np.stack([cx[chosen], cy[chosen]], axis=1).astype(np.float32)
    return StarField(points=points, flux=total[chosen].astype(np.float32), shape=data.shape)


def _is_resolved(sharp: np.ndarray, cx: float, cy: float) -> bool:
    """Ověří, že objekt je rozmazaný optikou, a ne ostrý vadný pixel.

    Porovná jádro 3×3 s prstencem 5×5 kolem něj. U rozostřené částice je
    prstenec jen o málo slabší než jádro; u horkého pixelu je prakticky na
    úrovni pozadí.
    """
    h, w = sharp.shape[:2]
    x, y = int(round(cx)), int(round(cy))
    if not (2 <= x < w - 2 and 2 <= y < h - 2):
        return False

    patch = sharp[y - 2 : y + 3, x - 2 : x + 3]
    core = float(patch[1:4, 1:4].mean())
    ring = float((patch.sum() - patch[1:4, 1:4].sum()) / 16.0)
    if core <= 0.0:
        return False
    return core <= HOT_PEAK_RATIO * max(ring, 1e-3)


# ---------------------------------------------------------------------------
# Odhad posunu
# ---------------------------------------------------------------------------

def estimate_shift(
    anchor: StarField,
    other: StarField,
    max_shift: int = DEFAULT_MAX_SHIFT,
    min_matches: int = DEFAULT_MIN_MATCHES,
    tolerance: float = MATCH_TOLERANCE_PX,
    estimate_rotation: bool = True,
) -> FrameShift:
    """Určí posun (a případně pootočení) souhvězdí ``other`` vůči ``anchor``.

    Vrací posun ve smyslu „o kolik je ``other`` posunuté oproti kotvě“, takže
    srovnání se provede posunem o ``−dx, −dy``.
    """
    if anchor.count == 0 or other.count == 0:
        return FrameShift(reason="prázdné souhvězdí")
    if anchor.count < min_matches or other.count < min_matches:
        return FrameShift(
            reason=f"nalezených částic: kotva {anchor.count}, snímek {other.count} "
                   f"(potřeba aspoň {min_matches})")

    # --- 1. hlasování o hrubém posunu -------------------------------------
    dx = other.points[:, None, 0] - anchor.points[None, :, 0]
    dy = other.points[:, None, 1] - anchor.points[None, :, 1]
    inside = (np.abs(dx) <= max_shift) & (np.abs(dy) <= max_shift)
    if not inside.any():
        return FrameShift(reason=f"žádná dvojice částic do vzdálenosti {max_shift} px")

    votes_x = dx[inside]
    votes_y = dy[inside]
    bins = 2 * int(max_shift) + 1
    span = (-max_shift - 0.5, max_shift + 0.5)
    hist, _xe, _ye = np.histogram2d(votes_x, votes_y, bins=bins, range=(span, span))
    peak = np.unravel_index(int(np.argmax(hist)), hist.shape)
    coarse_x = peak[0] - max_shift
    coarse_y = peak[1] - max_shift

    # --- 2. zpřesnění přes nejbližší souseda ------------------------------
    shifted = other.points - np.array([coarse_x, coarse_y], dtype=np.float32)
    pairs = _nearest_pairs(anchor.points, shifted, tolerance=max(tolerance, 1.0))
    if len(pairs) < min_matches:
        return FrameShift(
            dx=float(coarse_x), dy=float(coarse_y), matched=len(pairs),
            reason=f"spárovaných částic: {len(pairs)} (potřeba {min_matches})")

    anchor_idx = np.array([a for a, _b in pairs], dtype=np.int64)
    other_idx = np.array([b for _a, b in pairs], dtype=np.int64)
    src = other.points[other_idx].astype(np.float64)
    dst = anchor.points[anchor_idx].astype(np.float64)

    delta = src - dst
    fine_x = float(np.median(delta[:, 0]))
    fine_y = float(np.median(delta[:, 1]))

    angle_deg = 0.0
    if estimate_rotation and len(pairs) >= 12:
        angle_deg = _rotation_between(src, dst)
        if abs(angle_deg) > 5.0:            # nedůvěryhodné, scéna se neotáčí o víc
            angle_deg = 0.0

    if abs(angle_deg) > 1e-4:
        rotated = _apply_rotation(src, angle_deg)
        residual = rotated - dst
        fine_x = float(np.median(residual[:, 0]))
        fine_y = float(np.median(residual[:, 1]))
        errors = residual - np.array([fine_x, fine_y])
    else:
        angle_deg = 0.0
        errors = delta - np.array([fine_x, fine_y])

    rms = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
    return FrameShift(dx=fine_x, dy=fine_y, angle_deg=angle_deg,
                      matched=len(pairs), rms_px=rms, ok=True)


def _nearest_pairs(anchor_points: np.ndarray, shifted: np.ndarray,
                   tolerance: float) -> List[Tuple[int, int]]:
    """Spáruje každou hvězdu kotvy s nejbližší hvězdou snímku (vzájemně jednoznačně)."""
    if anchor_points.size == 0 or shifted.size == 0:
        return []

    diff = anchor_points[:, None, :] - shifted[None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=2))

    pairs: List[Tuple[int, int]] = []
    used_other = set()
    for anchor_index in np.argsort(distance.min(axis=1)):
        candidate = int(np.argmin(distance[anchor_index]))
        if candidate in used_other:
            continue
        if distance[anchor_index, candidate] > tolerance:
            continue
        used_other.add(candidate)
        pairs.append((int(anchor_index), candidate))
    return pairs


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> float:
    """Úhel otočení mezi dvěma sadami bodů (Umeyama bez změny měřítka) [°]."""
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)
    covariance = dst_centered.T @ src_centered
    u_mat, _s, vt_mat = np.linalg.svd(covariance)
    correction = np.diag([1.0, float(np.sign(np.linalg.det(u_mat @ vt_mat)))])
    rotation = u_mat @ correction @ vt_mat
    return float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))


def _apply_rotation(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return points @ matrix.T


# ---------------------------------------------------------------------------
# Srovnání obrazu
# ---------------------------------------------------------------------------

#: Interpolace při srovnání. Volba **není** kosmetická – viz docstring níže.
WARP_INTERPOLATION = cv2.INTER_LANCZOS4


def warp_to_anchor(image: np.ndarray, shift: FrameShift,
                   fill: Optional[float] = None) -> np.ndarray:
    """Posune (a případně pootočí) snímek do soustavy kotvy.

    Posun se provádí **subpixelově**: zaokrouhlení na celé pixely by nechalo
    až půl pixelu neshody, a to u ostré částice stačí na výrazný dipól
    v odečtu pozadí.

    Interpolace je **Lanczos** a to je zásadní volba. Srovnaný snímek se
    porovnává s referencí, která interpolací neprošla, takže rozmazání
    způsobené interpolací se projeví jako světlý prstenec kolem každé
    statické částice – tedy přesně jako falešná kontaminace.

    Změřeno na poli 120 částic posunutém o (9,02; −6,53) px (scéna vykreslená
    analyticky, aby do měření nevstoupila cizí interpolace):

    ================== ====================
    srovnání           falešných detekcí
    ================== ====================
    žádné (drift zůstane)  110
    nejbližší soused       112
    bilineární              37
    bikubická               10
    **Lanczos**              **3**
    ================== ====================

    Nejbližší soused nepomůže vůbec – neumí subpixelový posun. Lanczosovo
    jádro (8×8) zachovává spektrum obrazu nejlépe, proto se používá i za cenu
    vyšší výpočetní náročnosti (na snímku 1920×1080 asi 47 ms).

    Okraj se vyplní mediánem snímku, aby na hraně nevznikl umělý přechod –
    ta oblast se stejně ořezává.
    """
    if not shift.ok or shift.is_identity:
        return image

    data = np.ascontiguousarray(image, dtype=np.float32)
    border = float(np.median(data)) if fill is None else float(fill)
    height, width = data.shape[:2]
    return cv2.warpAffine(
        data, shift.matrix(), (width, height),
        flags=WARP_INTERPOLATION, borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


# ---------------------------------------------------------------------------
# Bezpečný ořez
# ---------------------------------------------------------------------------

def safe_crop_rect(
    full_shape: Tuple[int, int],
    drift_px: float,
    mode: str = "auto",
    fraction: float = 0.90,
    margin: float = CROP_MARGIN_PX,
) -> Tuple[Tuple[int, int, int, int], float]:
    """Spočítá středový výřez, do kterého nezasahuje neplatný okraj.

    Po srovnání chybí u každého snímku pruh na té straně, odkud se posunul.
    Ořezává se **stejným podílem v obou osách**, takže výřez má stejný poměr
    stran jako originál.

    ``mode="auto"`` ořeže jen tolik, kolik naměřený drift vyžaduje (plus
    rezerva), ``mode="fixed"`` vždy na zadaný podíl. Vrací ``((x, y, š, v),
    skutečný podíl)`` v pixelech plného rozlišení.
    """
    height, width = int(full_shape[0]), int(full_shape[1])
    if height <= 0 or width <= 0:
        return (0, 0, max(width, 1), max(height, 1)), 1.0

    fraction = min(max(float(fraction), MIN_CROP_FRACTION), 1.0)
    if mode == "fixed":
        used = fraction
    else:
        needed = 2.0 * (abs(float(drift_px)) + float(margin))
        used = 1.0 - needed / float(min(height, width))
        used = min(max(used, MIN_CROP_FRACTION), 1.0)

    new_w = max(16, int(round(width * used)))
    new_h = max(16, int(round(height * used)))
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    return (x, y, new_w, new_h), used


def intersect_roi(
    first: Optional[Tuple[int, int, int, int]],
    second: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Průnik dvou obdélníků (x, y, š, v); ``None`` znamená „celý snímek“."""
    if first is None:
        return second
    if second is None:
        return first

    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[0] + first[2], second[0] + second[2])
    y2 = min(first[1] + first[3], second[1] + second[3])
    if x2 <= x1 or y2 <= y1:
        raise AlignmentError(
            "Zvolený výřez (ROI) leží mimo oblast platnou po zarovnání snímků.")
    return (x1, y1, x2 - x1, y2 - y1)


__all__ = [
    "AlignmentError",
    "AlignmentModel",
    "DEFAULT_MAX_SHIFT",
    "DEFAULT_MIN_MATCHES",
    "DEFAULT_STAR_COUNT",
    "DEFAULT_STAR_SIGMA",
    "FrameShift",
    "StarField",
    "detect_stars",
    "estimate_shift",
    "find_hot_pixels",
    "repair_hot_pixels",
    "intersect_roi",
    "safe_crop_rect",
    "warp_to_anchor",
]
