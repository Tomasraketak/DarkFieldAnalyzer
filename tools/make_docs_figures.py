r"""Vygeneruje obrázky do ``docs/img/`` pro dokumentaci docs/JAK_TO_FUNGUJE.md.

Všechny mezikroky počítá **skutečné analytické jádro** aplikace, ne kreslicí
kód – obrázky proto vždy odpovídají tomu, co program opravdu dělá. Když se
změní algoritmus, stačí skript pustit znovu.

Pozadí scény se bere z referenčního ``.npz`` (klíč ``mean_mono``), aby snímky
vypadaly jako z reálné kamery. Bez reference se použije syntetické pozadí::

    py tools\make_docs_figures.py
    py tools\make_docs_figures.py --reference "C:\...\BMS fotky\reference\reference_20260908_142910.npz"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (  # noqa: E402
    AnalysisParams,
    analyze_prepared_frame,
    classify_components,
    compute_rates_and_phases,
    estimate_noise,
    separate_haze,
)

parser = argparse.ArgumentParser(description="Obrázky do dokumentace")
parser.add_argument("--reference", help="Cesta k reference_*.npz pro realistické pozadí")
parser.add_argument("--out", help="Výstupní složka (výchozí docs/img vedle skriptu)")
args = parser.parse_args()

OUT = args.out or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img")
os.makedirs(OUT, exist_ok=True)

#: Barvy tříd jsou stejné jako v prohlížeči a v grafech (viz exporter.COMPOSITION_LAYERS).
INK, MUTED, SURFACE, GRID = "#1B2A38", "#5A6B7A", "#FCFCFB", "#DDE3E8"
C_CLUSTER, C_POINT, C_FIBER, C_HAZE = "#B03A2E", "#D4900A", "#1D8348", "#2E86C1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

H, W = 400, 700


def load_background() -> np.ndarray:
    """Pozadí scény: reálná reference, jinak syntetická vinětace se zrnem."""
    if args.reference and os.path.isfile(args.reference):
        with np.load(args.reference, allow_pickle=False) as archive:
            for key in ("mean_mono", "mean", "bias_mono"):
                if key in archive:
                    plane = np.asarray(archive[key], dtype=np.float32)
                    print(f"pozadí z reference: {os.path.basename(args.reference)} "
                          f"{plane.shape[1]}×{plane.shape[0]} px")
                    return cv2.resize(plane, (W, H), interpolation=cv2.INTER_AREA)
        print("varování: .npz neobsahuje mono rovinu, použije se syntetické pozadí")

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1.0 - 0.55 * (((xx / W - 0.5) ** 2) + ((yy / H - 0.5) ** 2))
    noise = np.random.default_rng(1).normal(0, 0.6, (H, W)).astype(np.float32)
    print("pozadí: syntetické (spusťte s --reference pro realistické)")
    return np.clip(20.0 * vign + noise, 0, 255).astype(np.float32)


BIAS = load_background()
rng = np.random.default_rng(20260908)


def add_blob(canvas, cx, cy, r, amp):
    y0, y1 = max(0, int(cy - 4 * r)), min(canvas.shape[0], int(cy + 4 * r) + 1)
    x0, x1 = max(0, int(cx - 4 * r)), min(canvas.shape[1], int(cx + 4 * r) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))


def make_scene(fog_adu):
    s = np.zeros((H, W), dtype=np.float32)
    if fog_adu > 0:                       # difuzní opar s prostorovým gradientem
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        s += fog_adu * (0.45 + 0.55 * np.exp(-((xx - 0.32) ** 2 + (yy - 0.28) ** 2) / 0.22))
    for _ in range(45):                   # mikročástice (prach)
        add_blob(s, rng.integers(20, W - 20), rng.integers(20, H - 20),
                 rng.uniform(1.1, 1.6), rng.uniform(45, 85))
    for cx, cy, r in ((180, 110, 8.0), (520, 280, 9.0), (390, 80, 7.0)):
        add_blob(s, cx, cy, r, 170.0)     # velké shluky / kapky
    for x0, y0, ang, ln in ((90, 300, 0.35, 175), (430, 105, -0.7, 150), (250, 350, -0.45, 150)):
        for t in range(ln):               # vlákna pod různými úhly
            px, py = int(x0 + math.cos(ang) * t), int(y0 + math.sin(ang) * t)
            if 0 <= px < W - 2 and 0 <= py < H - 2:
                s[py:py + 2, px:px + 2] += 70.0
    return s


frame = np.clip(BIAS + make_scene(6.2) + rng.normal(0, 1.1, (H, W)).astype(np.float32), 0, 255)
params = AnalysisParams(binning=1, sigma=4.0, haze_threshold=4.0,
                        min_area_px=3, cluster_min_area_px=100)
T0 = datetime(2026, 9, 8, 14, 35, 0)
metrics, masks = analyze_prepared_frame(
    image=frame, bias=BIAS, params=params, index=0, filename="ukazka.png",
    filepath="ukazka.png", timestamp=T0, t0=T0, binning=1, generate_masks=True)
print(f"ukázkový snímek: pokrytí {metrics.total_coverage_pct:.2f} %, "
      f"částic {metrics.total_particle_count} "
      f"({metrics.point_count} bodů / {metrics.cluster_count} shluků / {metrics.fiber_count} vláken), "
      f"práh {metrics.applied_threshold:.2f} ADU, σ {metrics.bg_noise_sigma:.2f}")

# ==========================================================================
# 1) Řetězec zpracování
# ==========================================================================
diff, haze, sharp = masks["diff"], masks["haze"], masks["sharp"]
classified = np.zeros((H, W, 3), dtype=np.uint8)
base_gray = np.clip(frame / 255.0 * 90, 0, 255).astype(np.uint8)
classified[:] = base_gray[..., None]


def paint(mask, hexcolor):
    rgb = tuple(int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    sel = mask > 0
    for c in range(3):
        classified[..., c][sel] = rgb[c]


paint(masks["mask_haze"], C_HAZE)
paint(masks["mask_points"], C_POINT)
paint(masks["mask_clusters"], C_CLUSTER)
paint(masks["mask_fibers"], C_FIBER)

panels = [
    (frame, "1 · Snímek po normalizaci\n(float32, 0–255 ADU)", "gray", (0, 55)),
    (BIAS, "2 · Referenční pozadí (bias)\nvypadá skoro stejně – proto se odečítá", "gray", (0, 55)),
    (diff, "3 · Diference = snímek − bias\nstruktura pozadí zmizela", "gray", (0, 25)),
    (haze, "4 · Nízkofrekvenční složka\n= difuzní zamlžení", "gray", (0, 8)),
    (sharp, "5 · Ostrá složka = diference − opar\nzde se hledají částice", "gray", (0, 25)),
    (masks["mask_total"], "6 · Binární maska nad prahem\nmedián + 4σ (bílá = kontaminace)", "gray", (0, 255)),
]
fig, axes = plt.subplots(2, 4, figsize=(16, 6.2))
for ax, (data, title, cmap, (lo, hi)) in zip(axes.ravel(), panels):
    ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
axes[1, 2].imshow(classified)
axes[1, 2].set_title("7 · Klasifikace objektů\npodle plochy a tvaru", pad=6)
axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

ax = axes[1, 3]
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(handles=[
    Patch(facecolor=C_HAZE, label=f"Difuzní zamlžení – {metrics.haze_coverage_pct:.1f} % plochy"),
    Patch(facecolor=C_POINT, label=f"Mikročástice – {metrics.point_count} ks"),
    Patch(facecolor=C_CLUSTER, label=f"Velké shluky / kapky – {metrics.cluster_count} ks"),
    Patch(facecolor=C_FIBER, label=f"Vlákna / škrábance – {metrics.fiber_count} ks"),
], loc="upper center", frameon=False, fontsize=9.5, handlelength=1.4, labelspacing=0.9)
ax.text(0.5, 0.34, f"8 · Výsledek jednoho snímku", ha="center", transform=ax.transAxes,
        fontsize=10, fontweight="bold", color=INK)
ax.text(0.5, 0.10,
        f"celkové pokrytí {metrics.total_coverage_pct:.2f} %\n"
        f"práh {metrics.applied_threshold:.2f} ADU  ·  šum σ = {metrics.bg_noise_sigma:.2f} ADU\n"
        f"skóre čistoty {metrics.cleanliness_score:.0f} / 100",
        ha="center", transform=ax.transAxes, fontsize=9, color=MUTED, linespacing=1.7)

fig.suptitle("Řetězec zpracování jednoho snímku", fontsize=13, fontweight="bold", color=INK, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(os.path.join(OUT, "01_retezec.png"), dpi=115)
plt.close(fig)
print("  -> 01_retezec.png")

# ==========================================================================
# 2) Odhad šumu a práh – proč se šum měří z rozdílů sousedních pixelů
# ==========================================================================
# Postavíme mediánový bias ze tří snímků a pak analyzujeme jeden z NICH.
# U takového snímku je přes polovinu pixelů diference přesně nulová a klasický
# MAD vyjde téměř nulový – práh spadne pod šum a analýza „najde“ desítky tisíc
# neexistujících částic. Přesně tenhle případ řešil vývoj estimátoru.
bias_frames = [np.clip(BIAS + rng.normal(0, 1.1, (H, W)).astype(np.float32), 0, 255)
               for _ in range(3)]
median_bias = np.median(np.stack(bias_frames), axis=0).astype(np.float32)
self_diff = cv2.subtract(bias_frames[0], median_bias)      # snímek proti sobě samému

sample = self_diff.ravel()
med = float(np.median(sample))
mad_sigma = float(np.median(np.abs(sample - med))) * 1.4826           # naivní odhad
deltas = (self_diff[:, 1:] - self_diff[:, :-1]).ravel()               # použitý odhad
hf_sigma = float(np.median(np.abs(deltas - np.median(deltas)))) * 1.4826 / math.sqrt(2.0)
_, used_sigma = estimate_noise(self_diff)

def count_objects(diff_img, sigma_value):
    thr = max(med + 4.0 * sigma_value, 1.5, 0.5)
    m = (cv2.subtract(diff_img, separate_haze(diff_img)[0]) > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8, ltype=cv2.CV_32S)
    return int((st[1:, cv2.CC_STAT_AREA] >= 3).sum()), thr

n_mad, thr_mad = count_objects(self_diff, mad_sigma)
n_hf, thr_hf = count_objects(self_diff, hf_sigma)
print(f"bias snímek proti sobě: MAD σ={mad_sigma:.3f} → práh {thr_mad:.2f} → {n_mad} „částic“ | "
      f"rozdíly sousedů σ={hf_sigma:.3f} → práh {thr_hf:.2f} → {n_hf} částic "
      f"(estimate_noise vrací {used_sigma:.3f})")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

axL.hist(sharp.ravel(), bins=260, range=(-8, 30), color="#8FA6B8", log=True)
axL.axvline(metrics.applied_threshold, color=C_CLUSTER, lw=2)
axL.text(metrics.applied_threshold + 0.7, 4.5e3,
         f"práh = medián + 4σ\n= {metrics.applied_threshold:.2f} ADU",
         color=C_CLUSTER, fontsize=9.5, fontweight="bold", linespacing=1.5, va="top")
axL.text(14.0, 220, "vpravo od prahu = kontaminace\n(dlouhý ocas jasných pixelů)",
         color=MUTED, fontsize=9, linespacing=1.5)
axL.text(-7.5, 220, "šum pozadí\n(symetrický kolem nuly)",
         color=MUTED, fontsize=9, linespacing=1.5)
axL.set_title("Rozdělení jasu ostré složky – kde leží práh")
axL.set_xlabel("jas ostré složky [ADU nad pozadím]")
axL.set_ylabel("počet pixelů (log)")
axL.grid(axis="y", color=GRID, lw=0.8)
axL.set_axisbelow(True)

labels = ["MAD z celého\nrozdělení", "MAD z rozdílů\nsousedních pixelů\n(použito)"]
values = [n_mad, n_hf]
sigmas = [mad_sigma, hf_sigma]
bars = axR.bar(labels, values, color=[C_CLUSTER, C_FIBER], width=0.5)
axR.set_title("Kolik „částic“ najde analýza na čistém bias snímku")
axR.set_ylabel("nalezených objektů  (správně = 0)")
axR.grid(axis="y", color=GRID, lw=0.8)
axR.set_axisbelow(True)
axR.set_ylim(0, max(values) * 1.45 + 8)
for bar, value, sig in zip(bars, values, sigmas):
    if value == 0:      # nulový sloupec musí být vidět jako značka na základně
        axR.plot([bar.get_x(), bar.get_x() + bar.get_width()], [0, 0],
                 color=C_FIBER, lw=4, solid_capstyle="butt")
    axR.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.06,
             f"{value} objektů\nσ = {sig:.3f} ADU", ha="center", color=INK,
             fontsize=10, fontweight="bold", linespacing=1.5)

per_mpx = n_mad / (H * W / 1e6)
fig.suptitle(
    "Odhad šumu rozhoduje o všem: podceněný šum vyrobí částice tam, kde žádné nejsou\n"
    f"({n_mad} objektů na tomto {W}×{H} snímku ≈ {per_mpx:.0f} na megapixel, "
    f"tedy ≈ {per_mpx * 8.29:.0f} na jednom 4K snímku)",
    fontsize=11.5, fontweight="bold", color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "02_sum_a_prah.png"), dpi=115)
plt.close(fig)
print("  -> 02_sum_a_prah.png")

# ==========================================================================
# 3) Klasifikace tvaru – proč opsaný obdélník selže a momenty ne
# ==========================================================================
from analyzer import _shape_descriptors      # interní, ale pro ukázku je názorné

SH = 150
shapes = []
for angle_deg, name in ((0, "Vlákno 0°"), (45, "Vlákno 45°"), (90, "Vlákno 90°")):
    canvas = np.zeros((SH, SH), dtype=np.uint8)
    a = math.radians(angle_deg)
    for t in range(-52, 53):
        px, py = int(SH / 2 + math.cos(a) * t), int(SH / 2 + math.sin(a) * t)
        cv2.circle(canvas, (px, py), 2, 255, -1)
    shapes.append((canvas, name))

canvas = np.zeros((SH, SH), dtype=np.uint8)
cv2.circle(canvas, (SH // 2, SH // 2), 5, 255, -1)
shapes.append((canvas, "Mikročástice (81 px)"))
canvas = np.zeros((SH, SH), dtype=np.uint8)
cv2.circle(canvas, (SH // 2, SH // 2), 26, 255, -1)
shapes.append((canvas, "Velký shluk (2121 px)"))

fig, axes = plt.subplots(1, 5, figsize=(15, 5.0))
params_shape = AnalysisParams(binning=1, min_area_px=3, cluster_min_area_px=100,
                              fiber_aspect_ratio=2.8, fiber_min_length_px=12)
CLASS_COLOR = {1: C_POINT, 2: C_CLUSTER, 3: C_FIBER}
CLASS_NAME = {1: "mikročástice", 2: "velký shluk", 3: "vlákno"}

for ax, (canvas, name) in zip(axes, shapes):
    n, lab, st, _ = cv2.connectedComponentsWithStats(canvas, connectivity=8, ltype=cv2.CV_32S)
    elong, major = _shape_descriptors(lab, n, st)
    stats_row = st[1]
    bbox_w, bbox_h = float(stats_row[cv2.CC_STAT_WIDTH]), float(stats_row[cv2.CC_STAT_HEIGHT])
    bbox_ratio = max(bbox_w, bbox_h) / max(1.0, min(bbox_w, bbox_h))
    cat = int(classify_components(lab, n, st, params_shape, 1).category_lut[1])

    rgb = np.zeros((SH, SH, 3), dtype=np.uint8)
    rgb[:] = 245
    color = tuple(int(CLASS_COLOR[cat][i:i + 2], 16) for i in (1, 3, 5))
    for c in range(3):
        rgb[..., c][canvas > 0] = color[c]
    ax.imshow(rgb)
    ax.add_patch(plt.Rectangle((stats_row[cv2.CC_STAT_LEFT] - 0.5, stats_row[cv2.CC_STAT_TOP] - 0.5),
                               bbox_w, bbox_h, fill=False, ec=MUTED, lw=1.2, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name, pad=6)
    ok = "✔" if (bbox_ratio >= 2.8) == (cat == 3) else "✘"
    ax.set_xlabel(
        f"opsaný obdélník: {bbox_ratio:.2f}  {ok}\n"
        f"momenty (elipsa): {elong[1]:.2f}\n"
        f"→ {CLASS_NAME[cat]}",
        fontsize=9, color=INK, linespacing=1.7)

fig.suptitle("Protáhlost se počítá z momentů druhého řádu, ne z opsaného obdélníku\n"
             "u vlákna pod 45° dá obdélník poměr ≈ 1 (vypadá jako kulatý shluk), elipsa správně > 10",
             fontsize=11.5, fontweight="bold", color=INK, y=0.985)
fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.24, wspace=0.12)
fig.savefig(os.path.join(OUT, "03_tvary.png"), dpi=115)
plt.close(fig)
print("  -> 03_tvary.png")

# ==========================================================================
# 4) Časová řada, derivace a fáze děje
# ==========================================================================
# Scénář odpovídá typickému měření: dýchne se na sklíčko (zamlžení), to se
# odpaří a po celou dobu se usazuje prach. Fázi určuje SKUTEČNÁ funkce
# compute_rates_and_phases nad metrikami ze skutečné analýzy.
N = 26
series = []
for i in range(N):
    fog = 6.6 * math.exp(-((i - 4) ** 2) / 16.0) if i >= 1 else 0.6
    scene = make_scene(fog)
    for _ in range(3 * i):                       # postupně přibývající prach
        add_blob(scene, rng.integers(15, W - 15), rng.integers(15, H - 15),
                 rng.uniform(1.1, 1.6), rng.uniform(45, 85))
    img = np.clip(BIAS + scene + rng.normal(0, 1.1, (H, W)).astype(np.float32), 0, 255)
    m, _ = analyze_prepared_frame(
        image=img, bias=BIAS, params=params, index=i, filename=f"df_{i:05d}.png",
        filepath=f"df_{i:05d}.png", timestamp=T0 + timedelta(seconds=10 * i), t0=T0,
        binning=1, generate_masks=False)
    series.append(m)
series = compute_rates_and_phases(series)

t = np.array([m.time_s for m in series])
cov = np.array([m.total_coverage_pct for m in series])
hz = np.array([m.haze_coverage_pct for m in series])
cnt = np.array([m.total_particle_count for m in series])
rate = np.array([m.rate_coverage_pct_per_s for m in series])

PHASE_COLOR = {"nárůst / zamlžování": C_CLUSTER, "odpařování / ústup": C_HAZE, "stabilní": MUTED}
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.5, 8.6), sharex=True,
                                    gridspec_kw={"height_ratios": [1.3, 0.85, 0.95]})

ax1.plot(t, cov, color=INK, lw=2, label="Celkové pokrytí", zorder=3)
ax1.plot(t, hz, color=C_HAZE, lw=2, ls="--", label="z toho difuzní zamlžení", zorder=2)
for phase, color in PHASE_COLOR.items():
    sel = [i for i, m in enumerate(series) if m.phase == phase]
    if sel:
        ax1.scatter(t[sel], cov[sel], s=54, color=color, zorder=4,
                    edgecolor=SURFACE, linewidth=2, label=f"fáze: {phase}")
ax1.set_ylabel("pokrytí plochy [%]")
ax1.set_title("Zamlžení přijde a odpaří se, prach zůstává a přibývá")
ax1.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
ax1.set_ylim(-4, max(cov) * 1.42)

ax2.plot(t, cnt, color=C_POINT, lw=2)
ax2.scatter([t[0], t[-1]], [cnt[0], cnt[-1]], s=48, color=C_POINT,
            edgecolor=SURFACE, linewidth=2, zorder=3)
ax2.annotate(f"{cnt[0]} částic", (t[0], cnt[0]), xytext=(6, 10),
             textcoords="offset points", color=INK, fontsize=9, fontweight="bold")
ax2.annotate(f"{cnt[-1]} částic", (t[-1], cnt[-1]), xytext=(-8, -18),
             textcoords="offset points", color=INK, fontsize=9, fontweight="bold", ha="right")
ax2.set_ylabel("počet částic [ks]")
ax2.set_title("Počet pevných částic – na zamlžení nereaguje, roste monotónně")
ax2.set_ylim(0, max(cnt) * 1.3)

ax3.axhline(0, color=GRID, lw=1.2)
thr = max(0.02, 0.05 * float(np.percentile(np.abs(rate), 95)))
ax3.axhspan(-thr, thr, color=MUTED, alpha=0.15)
ax3.plot(t, rate, color=C_FIBER, lw=2)
ax3.text(t[-1], thr * 1.25, f"pásmo „stabilní“ (±{thr:.3f} %/s)  ", va="bottom", ha="right",
         color=MUTED, fontsize=9)
ax3.set_xlabel("čas od začátku měření [s]")
ax3.set_ylabel("d(pokrytí)/dt  [%/s]")
ax3.set_title("Derivace pokrytí z lokální lineární regrese – její znaménko určuje fázi")

for ax in (ax1, ax2, ax3):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_faze.png"), dpi=115)
plt.close(fig)
print("  -> 04_faze.png")
print("fáze:", {p: sum(1 for m in series if m.phase == p) for p in PHASE_COLOR})
print(f"pokrytí {cov[0]:.2f} % → max {cov.max():.2f} % → konec {cov[-1]:.2f} %; "
      f"částic {cnt[0]} → {cnt[-1]}")

# ==========================================================================
# 5) Drift scény: co způsobí a co s tím udělá zarovnání
# ==========================================================================
from alignment import (FrameShift, detect_stars, estimate_shift,  # noqa: E402
                       find_hot_pixels, repair_hot_pixels, warp_to_anchor)

DRIFT = (7.0, -5.0)
static_specs = [(rng.uniform(50, W - 50), rng.uniform(50, H - 50),
                 rng.uniform(1.8, 3.0), rng.uniform(80, 190)) for _ in range(90)]
hot_specs = [(int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20))) for _ in range(35)]


#: Vinětace patří OPTICE – s driftem sklíčka se neposouvá (na rozdíl od jeho obsahu).
_gy, _gx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = 1.0 - 1.6 * (((_gx / W - 0.5) ** 2) + ((_gy / H - 0.5) ** 2))


def drift_scene(dx=0.0, dy=0.0, extra=0, noise=1.0, seed=0):
    """Fyzikálně poctivá scéna s driftem.

    Posouvá se **celý obsah sklíčka** – jeho struktura i všechny částice.
    Vinětace optiky a vadné pixely senzoru zůstávají pevné vůči kameře.
    """
    slide = BIAS.copy()
    for cx, cy, radius, amp in static_specs:
        add_blob(slide, cx, cy, radius, amp)
    fresh = np.random.default_rng(4242)
    for _ in range(extra):
        add_blob(slide, fresh.uniform(50, W - 50), fresh.uniform(50, H - 50),
                 fresh.uniform(1.6, 2.4), fresh.uniform(70, 150))
    if dx or dy:
        slide = cv2.warpAffine(slide, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    image = slide * VIGNETTE + np.random.default_rng(seed).normal(0, noise, (H, W)).astype(np.float32)
    for hx, hy in hot_specs:
        image[hy, hx] += 210.0
    return np.clip(image, 0, 255)


reference_frame = drift_scene(seed=0)
drifted_frame = drift_scene(*DRIFT, extra=10, seed=1)

hot = find_hot_pixels(reference_frame)
anchor_stars = detect_stars(repair_hot_pixels(reference_frame, hot), hot_mask=hot)
measured = estimate_shift(anchor_stars, detect_stars(repair_hot_pixels(drifted_frame, hot), hot_mask=hot))
aligned_frame = warp_to_anchor(repair_hot_pixels(drifted_frame, hot), measured)
clean_reference = repair_hot_pixels(reference_frame, hot)
print(f"drift: skutečnost ({DRIFT[0]:+.2f},{DRIFT[1]:+.2f}) → naměřeno "
      f"({measured.dx:+.2f},{measured.dy:+.2f}) z {measured.matched} částic")


def detections(frame, reference):
    diff = cv2.subtract(frame, reference)
    haze, _small = separate_haze(diff)
    sharp = cv2.subtract(diff, haze)
    median, sigma = estimate_noise(sharp)
    mask = (sharp > max(median + 4.0 * sigma, 1.5)).astype(np.uint8)
    count, labels, stats, _c = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 3)
    keep = keep[keep != 0]
    return diff, len(keep)


diff_bad, n_bad = detections(drifted_frame, clean_reference)
diff_good, n_good = detections(aligned_frame, clean_reference)
print(f"detekcí: bez zarovnání {n_bad}, se zarovnáním {n_good} (skutečně přibylo 10)")

fig = plt.figure(figsize=(15.5, 7.4))
grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.32, wspace=0.16)

ax = fig.add_subplot(grid[0, 0])
ax.imshow(reference_frame, cmap="gray", vmin=0, vmax=60)
for point in anchor_stars.points[:60]:
    ax.add_patch(plt.Circle((point[0], point[1]), 9, fill=False, ec=C_FIBER, lw=1.1))
ax.set_title(f"Souhvězdí v kotvě\n{anchor_stars.count} zřetelných částic", pad=6)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(grid[0, 1])
ax.imshow(np.abs(diff_bad), cmap="gray", vmin=0, vmax=40)
ax.set_title(f"|snímek − reference| BEZ zarovnání\nkaždá statická částice = dipól → "
             f"{n_bad} detekcí", pad=6, color=C_CLUSTER)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(grid[0, 2])
ax.imshow(np.abs(diff_good), cmap="gray", vmin=0, vmax=40)
ax.set_title(f"|snímek − reference| SE zarovnáním\nzůstane jen skutečný přírůstek → "
             f"{n_good} detekcí", pad=6, color=C_FIBER)
ax.set_xticks([]); ax.set_yticks([])

# --- jak roste chyba s velikostí driftu ---------------------------------
ax = fig.add_subplot(grid[1, :])
shifts = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
without, with_align = [], []
for value in shifts:
    frame = drift_scene(value, -value * 0.7, extra=10, seed=2)
    without.append(detections(frame, clean_reference)[1])
    clean = repair_hot_pixels(frame, hot)
    shift = estimate_shift(anchor_stars, detect_stars(clean, hot_mask=hot))
    with_align.append(detections(warp_to_anchor(clean, shift), clean_reference)[1])

magnitudes = [math.hypot(v, v * 0.7) for v in shifts]
ax.plot(magnitudes, without, color=C_CLUSTER, lw=2, marker="o", ms=7,
        markeredgecolor=SURFACE, markeredgewidth=2, label="bez zarovnání")
ax.plot(magnitudes, with_align, color=C_FIBER, lw=2, marker="o", ms=7,
        markeredgecolor=SURFACE, markeredgewidth=2, label="se zarovnáním")
ax.axhline(10, color=MUTED, ls="--", lw=1.4)
ax.text(magnitudes[-1], 11, "skutečný počet nových částic  ", ha="right", va="bottom",
        color=MUTED, fontsize=9)
ax.set_xlabel("drift scény [px]")
ax.set_ylabel("nalezených objektů")
ax.set_title("Chyba roste už od jednoho pixelu – a zarovnání ji drží na skutečné hodnotě")
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)

fig.suptitle("Drift sklíčka: proč se snímky musí srovnat", fontsize=13, fontweight="bold",
             color=INK, y=0.985)
fig.savefig(os.path.join(OUT, "05_drift.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("  -> 05_drift.png")
print("  detekce podle driftu:", list(zip([round(m,1) for m in magnitudes], without, with_align)))
