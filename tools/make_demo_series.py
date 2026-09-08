"""Generátor syntetické série snímků z temného pole.

Slouží ke dvěma věcem:

* rychlé vyzkoušení aplikace bez reálných dat (``py tools/make_demo_series.py``),
* jako zdroj dat pro automatické testy, kde je pravda o snímku známá předem.

Simuluje se realistický průběh měření: čisté sklíčko → zapaření (opar rychle
naroste) → odpařování (opar mizí) → usazený prach, který zůstává.

Použití::

    py tools/make_demo_series.py --out "C:\\Users\\...\\demo" --frames 40
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frameio import imwrite_unicode  # noqa: E402


def synth_frame(
    shape: Tuple[int, int],
    rng: np.random.Generator,
    haze_amplitude: float = 0.0,
    particles: int = 0,
    fibers: int = 0,
    clusters: int = 0,
    noise_sigma: float = 2.0,
    hot_pixels: int = 30,
    seed_static: int = 12345,
) -> np.ndarray:
    """Vytvoří jeden syntetický snímek v temném poli (uint8).

    Statické vady (horké pixely, vinětace) jsou pro celou sérii stejné – právě
    ty má odečet biasu odstranit.
    """
    h, w = shape
    img = np.zeros((h, w), dtype=np.float32)

    # Statické pozadí: mírná vinětace + horké pixely senzoru (stejné v každém snímku)
    static_rng = np.random.default_rng(seed_static)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img += 3.0 * np.exp(-(((xx - w / 2) / (w * 0.7)) ** 2 + ((yy - h / 2) / (h * 0.7)) ** 2))
    for _ in range(hot_pixels):
        y = int(static_rng.integers(0, h))
        x = int(static_rng.integers(0, w))
        img[y, x] += static_rng.uniform(120, 220)

    # Difuzní opar: hladká nízkofrekvenční složka
    if haze_amplitude > 0:
        blob = np.exp(-(((xx - w * 0.55) / (w * 0.45)) ** 2 + ((yy - h * 0.5) / (h * 0.45)) ** 2))
        img += haze_amplitude * blob

    # Bodové mikročástice
    for _ in range(particles):
        y = int(rng.integers(3, h - 3))
        x = int(rng.integers(3, w - 3))
        radius = int(rng.integers(1, 3))
        cv2.circle(img, (x, y), radius, float(rng.uniform(90, 220)), -1)

    # Velké shluky / kapky
    for _ in range(clusters):
        y = int(rng.integers(20, h - 20))
        x = int(rng.integers(20, w - 20))
        radius = int(rng.integers(7, 14))
        cv2.circle(img, (x, y), radius, float(rng.uniform(70, 160)), -1)

    # Vlákna / škrábance – i šikmá, aby se otestovala klasifikace přes momenty
    for _ in range(fibers):
        y = int(rng.integers(30, h - 30))
        x = int(rng.integers(30, w - 30))
        angle = float(rng.uniform(0, math.pi))
        length = int(rng.integers(35, 70))
        dx = int(math.cos(angle) * length / 2)
        dy = int(math.sin(angle) * length / 2)
        cv2.line(img, (x - dx, y - dy), (x + dx, y + dy), float(rng.uniform(90, 180)), 2)

    img += rng.normal(0.0, noise_sigma, size=(h, w)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_series(
    out_dir: str,
    frames: int = 40,
    width: int = 960,
    height: int = 540,
    fps: float = 2.0,
    seed: int = 7,
) -> List[str]:
    """Vygeneruje sérii a vrátí seznam vytvořených souborů."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = datetime(2026, 9, 7, 14, 56, 21)
    written: List[str] = []

    for i in range(frames):
        phase = i / max(1, frames - 1)
        # Zapaření kolem 25 % série, odpařování do 75 %
        if phase < 0.2:
            haze = 0.0
        elif phase < 0.35:
            haze = 45.0 * (phase - 0.2) / 0.15
        elif phase < 0.75:
            haze = 45.0 * math.exp(-(phase - 0.35) * 6.0)
        else:
            haze = 0.0

        particles = int(5 + 45 * phase)
        clusters = 1 if phase > 0.4 else 0
        fibers = 2 if phase > 0.1 else 0

        img = synth_frame(
            (height, width),
            rng,
            haze_amplitude=haze,
            particles=particles,
            fibers=fibers,
            clusters=clusters,
        )

        stamp = start + timedelta(seconds=i / fps)
        name = f"df_{i + 1:05d}_{stamp.strftime('%Y%m%d_%H%M%S')}_{stamp.microsecond // 1000:03d}.png"
        path = os.path.join(out_dir, name)
        imwrite_unicode(path, img)
        written.append(path)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generátor demo série pro Dark-Field Analyzer")
    parser.add_argument("--out", default=os.path.join(os.getcwd(), "demo_darkfield"), help="Výstupní složka")
    parser.add_argument("--frames", type=int, default=40, help="Počet snímků")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    files = generate_series(args.out, args.frames, args.width, args.height, args.fps)
    print(f"Vytvořeno {len(files)} snímků ve složce: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
