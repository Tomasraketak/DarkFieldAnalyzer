r"""Spouštěcí skript aplikace Dark-Field Contamination Analyzer.

GUI::

    py main.py

Dávkové zpracování bez GUI::

    py main.py --folder "C:\Users\...\darkfield_20260907_145621" --bias 3 --binning 2

Zpracování všech podsložek jedním příkazem::

    py main.py --folder "C:\Users\...\BMS fotky" --batch
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List, Optional, Tuple

from analyzer import AnalysisParams, analyze_series
from exporter import export_all, summarize
from frameio import list_image_files, list_measurement_folders
from reference import ReferenceError


def _configure_console() -> None:
    """Zajistí, že se česká diakritika vypíše i v okně cmd.exe."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def parse_roi(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Rozparsuje ``--roi x,y,sirka,vyska`` (v pixelech plného rozlišení)."""
    if not text:
        return None
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("ROI musí mít tvar x,y,sirka,vyska")
    x, y, width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("Šířka i výška ROI musí být kladné")
    return (x, y, width, height)


def build_params(args: argparse.Namespace) -> AnalysisParams:
    return AnalysisParams(
        bias_frames=args.bias,
        bias_method=args.bias_method,
        threshold_mode=args.mode,
        sigma=args.sigma,
        absolute_threshold=args.absolute,
        haze_threshold=args.haze,
        min_area_px=args.min_area,
        cluster_min_area_px=args.cluster_area,
        fiber_aspect_ratio=args.aspect_ratio,
        fiber_min_length_px=args.fiber_length,
        saturation_adu=args.saturation,
        um_per_px=args.scale,
        binning=args.binning,
        roi=parse_roi(args.roi),
        workers=args.workers,
        exclude_bias_from_series=not args.include_bias,
        reference_mode=args.reference,
        reference_dir=args.reference_dir,
        match_reference_level=not args.no_level_match,
        mono_mode=args.mono,
    )


def run_folder(folder: str, args: argparse.Namespace) -> int:
    """Zanalyzuje jednu složku a uloží výstupy. Vrací návratový kód."""
    image_paths = list_image_files(folder, recursive=args.recursive)
    if not image_paths:
        print(f"Chyba: Ve složce '{folder}' nejsou žádné snímky (PNG/TIFF/BMP/JPG).")
        return 1

    params = build_params(args)
    print(f"\nSložka: {folder}")
    print(f"Snímků: {len(image_paths)}   |   bias: {params.bias_frames} ({params.bias_method})"
          f"   |   binning: {params.resolution_label}   |   barva: {params.mono_label}")

    total = len(image_paths)
    step = max(1, total // 20)

    def progress(done: int, count: int, metrics) -> None:
        if done % step == 0 or done == count:
            print(f"  {done:>5}/{count}  ({100.0 * done / count:5.1f} %)  "
                  f"pokrytí {metrics.total_coverage_pct:7.3f} %  čistota {metrics.cleanliness_score:5.1f}",
                  flush=True)

    try:
        result = analyze_series(image_paths, params, progress=None if args.quiet else progress)
    except ReferenceError as exc:
        # Režim --reference reference: bez použitelné reference se raději nic
        # nespočítá, aby výsledek nevznikl proti jinému pozadí, než uživatel čeká.
        print(f"Chyba: {exc}")
        print("       Použijte --reference auto (bias ze série) nebo --reference-dir.")
        return 1

    for note in result.notes:
        print(f"  · {note}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    for path, reason in result.failed_files[:10]:
        print(f"  ! {os.path.basename(path)}: {reason}")

    if not result.metrics:
        print("Chyba: Nepodařilo se zpracovat ani jeden snímek.")
        return 1

    per_frame = result.elapsed_s / len(result.metrics) * 1000.0
    print(f"Hotovo za {result.elapsed_s:.2f} s ({per_frame:.1f} ms/snímek).")
    if result.bias is not None:
        print(f"  Pozadí: {result.bias.origin_label}"
              + (f"   |   srovnání úrovně {result.bias.level_offset_adu:+.2f} ADU"
                 if result.bias.is_external else ""))

    summary = summarize(result)
    print(f"  Průměrné pokrytí: {summary['pokryti_prumer_pct']:.3f} %"
          f"   |   maximum: {summary['pokryti_max_pct']:.3f} % v {summary['pokryti_max_cas_s']:.2f} s")
    print(f"  Průměrná čistota: {summary['cistota_prumer_pct']:.1f} / 100"
          f"   |   částic {summary['castic_na_zacatku']} → {summary['castic_na_konci']}")
    print("  Fáze: " + ", ".join(f"{name}: {count}" for name, count in summary["faze"].items()))

    csv_path = args.export or os.path.join(folder, f"analyza_{os.path.basename(os.path.normpath(folder))}.csv")
    outputs = export_all(result, csv_path, os.path.basename(os.path.normpath(folder)))
    print("  Uloženo:")
    for kind, path in outputs.items():
        print(f"    {kind.upper():>4}: {path}")
    return 0


def run_cli(args: argparse.Namespace) -> int:
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Chyba: Složka '{folder}' neexistuje.")
        return 1

    if not args.batch:
        return run_folder(folder, args)

    folders = [path for path, _count in list_measurement_folders(folder)]
    if not folders:
        print(f"Chyba: Ve složce '{folder}' nejsou žádné podsložky se snímky.")
        return 1

    print(f"Dávkový režim: {len(folders)} složek k analýze.")
    failures = 0
    for path in folders:
        args_copy = argparse.Namespace(**vars(args))
        args_copy.export = None  # v dávce se ukládá vedle snímků
        failures += run_folder(path, args_copy)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dark-Field Slide Contamination Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder", help="Složka se snímky (bez ní se spustí GUI)")
    parser.add_argument("--batch", action="store_true", help="Zpracovat všechny podsložky")
    parser.add_argument("--recursive", action="store_true", help="Hledat snímky i v podsložkách")
    parser.add_argument("--bias", type=int, default=3, help="Počet snímků pro bias")
    parser.add_argument("--bias-method", choices=["median", "mean"], default="median", help="Metoda výpočtu biasu")
    parser.add_argument("--binning", type=int, choices=[0, 1, 2, 4], default=0, help="0 = auto, 1/2/4 = pevný binning")
    parser.add_argument("--mode", choices=["sigma", "absolute"], default="sigma", help="Režim prahu")
    parser.add_argument("--sigma", type=float, default=4.0, help="Násobek šumu σ")
    parser.add_argument("--absolute", type=float, default=12.0, help="Absolutní práh [ADU]")
    parser.add_argument("--haze", type=float, default=4.0, help="Práh zamlžení [ADU]")
    parser.add_argument("--min-area", type=int, default=3, help="Minimální plocha částice [px]")
    parser.add_argument("--cluster-area", type=int, default=100, help="Hranice velkého shluku [px]")
    parser.add_argument("--aspect-ratio", type=float, default=2.8, help="Protáhlost pro vlákno")
    parser.add_argument("--fiber-length", type=int, default=12, help="Minimální délka vlákna [px]")
    parser.add_argument("--saturation", type=float, default=250.0, help="Hranice hotspotu [ADU]")
    parser.add_argument("--scale", type=float, default=1.0, help="Kalibrace [µm/px]")
    parser.add_argument("--workers", type=int, default=0, help="Počet vláken (0 = auto)")
    parser.add_argument("--include-bias", action="store_true",
                        help="Ponechat bias snímky ve výsledné řadě (ve výchozím stavu se vynechávají)")
    parser.add_argument("--reference", choices=["auto", "reference", "serie"], default="auto",
                        help="Zdroj pozadí: auto = reference ze složky, jinak série")
    parser.add_argument("--reference-dir",
                        help="Složka s .npz referencemi (jinak se hledá podsložka „reference“)")
    parser.add_argument("--no-level-match", action="store_true",
                        help="Nesrovnávat úroveň externí reference se snímky")
    parser.add_argument("--mono", choices=["luma", "prumer", "maximum", "r", "g", "b"], default="luma",
                        help="Převod barevného snímku na intenzitu")
    parser.add_argument("--roi", help="Výřez k analýze ve tvaru x,y,sirka,vyska (px plného rozlišení)")
    parser.add_argument("--export", help="Cesta k výstupnímu CSV")
    parser.add_argument("--quiet", action="store_true", help="Nevypisovat průběh")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.folder:
            return run_cli(args)
        from gui import main as gui_main
        return gui_main()
    except KeyboardInterrupt:
        print("\nPřerušeno uživatelem.")
        return 130
    except Exception:  # noqa: BLE001 – poslední záchytná síť pro spuštění z .bat
        print("\n" + "=" * 70)
        print("DOŠLO K NEOČEKÁVANÉ CHYBĚ:")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)
        if sys.stdin and sys.stdin.isatty():
            input("\nStiskněte Enter pro zavření okna…")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
