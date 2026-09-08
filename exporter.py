"""Export výsledků: CSV tabulka, souhrnné grafy a strojově čitelný souhrn (JSON).

CSV se ukládá s BOM (``utf-8-sig``) a středníkem jako oddělovačem, aby se
korektně otevřel v českém Excelu i s diakritikou. Desetinná čárka se řídí
parametrem ``decimal_comma`` (výchozí ``True`` – české locale).
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyzer import AnalysisParams, FrameMetrics, SeriesResult  # noqa: E402


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS: List[tuple] = [
    # (název sloupce, atribut, formát)
    ("Index", "index", "{:d}"),
    ("Čas [s]", "time_s", "{:.3f}"),
    ("Časové razítko", None, None),
    ("Soubor", "filename", None),
    ("Bias snímek", None, None),
    ("Pokrytí celkem [%]", "total_coverage_pct", "{:.4f}"),
    ("Plocha celkem [px]", "total_area_px", "{:d}"),
    ("Plocha celkem [µm²]", "total_area_um2", "{:.2f}"),
    ("Pokrytí zamlžením [%]", "haze_coverage_pct", "{:.4f}"),
    ("Pokrytí zamlžením bez částic [%]", "haze_only_coverage_pct", "{:.4f}"),
    ("Průměr zamlžení [ADU]", "haze_mean_adu", "{:.2f}"),
    ("Max zamlžení [ADU]", "haze_max_adu", "{:.2f}"),
    ("Mikročástic [ks]", "point_count", "{:d}"),
    ("Plocha mikročástic [px]", "point_area_px", "{:d}"),
    ("Pokrytí mikročásticemi [%]", "point_area_pct", "{:.4f}"),
    ("Velkých shluků [ks]", "cluster_count", "{:d}"),
    ("Plocha shluků [px]", "cluster_area_px", "{:d}"),
    ("Pokrytí shluky [%]", "cluster_area_pct", "{:.4f}"),
    ("Vláken/škrábanců [ks]", "fiber_count", "{:d}"),
    ("Plocha vláken [px]", "fiber_area_px", "{:d}"),
    ("Pokrytí vlákny [%]", "fiber_area_pct", "{:.4f}"),
    ("Délka vláken celkem [px]", "fiber_total_length_px", "{:.1f}"),
    ("Částic celkem [ks]", "total_particle_count", "{:d}"),
    ("Hustota částic [ks/Mpx]", "particle_density_per_mpx", "{:.2f}"),
    ("Průměrná plocha částice [px]", "mean_particle_area_px", "{:.2f}"),
    ("Medián plochy částice [px]", "median_particle_area_px", "{:.2f}"),
    ("P90 plochy částice [px]", "p90_particle_area_px", "{:.2f}"),
    ("Největší částice [px]", "max_particle_area_px", "{:d}"),
    ("Průměrný ekv. průměr částice [µm]", "mean_particle_diameter_um", "{:.3f}"),
    ("Hotspoty (přesycené) [px]", "saturated_pixels_count", "{:d}"),
    ("Nehomogenita [%]", "spatial_heterogeneity_pct", "{:.1f}"),
    ("Těžiště X [%]", "centroid_x_pct", "{:.1f}"),
    ("Těžiště Y [%]", "centroid_y_pct", "{:.1f}"),
    ("Ostrost (var. Laplaciánu)", "focus_score", "{:.2f}"),
    ("Skóre čistoty [%]", "cleanliness_score", "{:.1f}"),
    ("Integrovaný signál [ADU]", "integrated_signal_adu", "{:.1f}"),
    ("Průměrný signál [ADU]", "mean_signal_adu", "{:.2f}"),
    ("Maximum signálu [ADU]", "max_signal_adu", "{:.2f}"),
    ("Úroveň pozadí [ADU]", "bg_level_adu", "{:.2f}"),
    ("Šum pozadí σ [ADU]", "bg_noise_sigma", "{:.2f}"),
    ("Aplikovaný práh [ADU]", "applied_threshold", "{:.2f}"),
    ("SNR [-]", "snr", "{:.2f}"),
    ("Rychlost pokrytí [%/s]", "rate_coverage_pct_per_s", "{:.4f}"),
    ("Rychlost zamlžení [%/s]", "rate_haze_pct_per_s", "{:.4f}"),
    ("Rychlost signálu [ADU/s]", "rate_signal_adu_per_s", "{:.4f}"),
    ("Rychlost částic [ks/s]", "rate_particles_per_s", "{:.2f}"),
    ("Fáze děje", "phase", None),
]


def _fmt(value, spec: Optional[str], decimal_comma: bool) -> str:
    if value is None:
        return ""
    if spec is None:
        return str(value)
    if spec.endswith("d}"):
        text = spec.format(int(round(float(value))))
    else:
        number = float(value)
        if not math.isfinite(number):
            return ""
        text = spec.format(number)
    return text.replace(".", ",") if decimal_comma else text


def export_to_csv(
    metrics_list: Sequence[FrameMetrics],
    output_path: str,
    params: Optional[AnalysisParams] = None,
    folder_name: str = "",
    extra_header: Optional[Dict[str, object]] = None,
    decimal_comma: bool = True,
) -> str:
    """Zapíše podrobnou tabulku všech metrik. Vrací cestu k souboru."""
    if not metrics_list:
        raise ValueError("Žádná data k exportu.")

    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")

        writer.writerow(["# Analýza kontaminace sklíčka v temném poli (Dark-Field Analyzer)"])
        writer.writerow(["# Datum analýzy", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        if folder_name:
            writer.writerow(["# Zdrojová složka", folder_name])
        writer.writerow(["# Počet snímků", len(metrics_list)])
        if params:
            writer.writerow(["# Počet snímků biasu", params.bias_frames])
            writer.writerow(["# Metoda biasu", params.bias_method])
            writer.writerow(["# Binning", params.resolution_label])
            writer.writerow(["# Režim prahu", params.threshold_mode])
            writer.writerow(["# Sigma násobek šumu", _fmt(params.sigma, "{:.2f}", decimal_comma)])
            writer.writerow(["# Absolutní práh [ADU]", _fmt(params.absolute_threshold, "{:.2f}", decimal_comma)])
            writer.writerow(["# Práh zamlžení [ADU]", _fmt(params.haze_threshold, "{:.2f}", decimal_comma)])
            writer.writerow(["# Min. plocha částice [px]", params.min_area_px])
            writer.writerow(["# Hranice velkého shluku [px]", params.cluster_min_area_px])
            writer.writerow(["# Min. protáhlost vlákna", _fmt(params.fiber_aspect_ratio, "{:.2f}", decimal_comma)])
            writer.writerow(["# Kalibrace [µm/px]", _fmt(params.um_per_px, "{:.4f}", decimal_comma)])
        for key, value in (extra_header or {}).items():
            writer.writerow([f"# {key}", value])
        writer.writerow([])

        writer.writerow([name for name, _attr, _spec in CSV_COLUMNS])
        for metrics in metrics_list:
            row = []
            for name, attr, spec in CSV_COLUMNS:
                if name == "Časové razítko":
                    row.append(metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                elif name == "Bias snímek":
                    row.append("ano" if metrics.is_bias_frame else "")
                else:
                    row.append(_fmt(getattr(metrics, attr), spec, decimal_comma))
            writer.writerow(row)

    return output_path


# ---------------------------------------------------------------------------
# Souhrn (JSON)
# ---------------------------------------------------------------------------

def summarize(result: SeriesResult) -> Dict[str, object]:
    """Sestaví souhrnnou statistiku celé série (pro protokol i JSON export)."""
    metrics = result.metrics
    if not metrics:
        return {}

    coverage = [m.total_coverage_pct for m in metrics]
    cleanliness = [m.cleanliness_score for m in metrics]
    peak = max(metrics, key=lambda m: m.total_coverage_pct)
    worst = min(metrics, key=lambda m: m.cleanliness_score)

    phases: Dict[str, int] = {}
    for m in metrics:
        phases[m.phase] = phases.get(m.phase, 0) + 1

    duration = metrics[-1].time_s - metrics[0].time_s
    composition = {
        label: round(sum(values) / len(values), 4) if values else 0.0
        for label, _color, values in composition_series(metrics)
    }
    return {
        "slozeni_prumerne_pokryti_pct": composition,
        "pocet_snimku": len(metrics),
        "delka_mereni_s": round(duration, 3),
        "pokryti_prumer_pct": round(sum(coverage) / len(coverage), 4),
        "pokryti_max_pct": round(peak.total_coverage_pct, 4),
        "pokryti_max_cas_s": round(peak.time_s, 3),
        "pokryti_max_snimek": peak.filename,
        "cistota_prumer_pct": round(sum(cleanliness) / len(cleanliness), 2),
        "cistota_min_pct": round(worst.cleanliness_score, 2),
        "cistota_min_snimek": worst.filename,
        "castic_na_konci": metrics[-1].total_particle_count,
        "castic_na_zacatku": metrics[0].total_particle_count,
        "max_rychlost_pokryti_pct_s": round(max(m.rate_coverage_pct_per_s for m in metrics), 4),
        "min_rychlost_pokryti_pct_s": round(min(m.rate_coverage_pct_per_s for m in metrics), 4),
        "faze": phases,
        "cas_analyzy_s": round(result.elapsed_s, 2),
        "nactene_chyby": len(result.failed_files),
        "varovani": list(result.warnings),
        "synteticka_casova_osa": result.synthetic_time_axis,
    }


def export_summary_json(result: SeriesResult, output_path: str, folder_name: str = "") -> str:
    """Uloží souhrn měření a použité parametry do JSON (pro další zpracování)."""
    payload = {
        "slozka": folder_name,
        "vytvoreno": datetime.now().isoformat(timespec="seconds"),
        "parametry": asdict(result.params) if result.params else {},
        "souhrn": summarize(result),
    }
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return output_path


# ---------------------------------------------------------------------------
# Grafy
# ---------------------------------------------------------------------------

#: Kategorie kontaminace pro graf složení plochy.
#:
#: Pořadí je zároveň pořadím vrstev zdola nahoru: usazené částice tvoří
#: stabilní základ, proměnlivý opar leží nahoře. Barvy odpovídají barvám
#: v prohlížeči snímků (shluky červeně, mikročástice žlutooranžově, vlákna
#: zeleně, zamlžení modře) a jsou zvolené tak, aby sousední vrstvy zůstaly
#: rozlišitelné i při barvosleposti – červená a zelená proto nikdy nesousedí.
COMPOSITION_LAYERS = [
    ("cluster_area_pct", "Velké shluky / kapky", "#B03A2E"),
    ("point_area_pct", "Mikročástice (prach)", "#D4900A"),
    ("fiber_area_pct", "Vlákna / škrábance", "#1D8348"),
    ("haze_only_coverage_pct", "Difuzní zamlžení", "#2E86C1"),
]

#: Barva pozadí grafu – používá se i jako mezera mezi vrstvami.
SURFACE_COLOR = "#FFFFFF"


def composition_series(metrics: Sequence[FrameMetrics]) -> List[Tuple[str, str, List[float]]]:
    """Vrátí data pro graf složení jako ``[(popis, barva, hodnoty v %), ...]``.

    Složky jsou disjunktní (opar se počítá bez plochy částic), takže jejich
    součet je přesně celkové pokrytí snímku.
    """
    return [
        (label, color, [getattr(m, attr) for m in metrics])
        for attr, label, color in COMPOSITION_LAYERS
    ]


def plot_composition(
    ax_area,
    ax_share,
    metrics: Sequence[FrameMetrics],
    ax_particles=None,
) -> None:
    """Vykreslí složení kontaminace.

    ``ax_area``      – vrstvený graf pokrytí plochy [% snímku] v čase,
    ``ax_share``     – průměrné zastoupení typů v kontaminované ploše [%],
    ``ax_particles`` – volitelně tentýž graf bez zamlžení.

    Samostatný panel bez zamlžení má svůj důvod: opar se počítá jako plocha nad
    prahem 4 ADU, takže při zapaření zabere klidně 90 % snímku, zatímco částice
    bývají v desetinách procenta. Ve společném lineárním grafu by tenké vrstvy
    zanikly a nešly by odečíst.
    """
    metrics = list(metrics)
    times = [m.time_s for m in metrics]
    layers = composition_series(metrics)
    means = [sum(values) / len(values) if values else 0.0 for _label, _color, values in layers]

    ax_area.stackplot(
        times,
        *[values for _label, _color, values in layers],
        # V legendě je rovnou i průměrné pokrytí – u tenkých vrstev je to
        # jediné místo, kde se jejich hodnota dá přečíst.
        labels=[f"{label}  (⌀ {mean:.3f} %)" for (label, _color, _values), mean in zip(layers, means)],
        colors=[color for _label, color, _values in layers],
        edgecolor=SURFACE_COLOR,
        linewidth=1.2,          # 2px mezera mezi vrstvami při běžném DPI
    )
    ax_area.set_xlabel("Čas od počátku [s]", fontweight="bold")
    ax_area.set_ylabel("Pokrytí plochy snímku [%]", fontweight="bold")
    ax_area.grid(True, linestyle="--", alpha=0.4)
    ax_area.margins(x=0)
    ax_area.set_ylim(bottom=0)

    handles, labels_text = ax_area.get_legend_handles_labels()
    ax_area.legend(handles[::-1], labels_text[::-1], loc="best", fontsize=9, frameon=True, framealpha=0.92)
    ax_area.set_title(
        "Podíl jednotlivých typů kontaminace na ploše snímku",
        fontsize=11, fontweight="bold",
    )

    if ax_particles is not None:
        solid = [layer for layer in layers if layer[0] != COMPOSITION_LAYERS[-1][1]]
        ax_particles.stackplot(
            times,
            *[values for _label, _color, values in solid],
            labels=[label for label, _color, _values in solid],
            colors=[color for _label, color, _values in solid],
            edgecolor=SURFACE_COLOR,
            linewidth=1.2,
        )
        ax_particles.set_xlabel("Čas od počátku [s]", fontweight="bold")
        ax_particles.set_ylabel("Pokrytí plochy snímku [%]", fontweight="bold")
        ax_particles.grid(True, linestyle="--", alpha=0.4)
        ax_particles.margins(x=0)
        ax_particles.set_ylim(bottom=0)
        handles_p, labels_p = ax_particles.get_legend_handles_labels()
        ax_particles.legend(handles_p[::-1], labels_p[::-1], loc="best", fontsize=9,
                            frameon=True, framealpha=0.92)
        ax_particles.set_title(
            "Totéž bez zamlžení – jen pevné částice (jiné měřítko osy Y)",
            fontsize=11, fontweight="bold",
        )

    # Průměrné zastoupení v kontaminované ploše (vodorovný vrstvený pruh)
    total = sum(means)
    left = 0.0
    for (label, color, _values), mean_pct in zip(layers, means):
        share = 100.0 * mean_pct / total if total > 0 else 0.0
        ax_share.barh(0, share, left=left, color=color, edgecolor=SURFACE_COLOR, height=0.55, linewidth=1.2)
        if share >= 4.0:        # popisek přímo v pruhu, jen když se tam vejde
            ax_share.text(
                left + share / 2.0, 0, f"{share:.0f} %",
                ha="center", va="center", fontsize=9, fontweight="bold", color="#FFFFFF",
            )
        left += share

    ax_share.set_xlim(0, 100)
    ax_share.set_ylim(-0.5, 0.5)
    ax_share.set_yticks([])
    ax_share.set_xlabel("Zastoupení v kontaminované ploše [%]", fontweight="bold")
    ax_share.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax_share.set_axisbelow(True)
    ax_share.set_title(
        f"Průměrné zastoupení typů  (kontaminace celkem ⌀ {total:.3f} % plochy)",
        fontsize=9.5, fontweight="bold",
    )

def _shade_phases(ax, metrics: Sequence[FrameMetrics]) -> None:
    """Podbarví oblasti nárůstu (červeně) a ústupu (modře)."""
    colors = {"nárůst / zamlžování": ("#C0392B", 0.08), "odpařování / ústup": ("#2980B9", 0.08)}
    start = 0
    for i in range(1, len(metrics) + 1):
        end_of_run = i == len(metrics) or metrics[i].phase != metrics[start].phase
        if end_of_run:
            style = colors.get(metrics[start].phase)
            if style:
                ax.axvspan(metrics[start].time_s, metrics[i - 1].time_s, color=style[0], alpha=style[1], lw=0)
            start = i


def export_summary_plots(
    metrics_list: Sequence[FrameMetrics],
    output_image_path: str,
    title: str = "Průběh analýzy kontaminace sklíčka",
    dpi: int = 160,
) -> str:
    """Vykreslí a uloží souhrnný 4panelový graf."""
    if not metrics_list:
        raise ValueError("Žádná data k vykreslení.")

    metrics = list(metrics_list)
    times = [m.time_s for m in metrics]

    fig = plt.figure(figsize=(12, 20))
    # Předposlední řádek je jen mezera, aby popisek osy 6. panelu nenarazil
    # do titulku pruhu se zastoupením typů.
    grid = fig.add_gridspec(
        8, 1, height_ratios=[1, 1, 1, 1, 1, 1, 0.16, 0.32],
        hspace=0.34, top=0.965, bottom=0.04, left=0.085, right=0.93,
    )
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in range(1, 6)]
    ax_share = fig.add_subplot(grid[7, 0])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    # 1. Pokrytí a skóre čistoty
    ax1 = axes[0]
    _shade_phases(ax1, metrics)
    lines = ax1.plot(times, [m.total_coverage_pct for m in metrics], "k-", label="Celkové pokrytí [%]", linewidth=2.2)
    lines += ax1.plot(times, [m.haze_coverage_pct for m in metrics], "b--", label="Zamlžení / kondenzace [%]", alpha=0.85)
    ax1.set_ylabel("Pokrytí plochy [%]", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1_twin = ax1.twinx()
    lines += ax1_twin.plot(
        times, [m.cleanliness_score for m in metrics], color="#27AE60", linestyle="-.",
        label="Skóre čistoty [0–100 %]", linewidth=1.8,
    )
    ax1_twin.set_ylabel("Čistota [%]", color="#27AE60", fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="#27AE60")
    ax1_twin.set_ylim(0, 105)
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True, fontsize=9)
    ax1.set_title("1. Vývoj znečištění a skóre čistoty sklíčka", fontsize=11, fontweight="bold")

    # 2. Počty částic
    ax2 = axes[1]
    _shade_phases(ax2, metrics)
    ax2.plot(times, [m.point_count for m in metrics], color="#E67E22", label="Mikročástice (prach)", marker="o", markersize=3)
    ax2.plot(times, [m.cluster_count for m in metrics], color="#C0392B", label="Velké shluky", marker="s", markersize=3)
    ax2.plot(times, [m.fiber_count for m in metrics], color="#2980B9", label="Vlákna / škrábance", marker="^", markersize=3)
    ax2.set_ylabel("Počet [ks]", fontweight="bold")
    ax2.set_title("2. Počty částic podle kategorií", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Signál a hotspoty
    ax3 = axes[2]
    _shade_phases(ax3, metrics)
    lines3 = ax3.plot(times, [m.mean_signal_adu for m in metrics], color="#8E44AD", label="Průměrný signál kontaminace [ADU]", linewidth=2)
    lines3 += ax3.plot(times, [m.bg_noise_sigma for m in metrics], color="#95A5A6", linestyle=":", label="Šum pozadí σ [ADU]", linewidth=1.4)
    ax3.set_ylabel("Signál [ADU]", color="#8E44AD", fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#8E44AD")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3_twin = ax3.twinx()
    lines3 += ax3_twin.plot(
        times, [m.saturated_pixels_count for m in metrics], color="#D35400", linestyle="--",
        label="Hotspoty (přesycené pixely) [px]", linewidth=1.5,
    )
    ax3_twin.set_ylabel("Hotspoty [px]", color="#D35400", fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#D35400")
    ax3.legend(lines3, [l.get_label() for l in lines3], loc="upper right", frameon=True, fontsize=9)
    ax3.set_title("3. Intenzita signálu, šum pozadí a optické hotspoty", fontsize=11, fontweight="bold")

    # 4. Rychlost změny a nehomogenita
    ax4 = axes[3]
    _shade_phases(ax4, metrics)
    ax4.axhline(0, color="gray", linewidth=1, alpha=0.6)
    lines4 = ax4.plot(times, [m.rate_coverage_pct_per_s for m in metrics], color="#C0392B", label="d(Pokrytí)/dt [%/s]", linewidth=2)
    ax4.set_ylabel("Rychlost [%/s]", color="#C0392B", fontweight="bold")
    ax4.tick_params(axis="y", labelcolor="#C0392B")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4_twin = ax4.twinx()
    lines4 += ax4_twin.plot(
        times, [m.spatial_heterogeneity_pct for m in metrics], color="#2C3E50", linestyle=":",
        label="Nehomogenita rozložení [%]", linewidth=1.8,
    )
    ax4_twin.set_ylabel("Nehomogenita [%]", color="#2C3E50", fontweight="bold")
    ax4_twin.tick_params(axis="y", labelcolor="#2C3E50")
    ax4_twin.set_ylim(0, 105)
    ax4.legend(lines4, [l.get_label() for l in lines4], loc="upper right", frameon=True, fontsize=9)
    ax4.set_title("4. Dynamika děje a prostorová nehomogenita (podbarveno podle fáze)", fontsize=11, fontweight="bold")

    # 5. a 6. Složení kontaminace podle typu
    ax5, ax6 = axes[4], axes[5]
    plot_composition(ax5, ax_share, metrics, ax_particles=ax6)
    ax5.set_title("5. Podíl typů kontaminace na ploše snímku", fontsize=11, fontweight="bold")
    ax5.set_xlabel("")
    ax6.set_title("6. Totéž bez zamlžení – jen pevné částice", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Čas od počátku [s]", fontsize=10, fontweight="bold")

    directory = os.path.dirname(os.path.abspath(output_image_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output_image_path, dpi=dpi)
    plt.close(fig)
    return output_image_path


def export_all(result: SeriesResult, csv_path: str, folder_name: str = "") -> Dict[str, str]:
    """Uloží CSV, graf i JSON souhrn vedle sebe. Vrací mapu ``{typ: cesta}``."""
    base = os.path.splitext(csv_path)[0]
    outputs = {
        "csv": export_to_csv(result.metrics, csv_path, result.params, folder_name),
        "png": export_summary_plots(result.metrics, base + "_grafy.png", f"Analýza kontaminace – {folder_name or 'měření'}"),
        "json": export_summary_json(result, base + "_souhrn.json", folder_name),
    }
    return outputs
