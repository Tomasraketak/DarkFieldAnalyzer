"""Interaktivní prohlížeč snímků s barevnou klasifikační maskou (PyQt6).

Opravy proti původní verzi:

* Náhled se počítá ve **stejné geometrii jako analýza**. Původní verze
  zmenšovala snímek čtyřikrát, ale bias nechala v poloviční velikosti –
  ``cv2.subtract`` pak vyhodil výjimku uvnitř Qt slotu, což PyQt6 řeší
  ukončením celého procesu (aplikace „zmizela“ hned po dokončení analýzy).
* Chyba při vykreslování už nemůže shodit aplikaci; zobrazí se v stavovém řádku.
* ``QImage`` dostává souvislé pole a hotový pixmap se kopíruje, takže nevzniká
  odkaz na uvolněnou paměť.
* Překryv se míchá jen v místě masek – zbytek snímku zůstává v plném jasu.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from analyzer import AnalysisParams, BiasModel, analyze_prepared_frame, apply_geometry
from frameio import build_time_axis, imwrite_unicode, load_frame

#: Barvy jednotlivých kategorií kontaminace (RGB).
CATEGORY_COLORS = {
    "mask_haze": (0, 200, 255),      # azurová – difuzní zamlžení
    "mask_points": (255, 255, 0),    # žlutá – mikročástice
    "mask_clusters": (255, 40, 40),  # červená – velké shluky
    "mask_fibers": (40, 255, 40),    # zelená – vlákna a škrábance
}


def normalize_to_u8(image: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Převede float32 ADU na uint8 pro zobrazení."""
    return np.clip(image * gain, 0, 255).astype(np.uint8)


def create_color_overlay(
    image_u8: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.6,
    centroid: Optional[Tuple[float, float]] = None,
    saturation_adu: float = 250.0,
) -> np.ndarray:
    """Vytvoří RGB náhled s barevně odlišenými typy kontaminace."""
    base = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)
    overlay = base.copy()
    touched = np.zeros(image_u8.shape, dtype=bool)

    for key, color in CATEGORY_COLORS.items():
        mask = masks.get(key)
        if mask is None:
            continue
        selected = mask > 0
        if selected.any():
            overlay[selected] = color
            touched |= selected

    hotspots = image_u8 >= int(saturation_adu)
    if hotspots.any():
        overlay[hotspots] = (255, 0, 255)   # fialová – přesycené pixely
        touched |= hotspots

    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
    blended[~touched] = base[~touched]      # nezasažené okolí zůstává v plném jasu

    if centroid is not None:
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        h, w = image_u8.shape[:2]
        if 0 <= cx < w and 0 <= cy < h:
            cv2.drawMarker(
                blended, (cx, cy), (255, 255, 0),
                markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2,
            )
    return blended


class ImageViewerWidget(QWidget):
    """Widget pro vizuální kontrolu snímků s klasifikační maskou."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_paths: List[str] = []
        self.bias: Optional[BiasModel] = None
        self.params: Optional[AnalysisParams] = None
        self.timestamps: List = []
        self.current_index = 0
        self._last_rgb: Optional[np.ndarray] = None

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(60)
        self.debounce_timer.timeout.connect(self.render_current_frame)

        self.init_ui()

    # -- rozhraní -----------------------------------------------------------

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Režim:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Barevná klasifikace (Overlay)",
            "Původní snímek",
            "Diference (snímek − bias)",
            "Difuzní zamlžení (Haze)",
            "Ostrá složka (částice/vlákna)",
            "Detekovaná maska (binárně)",
        ])
        self.view_mode_combo.currentIndexChanged.connect(self.trigger_render)
        self.view_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.view_mode_combo.setMinimumContentsLength(16)
        controls.addWidget(self.view_mode_combo, 1)

        self.chk_fast = QCheckBox("Rychlý náhled (2× zmenšeno)")
        self.chk_fast.setToolTip(
            "Zrychlí posouvání u 4K snímků. Čísla ve stavovém řádku pak mohou\n"
            "být mírně jiná než v tabulce, protože se počítají z menšího obrazu."
        )
        self.chk_fast.stateChanged.connect(self.trigger_render)
        controls.addWidget(self.chk_fast)

        controls.addSpacing(10)
        controls.addWidget(QLabel("Snímek:"))
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.valueChanged.connect(self.on_frame_spin_changed)
        controls.addWidget(self.spin_frame)

        self.lbl_frame_info = QLabel("0 / 0")
        controls.addWidget(self.lbl_frame_info)

        self.btn_save = QPushButton("Uložit náhled…")
        self.btn_save.clicked.connect(self.save_current_view)
        self.btn_save.setEnabled(False)
        controls.addWidget(self.btn_save)

        controls.addStretch()
        layout.addLayout(controls)

        # Legenda má vlastní řádek a zalamuje se – v jedné liště s ovládáním
        # si totiž vynutila minimální šířku okna přes 1300 px.
        legend = QLabel(
            "<b>Legenda:</b> "
            "<span style='color:#00A8DF;'>■</span> zamlžení &nbsp;"
            "<span style='color:#D4900A;'>■</span> mikročástice &nbsp;"
            "<span style='color:#C0392B;'>■</span> shluky &nbsp;"
            "<span style='color:#1D8348;'>■</span> vlákna &nbsp;"
            "<span style='color:#B03AB0;'>■</span> hotspoty &nbsp;"
            "<span style='color:#8A6D0B;'>✚</span> těžiště kontaminace"
        )
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size: 11px;")
        layout.addWidget(legend)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(240, 180)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #151515;")
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area, 1)

        self.status_lbl = QLabel("Vyberte složku a spusťte analýzu.")
        self.status_lbl.setStyleSheet("color: #666; font-size: 11px;")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

    # -- data ---------------------------------------------------------------

    def set_dataset(self, image_paths: List[str], bias: Optional[BiasModel], params: AnalysisParams) -> None:
        self.image_paths = list(image_paths)
        self.bias = bias
        self.params = params
        self.timestamps, _ = build_time_axis(self.image_paths, assumed_fps=params.assumed_fps)

        count = len(self.image_paths)
        blocked = self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(blocked)

        blocked = self.spin_frame.blockSignals(True)
        self.spin_frame.setRange(0, max(0, count - 1))
        self.spin_frame.setValue(0)
        self.spin_frame.blockSignals(blocked)

        if count and bias is not None:
            self.trigger_render()
        else:
            self.image_label.clear()
            self.btn_save.setEnabled(False)
            self.status_lbl.setText("Žádné snímky k zobrazení.")

    # -- události -----------------------------------------------------------

    def on_slider_changed(self, value: int) -> None:
        if value != self.spin_frame.value():
            self.spin_frame.setValue(value)
        self.trigger_render()

    def on_frame_spin_changed(self, value: int) -> None:
        if value != self.slider.value():
            self.slider.setValue(value)
        self.trigger_render()

    def trigger_render(self) -> None:
        self.debounce_timer.start()

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        if self._last_rgb is not None:
            self._show_rgb(self._last_rgb)

    # -- vykreslení ---------------------------------------------------------

    def render_current_frame(self) -> None:
        """Načte snímek, spočítá masky a vykreslí zvolený režim.

        Celý výpočet je v ``try``: výjimka uvnitř Qt slotu by jinak ukončila
        celou aplikaci bez jakékoliv hlášky.
        """
        if not self.image_paths or self.bias is None or self.params is None:
            return

        index = max(0, min(self.slider.value(), len(self.image_paths) - 1))
        self.current_index = index
        path = self.image_paths[index]
        filename = os.path.basename(path)
        self.lbl_frame_info.setText(f"{index + 1} / {len(self.image_paths)}")

        try:
            frame = load_frame(
                path, full_scale=self.bias.full_scale, mono_mode=self.params.mono_mode
            )
            binning = self.bias.binning * (2 if self.chk_fast.isChecked() else 1)
            image = apply_geometry(frame.data, binning, self.params.roi)

            now = datetime.now()
            timestamp = self.timestamps[index] if index < len(self.timestamps) else now
            t0 = self.timestamps[0] if self.timestamps else timestamp

            metrics, masks = analyze_prepared_frame(
                image=image,
                bias=self.bias.data,          # rozměr se srovná automaticky
                params=self.params,
                index=index,
                filename=filename,
                filepath=path,
                timestamp=timestamp,
                t0=t0,
                binning=binning,
                generate_masks=True,
                level_offset=self.bias.level_offset_adu,
            )
            rgb = self._compose_view(image, masks, metrics)
        except Exception as exc:  # noqa: BLE001 – slot nesmí propustit výjimku
            self.image_label.clear()
            self.btn_save.setEnabled(False)
            self.status_lbl.setText(f"Snímek {filename} nelze zobrazit: {exc}")
            traceback.print_exc()
            return

        self._last_rgb = rgb
        self.btn_save.setEnabled(True)
        self._show_rgb(rgb)

        self.status_lbl.setText(
            f"Soubor: {filename}   |   Čas: {metrics.time_s:.2f} s   |   "
            f"Pokrytí: {metrics.total_coverage_pct:.3f} % (zamlžení {metrics.haze_coverage_pct:.2f} %)   |   "
            f"Částic: {metrics.total_particle_count} "
            f"(prach {metrics.point_count}, shluky {metrics.cluster_count}, vlákna {metrics.fiber_count})   |   "
            f"Práh: {metrics.applied_threshold:.1f} ADU   |   "
            f"Nehomogenita: {metrics.spatial_heterogeneity_pct:.1f} %   |   "
            f"Čistota: {metrics.cleanliness_score:.1f}/100"
        )

    def _compose_view(self, image: np.ndarray, masks: Dict[str, np.ndarray], metrics) -> np.ndarray:
        mode = self.view_mode_combo.currentIndex()
        image_u8 = normalize_to_u8(image)

        if mode == 0:
            centroid = (
                metrics.centroid_x_pct / 100.0 * image.shape[1],
                metrics.centroid_y_pct / 100.0 * image.shape[0],
            )
            return create_color_overlay(
                image_u8, masks, centroid=centroid, saturation_adu=self.params.saturation_adu
            )
        if mode == 1:
            return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)
        if mode == 2:
            colored = cv2.applyColorMap(normalize_to_u8(masks["diff"], gain=3.0), cv2.COLORMAP_INFERNO)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        if mode == 3:
            colored = cv2.applyColorMap(normalize_to_u8(masks["haze"], gain=4.0), cv2.COLORMAP_OCEAN)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        if mode == 4:
            return cv2.cvtColor(normalize_to_u8(masks["sharp"], gain=4.0), cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(masks["mask_total"], cv2.COLOR_GRAY2RGB)

    def _show_rgb(self, rgb: np.ndarray) -> None:
        """Vykreslí RGB pole do QLabel s dopočítaným zmenšením na velikost okna."""
        height, width = rgb.shape[:2]
        viewport = self.scroll_area.viewport()
        target_w = max(160, viewport.width() - 4)
        target_h = max(120, viewport.height() - 4)
        scale = min(target_w / width, target_h / height, 1.0)

        if scale < 0.99:
            preview = cv2.resize(
                rgb, (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview = rgb

        preview = np.ascontiguousarray(preview)
        h, w = preview.shape[:2]
        qimage = QImage(preview.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

    def save_current_view(self) -> None:
        """Uloží aktuálně zobrazený náhled jako PNG (pro protokol)."""
        if self._last_rgb is None or not self.image_paths:
            return
        default = os.path.splitext(self.image_paths[self.current_index])[0] + "_nahled.png"
        path, _ = QFileDialog.getSaveFileName(self, "Uložit náhled", default, "PNG (*.png)")
        if not path:
            return
        bgr = cv2.cvtColor(self._last_rgb, cv2.COLOR_RGB2BGR)
        if imwrite_unicode(path, bgr):
            self.status_lbl.setText(f"Náhled uložen: {path}")
        else:
            QMessageBox.warning(self, "Chyba", f"Náhled se nepodařilo uložit:\n{path}")
