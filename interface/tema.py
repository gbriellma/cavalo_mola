"""Tema visual, paleta ('Asfalto & Sinalização'), fontes e conversores Qt."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtGui import QFont, QFontDatabase, QImage

CORES = {
    "bg_app": "#F1F4F8",
    "bg_card": "#FFFFFF",
    "bg_subtle": "#F8FAFC",
    "bg_header": "#161B22",
    "bg_canvas": "#0E131A",
    "border": "#D8E0EA",
    "border_strong": "#CBD5E1",
    "text": "#0F172A",
    "text_sec": "#475569",
    "text_muted": "#64748B",
    "teal": "#0F4C5C",
    "teal_hover": "#15657A",
    "teal_soft": "#E6F3F6",
    "amber": "#D97706",
    "amber_bright": "#F59E0B",
    "amber_soft": "#FEF3C7",
    "sev_nenhum_bg": "#D1FAE5",
    "sev_nenhum_fg": "#065F46",
    "sev_baixa_bg": "#E0F2FE",
    "sev_baixa_fg": "#0369A1",
    "sev_media_bg": "#FEF3C7",
    "sev_media_fg": "#92400E",
    "sev_alta_bg": "#FEE2E2",
    "sev_alta_fg": "#991B1B",
}


def escolher_familia_ui() -> str:
    familias = set(QFontDatabase().families())
    for cand in ("Segoe UI", "Ubuntu Sans", "Noto Sans", "Ubuntu", "Liberation Sans", "DejaVu Sans"):
        if cand in familias:
            return cand
    return QFont().defaultFamily()


def escolher_familia_mono() -> str:
    familias = set(QFontDatabase().families())
    for cand in ("JetBrains Mono", "Consolas", "Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono"):
        if cand in familias:
            return cand
    return "Monospace"


def bgr_para_qimage(img_bgr: np.ndarray) -> QImage:
    """Converte matriz OpenCV BGR uint8 para QImage RGB independente em memória."""
    if img_bgr.ndim == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return qimg.copy()


def gerar_folha_estilo(familia_ui: str, familia_mono: str) -> str:
    return f"""
    QMainWindow, QDialog {{
        background-color: {CORES['bg_app']};
        color: {CORES['text']};
        font-family: "{familia_ui}";
        font-size: 13px;
    }}
    QWidget {{
        font-family: "{familia_ui}";
        color: {CORES['text']};
    }}
    QFrame#HeaderBar {{
        background-color: {CORES['bg_header']};
        border-bottom: 2px solid #F59E0B;
    }}
    QLabel#BrandBadge {{
        background-color: #F59E0B;
        color: #111827;
        font-weight: 800;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
    }}
    QLabel#HeaderTitle {{
        color: #F8FAFC;
        font-size: 15px;
        font-weight: 700;
    }}
    QLabel#PerfPill {{
        background-color: #212834;
        color: #E2E8F0;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 4px 11px;
        font-family: "{familia_mono}";
        font-size: 11px;
        font-weight: 600;
    }}
    QPushButton[navTab="true"] {{
        background-color: transparent;
        color: #CBD5E1;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton[navTab="true"]:hover {{
        background-color: #242C38;
        color: #FFFFFF;
    }}
    QPushButton[navTab="true"]:checked {{
        background-color: #F59E0B;
        color: #111827;
        font-weight: 700;
    }}

    /* Cards brancos */
    QFrame[card="true"] {{
        background-color: {CORES['bg_card']};
        border: 1px solid {CORES['border']};
        border-radius: 10px;
    }}
    QFrame[subtleBox="true"] {{
        background-color: {CORES['bg_subtle']};
        border: 1px solid #E2E8F0;
        border-radius: 7px;
    }}

    /* Botões gerais */
    QPushButton {{
        background-color: #FFFFFF;
        color: {CORES['text']};
        border: 1px solid {CORES['border_strong']};
        border-radius: 6px;
        padding: 5px 11px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: #F1F5F9;
        border-color: #94A3B8;
    }}
    QPushButton:pressed {{
        background-color: #E2E8F0;
    }}
    QPushButton:disabled {{
        color: #94A3B8;
        background-color: #F8FAFC;
        border-color: #E2E8F0;
    }}
    QPushButton[variant="primary"] {{
        background-color: {CORES['teal']};
        color: #FFFFFF;
        border: 1px solid {CORES['teal']};
    }}
    QPushButton[variant="primary"]:hover {{
        background-color: {CORES['teal_hover']};
        border-color: {CORES['teal_hover']};
    }}
    QPushButton[variant="amber"] {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #F59E0B;
    }}
    QPushButton[variant="amber"]:hover {{
        background-color: #FDE68A;
    }}

    /* Botões de sub-etapas e modos segmentados */
    QPushButton[pill="true"] {{
        background-color: #F8FAFC;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 4px 9px;
        font-size: 11.5px;
        font-weight: 600;
    }}
    QPushButton[pill="true"]:hover {{
        background-color: #E2E8F0;
    }}
    QPushButton[pill="true"]:checked {{
        background-color: {CORES['teal']};
        color: #FFFFFF;
        border-color: {CORES['teal']};
    }}
    QPushButton[iconBtn="true"] {{
        padding: 3px 0px;
        min-width: 26px;
        max-width: 26px;
        font-size: 13px;
        font-weight: 800;
    }}

    /* Stepper do Pipeline */
    QPushButton[stepBtn="true"] {{
        background-color: #FFFFFF;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 6px 10px;
        text-align: left;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton[stepBtn="true"]:hover {{
        border-color: {CORES['teal']};
        background-color: #F8FAFC;
    }}
    QPushButton[stepBtn="true"][branch="teal"]:checked {{
        background-color: {CORES['teal']};
        color: #FFFFFF;
        border: 2px solid #093540;
    }}
    QPushButton[stepBtn="true"][branch="amber"]:checked {{
        background-color: #B45309;
        color: #FFFFFF;
        border: 2px solid #78350F;
    }}

    /* Combos e entradas */
    QComboBox, QLineEdit {{
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 4px 9px;
        font-size: 12px;
        min-height: 22px;
    }}
    QComboBox:hover, QLineEdit:focus {{
        border-color: {CORES['teal']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QCheckBox {{
        font-size: 12px;
        font-weight: 600;
        spacing: 6px;
    }}

    /* Abas laterais */
    QTabWidget::pane {{
        border: 1px solid {CORES['border']};
        border-radius: 8px;
        background: #FFFFFF;
        top: -1px;
    }}
    QTabBar::tab {{
        background: #F1F5F9;
        color: #475569;
        border: 1px solid {CORES['border']};
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 6px 12px;
        margin-right: 3px;
        font-size: 12px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background: #FFFFFF;
        color: {CORES['teal']};
        font-weight: 700;
    }}

    /* Tabelas */
    QTableWidget {{
        background-color: #FFFFFF;
        alternate-background-color: #F8FAFC;
        gridline-color: #E2E8F0;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        font-size: 12px;
        selection-background-color: {CORES['teal_soft']};
        selection-color: #0F172A;
    }}
    QHeaderView::section {{
        background-color: #F1F5F9;
        color: #334155;
        font-weight: 700;
        font-size: 11px;
        border: none;
        border-bottom: 1px solid #CBD5E1;
        border-right: 1px solid #E2E8F0;
        padding: 6px 8px;
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        border: 1px solid #CBD5E1;
        height: 6px;
        background: #E2E8F0;
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {CORES['teal']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: #FFFFFF;
        border: 2px solid {CORES['teal']};
        width: 14px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: #F59E0B;
        border-color: #B45309;
    }}

    /* Scrollbars discretas */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 9px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: #CBD5E1;
        min-height: 28px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #94A3B8;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    """
