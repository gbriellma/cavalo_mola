"""Pipeline completo: aquisição -> pré-processamento -> segmentação -> descrição ->
classificação -> quantificação -> visualização."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .parametros import Parametros
from .quantificacao import quantificar
from .realce import preprocessar
from .segmentacao import segmentar_buracos, segmentar_trincas


def estimar_roi(img_bgr: np.ndarray) -> np.ndarray:
    """ROI quando não há máscara de pista: pixels de baixa saturação (asfalto cinza)
    abaixo do horizonte aproximado, suavizados por morfologia."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] < 60).astype(np.uint8)
    m[: img_bgr.shape[0] // 3] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return m > 0


def processar(img_bgr, roi=None, p: Parametros | None = None, modelos: dict | None = None) -> dict:
    p = p or Parametros()
    modelos = modelos or {}
    t0 = time.perf_counter()
    if roi is None:
        roi = estimar_roi(img_bgr)
    pre = preprocessar(img_bgr, roi, p)
    trincas = segmentar_trincas(pre["realcada"], roi, p, sat=pre["saturacao"], modelo=modelos.get("trinca"))
    buracos = segmentar_buracos(pre["realcada"], roi, p, sat=pre["saturacao"], modelo=modelos.get("buraco"), L=pre["L"])
    # Resolução de conflito: pixels em buraco não contam como trinca
    trincas["mascara"] &= ~buracos["mascara"]
    q = quantificar(trincas, buracos, roi)
    q["tempo_s"] = round(time.perf_counter() - t0, 3)
    return {"roi": roi, "pre": pre, "trincas": trincas, "buracos": buracos, "quant": q}
