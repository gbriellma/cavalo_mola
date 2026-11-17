"""Descrição e representação de regiões (descritores de forma) a partir de momentos."""
from __future__ import annotations

import cv2
import numpy as np

from .morfologia import afinamento_zhang_suen, pontos_esqueleto


def descrever(comp: np.ndarray, com_esqueleto: bool = True) -> dict:
    """Calcula descritores de uma região binária (recorte do componente)."""
    m = comp.astype(np.uint8)
    area = int(m.sum())
    M = cv2.moments(m, binaryImage=True)
    mu20, mu02, mu11 = M["mu20"] / area, M["mu02"] / area, M["mu11"] / area
    # Autovalores da matriz de covariância -> eixos principais (PCA)
    delta = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + delta) / 2, max((mu20 + mu02 - delta) / 2, 1e-9)
    alongamento = float(np.sqrt(l1 / l2))
    excentricidade = float(np.sqrt(1 - l2 / l1)) if l1 > 0 else 0.0
    # Orientação do eixo principal em graus: 0 = horizontal, 90 = vertical
    theta = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    angulo = float(abs(theta))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimetro = float(sum(cv2.arcLength(c, True) for c in cnts))
    hull = cv2.convexHull(np.vstack(cnts)) if cnts else None
    area_hull = float(cv2.contourArea(hull)) if hull is not None else area
    solidez = area / area_hull if area_hull > 0 else 1.0
    circularidade = 4 * np.pi * area / perimetro ** 2 if perimetro > 0 else 0.0
    d = {"area": area, "perimetro": perimetro, "alongamento": alongamento,
         "excentricidade": excentricidade, "angulo": angulo, "solidez": float(solidez),
         "circularidade": float(circularidade)}
    if com_esqueleto:
        esq = afinamento_zhang_suen(m)
        comp_esq = int(esq.sum())
        extremos, bif = pontos_esqueleto(esq)
        d.update({"comprimento": comp_esq, "largura_media": area / max(comp_esq, 1),
                  "extremidades": extremos, "bifurcacoes": bif, "esqueleto": esq})
    return d


def componentes(mask: np.ndarray, area_min: int = 1):
    """Itera sobre componentes conectados (8-viz) devolvendo (recorte, bbox, rótulo)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < area_min:
            continue
        yield lab[y:y + h, x:x + w] == i, (x, y, w, h), i, lab
