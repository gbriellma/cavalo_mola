"""Morfologia matemática implementada com NumPy (afinamento, preenchimento, análise de esqueleto)."""
from __future__ import annotations

import cv2
import numpy as np


def _vizinhos(img: np.ndarray):
    """Retorna P2..P9 (sentido horário a partir do norte) para cada pixel."""
    p = np.pad(img, 1)
    return [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]


def afinamento_zhang_suen(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Esqueletização de Zhang-Suen (1984), vetorizada em NumPy."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        mudou = False
        for passo in (0, 1):
            P = _vizinhos(img)
            B = sum(P)
            seq = P + [P[0]]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            P2, P3, P4, P5, P6, P7, P8, P9 = P
            if passo == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            rem = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if rem.any():
                img[rem] = 0
                mudou = True
        if not mudou:
            break
    return img.astype(bool)


def pontos_esqueleto(esq: np.ndarray) -> tuple[int, int]:
    """Conta extremidades (1 vizinho) e bifurcações (>=3 vizinhos) do esqueleto."""
    e = esq.astype(np.uint8)
    n = sum(_vizinhos(e))
    return int(((n == 1) & esq).sum()), int(((n >= 3) & esq).sum())


def preencher_buracos(mask: np.ndarray) -> np.ndarray:
    """Preenchimento de buracos por reconstrução a partir da borda (flood fill)."""
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return (m | cv2.bitwise_not(ff)) > 0


def elemento(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
