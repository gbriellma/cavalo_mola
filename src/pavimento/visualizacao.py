"""Visualização (somente para relatório/slides; não faz parte do núcleo determinístico)."""
from __future__ import annotations

import cv2
import numpy as np


def _bgr(im):
    if im.dtype == bool:
        im = im.astype(np.uint8) * 255
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def sobrepor(g, roi, trincas, esqueleto, buracos):
    out = _bgr(g).copy()
    cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 200, 0), 1)
    out[trincas] = (0, 0, 220)
    out[esqueleto] = (0, 255, 255)
    cnts, _ = cv2.findContours(buracos.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 0, 255), 2)
    return out


def comparar(g, pred_t, pred_b, gt_t, gt_b):
    """verde = rótulo não detectado; vermelho = falso positivo; amarelo = acerto."""
    out = (_bgr(g) * 0.5).astype(np.uint8)
    for pr, gt in ((pred_t, gt_t), (pred_b, gt_b)):
        out[gt & ~pr] = (0, 200, 0)
        out[pr & ~gt] = (0, 0, 255)
        out[pr & gt] = (0, 255, 255)
    return out


def _rotulo(im, txt):
    im = _bgr(im).copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def painel(etapas, colunas=3):
    tiles = [_rotulo(im, t) for t, im in etapas]
    h, w = tiles[0].shape[:2]
    tiles = [cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST) for t in tiles]
    while len(tiles) % colunas:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + colunas]) for i in range(0, len(tiles), colunas)])


def legenda(im, q):
    im = im.copy()
    linhas = [f"Indice de dano: {q['indice_dano_1000']/10:.1f}/100 ({q['severidade']})",
              f"Trincas: {q['n_trincas']}  comp={q['comprimento_trincas_px']}px  larg={q['largura_media_trincas_16']/16:.1f}px",
              f"Buracos: {q['n_buracos']}  area={q['area_buracos_pct_100']/100:.2f}% da pista"]
    y0 = im.shape[0] - 20 * len(linhas) - 6
    cv2.rectangle(im, (0, y0), (im.shape[1], im.shape[0]), (0, 0, 0), -1)
    for i, t in enumerate(linhas):
        cv2.putText(im, t, (6, y0 + 16 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return im
