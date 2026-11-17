"""Visualização: sobreposições e painel de etapas intermediárias (somente OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

COR_TRINCA = (0, 0, 255)     # vermelho (BGR)
COR_BURACO = (255, 0, 255)   # magenta
COR_GT = (0, 255, 0)         # verde


def sobrepor(img, trincas, buracos, roi=None, alpha=0.55):
    out = img.copy()
    if roi is not None:
        cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 200, 0), 1)
    cor = out.copy()
    cor[trincas] = COR_TRINCA
    cor[buracos] = COR_BURACO
    out = cv2.addWeighted(cor, alpha, out, 1 - alpha, 0)
    cnts, _ = cv2.findContours(buracos.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, COR_BURACO, 2)
    return out


def comparar_gt(img, pred_t, pred_b, gt_t, gt_b):
    """Verde = rótulo; vermelho/magenta = predição; amarelo = acerto."""
    out = (img * 0.5).astype(np.uint8)
    for pred, gt in ((pred_t, gt_t), (pred_b, gt_b)):
        out[gt & ~pred] = COR_GT
        out[pred & ~gt] = (0, 0, 255)
        out[pred & gt] = (0, 255, 255)
    return out


def _rotulo(img, texto):
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, texto, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def painel(etapas: list[tuple[str, np.ndarray]], colunas: int = 3, largura: int = 512):
    """Monta uma grade com as etapas do pipeline."""
    tiles = []
    for nome, im in etapas:
        if im.dtype == bool:
            im = im.astype(np.uint8) * 255
        h = int(im.shape[0] * largura / im.shape[1])
        tiles.append(_rotulo(cv2.resize(im, (largura, h), interpolation=cv2.INTER_AREA), nome))
    h = tiles[0].shape[0]
    while len(tiles) % colunas:
        tiles.append(np.zeros((h, largura, 3), np.uint8))
    linhas = [np.hstack(tiles[i:i + colunas]) for i in range(0, len(tiles), colunas)]
    return np.vstack(linhas)


def legenda_quantificacao(img, q: dict):
    out = img.copy()
    linhas = [f"Indice de dano: {q['indice_dano']} ({q['severidade']})",
              f"Trincas: {q['n_trincas']}  comp={q['comprimento_trincas_px']}px  larg={q['largura_media_trincas_px']}px",
              f"Buracos: {q['n_buracos']}  area={q['area_buracos_pct_pista']}% da pista"]
    cv2.rectangle(out, (0, out.shape[0] - 26 * len(linhas) - 8), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    for i, t in enumerate(linhas):
        cv2.putText(out, t, (10, out.shape[0] - 26 * (len(linhas) - i) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
