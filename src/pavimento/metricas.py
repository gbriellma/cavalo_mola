"""Métricas de avaliação pixel a pixel e por imagem."""
from __future__ import annotations

import cv2
import numpy as np


def confusao(pred: np.ndarray, gt: np.ndarray, roi=None, tolerancia: int = 0):
    """TP/FP/FN em pixels. Com tolerância (px), usa dilatação para casar
    predições próximas ao rótulo (comum em trincas, cujo rótulo é impreciso)."""
    if roi is not None:
        pred, gt = pred & roi, gt & roi
    if tolerancia > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerancia + 1,) * 2)
        gt_d = cv2.dilate(gt.astype(np.uint8), k) > 0
        pr_d = cv2.dilate(pred.astype(np.uint8), k) > 0
        tp_p = int((pred & gt_d).sum())          # predições corretas
        fp = int((pred & ~gt_d).sum())
        tp_g = int((gt & pr_d).sum())            # rótulos recuperados
        fn = int((gt & ~pr_d).sum())
        return tp_p, fp, tp_g, fn
    tp = int((pred & gt).sum())
    return tp, int((pred & ~gt).sum()), tp, int((~pred & gt).sum())


def resumir(tp_p, fp, tp_g, fn) -> dict:
    prec = tp_p / (tp_p + fp) if tp_p + fp else 0.0
    rec = tp_g / (tp_g + fn) if tp_g + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp_p / (tp_p + fp + fn) if tp_p + fp + fn else 0.0
    return {"precisao": round(prec, 4), "revocacao": round(rec, 4), "f1": round(f1, 4), "iou": round(iou, 4)}


def deteccao_imagem(preds: list[bool], gts: list[bool]) -> dict:
    p, g = np.array(preds), np.array(gts)
    tp, fp, fn, tn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()), int((~p & ~g).sum())
    r = resumir(tp, fp, tp, fn)
    r["acuracia"] = round((tp + tn) / max(len(p), 1), 4)
    r.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return r
