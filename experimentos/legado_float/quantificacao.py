"""Quantificação dos danos por imagem (comprimentos, larguras, áreas e índice)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .classificacao import severidade, tipo_trinca


def quantificar(trincas: dict, buracos: dict, roi: np.ndarray) -> dict:
    area_roi = max(int(roi.sum()), 1)
    regs_t = trincas["regioes"]
    for d in regs_t:
        d["tipo"] = tipo_trinca(d)
    comp_total = sum(d["comprimento"] for d in regs_t)
    area_t = int(trincas["mascara"].sum())
    area_b = int(buracos["mascara"].sum())
    tipos = Counter(d["tipo"] for d in regs_t)
    # Densidade de trincas (px de trinca por 1000 px de pista) e % de área com buraco
    dens = 1000 * comp_total / area_roi
    pct_b = 100 * area_b / area_roi
    # Índice de dano (0-100), heurístico: buracos pesam mais que trincas,
    # trincas em malha (couro de jacaré) pesam mais que isoladas
    peso_jacare = 1.0 + 0.5 * tipos.get("couro_de_jacare", 0) / max(len(regs_t), 1)
    indice = float(min(100, 4.0 * dens * peso_jacare + 25.0 * pct_b))
    return {
        "n_trincas": len(regs_t),
        "comprimento_trincas_px": int(comp_total),
        "largura_media_trincas_px": round(area_t / max(comp_total, 1), 2),
        "area_trincas_px": area_t,
        "densidade_trincas": round(dens, 3),
        "tipos_trinca": dict(tipos),
        "n_buracos": len(buracos["regioes"]),
        "area_buracos_px": area_b,
        "area_buracos_pct_pista": round(pct_b, 3),
        "maior_buraco_px": max((d["area"] for d in buracos["regioes"]), default=0),
        "indice_dano": round(indice, 2),
        "severidade": severidade(indice),
    }
