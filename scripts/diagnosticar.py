"""Diagnóstico de buracos restrito ao treino, sem executar ao importar."""
import argparse
import json
import os

import _caminho
import cv2
import numpy as np
from _comum import preparar, somente_treino, validar_amostras
from pavimento.dataset import listar_amostras, particao
from pavimento.nucleo import analisar
from pavimento.parametros import Parametros


def componentes(mask, g, roi):
    """Escurecimento = mediana da pista - média do componente (níveis 0..255)."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    mediana = float(np.median(g[roi])) if roi.any() else 0.0
    saida = []
    for i in range(1, n):
        m = labels == i
        contornos, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimetro = sum(cv2.arcLength(c, True) for c in contornos)
        area_contorno = sum(cv2.contourArea(c) for c in contornos)
        hull = cv2.convexHull(np.concatenate(contornos))
        area_hull = cv2.contourArea(hull)
        saida.append({"area_px": int(m.sum()), "escurecimento_medio": mediana - float(g[m].mean()),
                      "circularidade": 4 * np.pi * area_contorno / perimetro**2 if perimetro else 0.0,
                      "solidez": area_contorno / area_hull if area_hull else 0.0})
    return saida


def diagnosticar(dados, p):
    nome, g, roi, _, gb = dados
    gt = gb & roi
    pred = analisar(g, roi, p)["buracos"] & roi
    return {"nome": nome, "fp_px": int((pred & ~gt).sum()), "fn_px": int((gt & ~pred).sum()),
            "area_gt_px": int(gt.sum()), "area_prevista_px": int(pred.sum()),
            "componentes_gt": componentes(gt, g, roi), "componentes_previstos": componentes(pred, g, roi)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--particao", choices=["treino"], default="treino")
    ap.parse_args()
    pastas = [d for d in listar_amostras(_caminho.dataset_atual()) if particao(os.path.basename(d)) == "treino"]
    somente_treino(pastas)
    validar_amostras(pastas)
    p = Parametros.carregar(_caminho.PARAMS) if os.path.exists(_caminho.PARAMS) else Parametros()
    res = [diagnosticar(preparar(d), p) for d in pastas]
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
    ag = {k: sum(r[k] for r in res) for k in ("fp_px", "fn_px", "area_gt_px", "area_prevista_px")}
    for grupo in ("componentes_gt", "componentes_previstos"):
        cs = [c for r in res for c in r[grupo]]
        ag[grupo] = {"n": len(cs), **{k: float(np.mean([c[k] for c in cs])) if cs else None
                        for k in ("escurecimento_medio", "circularidade", "solidez")}}
    print(json.dumps({"agregado": ag}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}")
