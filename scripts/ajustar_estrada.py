"""Ajuste determinístico (busca coordenada) dos parâmetros da detecção da estrada no
TREINO do RTK. Métrica: IoU médio da máscara de estrada (classes 1..12 do RTK).
Uso: python scripts/ajustar_estrada.py
"""
import json, os
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

import _caminho
from _rtk import carregar, listar
from pavimento.estrada import ParamEstrada, crescer_estrada, saturacao_hsi
from pavimento.nucleo import intensidade_hsi

GRADE = {"bloco": [8, 12, 16], "tol_i": [40, 56, 72, 90, 120], "tol_s": [10, 20, 30, 45, 60, 90],
         "tol_d": [3, 6, 10, 16, 24, 40], "semente_altura": [3, 4, 6, 8], "semente_larg_pct": [20, 30, 40],
         "fecho": [1, 3, 5, 7]}
DADOS = []
ARQ = os.path.join(_caminho.RAIZ, "params_estrada.json")


def _init(nomes):
    global DADOS
    DADOS = []
    for n in nomes:
        bgr, lab = carregar(n)
        rgb = bgr[..., ::-1]
        DADOS.append((intensidade_hsi(rgb), saturacao_hsi(rgb), lab > 0))


def _nota(pd):
    p = ParamEstrada(**pd)
    ious = []
    for i_img, s_img, gt in DADOS:
        m, _ = crescer_estrada(i_img, s_img, p)
        ious.append((m & gt).sum() / max((m | gt).sum(), 1))
    return float(np.mean(ious))


def main():
    p = ParamEstrada(**json.load(open(ARQ))) if os.path.exists(ARQ) else ParamEstrada()
    nomes = listar("train")
    with Pool(os.cpu_count(), initializer=_init, initargs=(nomes,)) as pool:
        melhor = pool.apply(_nota, (asdict(p),))
        print(f"inicial IoU {melhor:.4f}", flush=True)
        for _ in range(2):
            for k, vals in GRADE.items():
                cands = [replace(p, **{k: v}) for v in vals if v != getattr(p, k)]
                notas = pool.map(_nota, [asdict(c) for c in cands])
                for c, n in zip(cands, notas):
                    if n > melhor + 1e-9:
                        melhor, p = n, c
                print(f"{k}={getattr(p, k)}  IoU={melhor:.4f}", flush=True)
    json.dump(asdict(p), open(ARQ, "w"), indent=2)
    print(json.dumps(asdict(p)))


if __name__ == "__main__":
    main()
