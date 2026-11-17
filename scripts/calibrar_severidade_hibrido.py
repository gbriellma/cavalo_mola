"""Calibra os limiares de severidade aplicados ao índice PREVISTO pelo híbrido, nas imagens de
VALIDAÇÃO: escolhe (t_media, t_alta) que maximizam a concordância exata com a severidade
calculada pelas máscaras rotuladas (desempate: concordância até 1 nível). Grava em params_pixel.json.
Uso: python scripts/calibrar_severidade_hibrido.py
"""
import json, os
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

import _caminho
from _comum import preparar, referencia_rotulo
from pavimento.boosting import carregar
from pavimento.dataset import particao
from pavimento.hibrido import carregar_classificador, ParamPixel, analisar_hibrido
from pavimento.parametros import Parametros

ARQ = os.path.join(_caminho.RAIZ, os.environ.get("PARAMS_PIXEL", "params_pixel.json"))
NIV = {"nenhum": 0, "baixa": 1, "media": 2, "alta": 3}


def _uma(nome):
    p = Parametros.carregar(_caminho.PARAMS)
    pp = replace(ParamPixel(**json.load(open(ARQ))), sev_media=0, sev_alta=0)
    m = carregar_classificador(_caminho.RAIZ, os.environ.get("CLASSIFICADOR", "bayes"))
    d = preparar(os.path.join(_caminho.DATASET, nome))
    r = analisar_hibrido(d[1], d[2], p, m, pp)
    return r["quant"]["indice_dano_1000"], NIV[referencia_rotulo(d, p)["severidade"]]


def nivel(ind, t0, t1, t2):
    return np.where(ind < t0, 0, np.where(ind < t1, 1, np.where(ind < t2, 2, 3)))


def main():
    p = Parametros.carregar(_caminho.PARAMS)
    todos = sorted(f[:-4] for f in os.listdir(_caminho.CACHE) if f.endswith(".npz"))
    nomes = [n for i, n in enumerate(todos) if particao(n) == "treino" and i % 5 == 0]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(_uma, nomes)
    ind = np.array([r[0] for r in res]); ref = np.array([r[1] for r in res])
    cands = np.unique(np.percentile(ind, np.arange(2, 99, 2)).astype(int))
    melhor = None
    for i, t1 in enumerate(cands):
        for t2 in cands[i + 1:]:
            pr = nivel(ind, p.sev_min_1000, t1, t2)
            chave = ((pr == ref).mean(), (np.abs(pr - ref) <= 1).mean())
            if melhor is None or chave > melhor[0]:
                melhor = (chave, int(t1), int(t2))
    (ac, ac1), t1, t2 = melhor
    pp = replace(ParamPixel(**json.load(open(ARQ))), sev_media=t1, sev_alta=t2)
    json.dump(asdict(pp), open(ARQ, "w"), indent=2)
    print(f"{len(res)} imagens de validação | limiares do índice previsto: média>={t1}, alta>={t2} "
          f"| acerto {ac:.3f}, até 1 nível {ac1:.3f}")


if __name__ == "__main__":
    main()
