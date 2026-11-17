"""Ajuste determinístico (busca coordenada) dos parâmetros do ESTÁGIO DE PIXELS do híbrido,
nas imagens de VALIDAÇÃO do classificador de blocos (fora do teste e fora do treino das árvores). Objetivo: F1 de trincas (tol. 2 px)
+ F1 de buracos + correlação de Pearson do comprimento e da área com o rótulo
- 0,5·|log(comprimento total previsto/real)|.  Saída: params_pixel.json
Uso: python scripts/ajustar_pixel.py [--n 60]
"""
import argparse, json, os, time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

import _caminho
from _comum import preparar, somente_treino, validar_amostras
from pavimento.blocos import descritores
from pavimento.boosting import carregar
from pavimento.dataset import particao
from pavimento.hibrido import carregar_classificador, ParamPixel, analisar_hibrido
from pavimento.metricas import confusao, resumir
from pavimento.nucleo import afinar
from pavimento.parametros import Parametros

GRADE = {"a16": [6, 8, 10, 12, 16, 20], "b16": [0, 16, 32, 48, 64], "fecho": [1, 3, 5], "area_min": [4, 6, 10, 15, 25, 40],
         "comp_min": [2, 4, 6, 10, 14], "larg_max16": [64, 96, 128, 192], "margem": [0, 1, 2],
         "buraco_fecho": [1, 5, 9, 13], "buraco_area_min": [20, 40, 80, 150], "buraco_modo": [0, 1, 2], "buraco_desloc": [-8192, -4096, -2048, 0, 2048, 4096],
         "trinca_desloc": [-100000, -4096, -2048, 0, 2048],
         "mediana": [0, 1]}
D = []
ARQ = os.path.join(_caminho.RAIZ, os.environ.get("PARAMS_PIXEL", "params_pixel.json"))


def _init(pastas):
    global D, M, P
    somente_treino(pastas)
    validar_amostras(pastas)
    M = carregar_classificador(_caminho.RAIZ, os.environ.get("CLASSIFICADOR", "bayes"))
    P = Parametros.carregar(_caminho.PARAMS)
    D = []
    for pasta in pastas:
        _, g, roi, gt, gb = preparar(pasta)
        D.append((g, roi, gt, gb, int(afinar(gt & roi).sum()), int((gb & roi).sum())))


def _nota(pd):
    pp = ParamPixel(**pd)
    ct, cb, lp, lr, bp, br = [0] * 4, [0] * 4, [], [], [], []
    for g, roi, gt, gb, l_ref, b_ref in D:
        r = analisar_hibrido(g, roi, P, M, pp)
        ct = [a + b for a, b in zip(ct, confusao(r["trincas"], gt, roi, 2))]
        cb = [a + b for a, b in zip(cb, confusao(r["buracos"], gb, roi))]
        lp.append(r["quant"]["comprimento_trincas_px"]); lr.append(l_ref)
        bp.append(r["quant"]["area_buracos_px"]); br.append(b_ref)
    pc = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    vies = abs(float(np.log((sum(lp) + 1) / (sum(lr) + 1))))
    ft, fb = resumir(*ct)["f1"], resumir(*cb)["f1"]
    return ft + fb + pc(lp, lr) + pc(bp, br) - 0.5 * vies, {"f1_trinca_tol2": ft, "f1_buraco": fb, "pearson_comp": round(pc(lp, lr), 4),
                                                             "pearson_area": round(pc(bp, br), 4), "vies_log": round(vies, 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--params", nargs="*")
    a = ap.parse_args()
    if a.n <= 0:
        ap.error("--n deve ser positivo")
    raiz = _caminho.dataset_atual()
    if not os.path.isdir(_caminho.CACHE):
        raise ValueError(f"Cache não encontrado: {_caminho.CACHE}. Prepare as amostras antes do ajuste por pixel.")
    # imagens de VALIDAÇÃO do classificador de blocos (não usadas para aprender as árvores)
    # Manter os índices do cache usados na divisão de treinar_bayes.py.
    todos = sorted(f[:-4] for f in os.listdir(_caminho.CACHE) if f.endswith(".npz"))
    pastas = [os.path.join(raiz, n) for i, n in enumerate(todos) if particao(n) == "treino" and i % 5 == 0]
    # Falhar no processo principal, antes do Pool: exceções no initializer
    # fariam o Pool recriar trabalhadores indefinidamente. Cache não é dataset.
    somente_treino(pastas)
    validar_amostras(pastas)
    pastas = pastas[:: max(1, len(pastas) // a.n)][: a.n]
    pp = ParamPixel(**json.load(open(ARQ))) if os.path.exists(ARQ) else ParamPixel()
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=_init, initargs=(pastas,)) as pool:
        melhor, info = pool.apply(_nota, (asdict(pp),))
        print(f"inicial {melhor:.4f} {info}", flush=True)
        for k in (a.params or GRADE):
            if time.time() - t0 > 140:
                print("tempo esgotado; rode de novo com --params para continuar", flush=True); break
            cands = [replace(pp, **{k: v}) for v in GRADE[k] if v != getattr(pp, k)]
            for c, (n, inf) in zip(cands, pool.map(_nota, [asdict(c) for c in cands])):
                if n > melhor + 1e-9:
                    melhor, info, pp = n, inf, c
            json.dump(asdict(pp), open(ARQ, "w"), indent=2)
            print(f"{k}={getattr(pp, k)} nota={melhor:.4f} {info} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}")
