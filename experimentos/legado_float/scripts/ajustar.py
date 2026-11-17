"""Ajuste de parâmetros por busca aleatória no conjunto de TREINO.

O pré-processamento é calculado uma vez (cache) e só a segmentação varia.
Trincas e buracos são ajustados separadamente. Resultado: params.json na raiz.

Uso: python scripts/ajustar.py --n-imagens 60 --iter 30 [--alvo trinca|buraco|ambos]
(pode ser executado várias vezes: o melhor resultado é mantido em outputs/ajuste_estado.json)
"""
import argparse, json, os, random
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

import _caminho  # noqa: F401
from pavimento.dataset import carregar, listar_amostras, particao
from pavimento.metricas import confusao, deteccao_imagem, resumir
from pavimento.parametros import Parametros
from pavimento.realce import preprocessar
from pavimento.segmentacao import segmentar_buracos, segmentar_trincas

ESPACO_T = {"trinca_bh_k": [9, 11, 15, 21], "trinca_k_std": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "trinca_area_min": [30, 60, 100, 150], "trinca_alongamento_min": [2.0, 3.0, 4.0, 5.0],
            "trinca_comprimento_min": [15, 25, 40, 60]}
ESPACO_B = {"buraco_tex_k": [15, 21, 31], "buraco_abertura": [5, 7, 11, 15],
            "buraco_k_std": [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0],
            "buraco_area_min": [300, 600, 1000, 2000], "buraco_solidez_min": [0.45, 0.6, 0.75, 0.85],
            "buraco_alongamento_max": [2.5, 3.0, 4.0], "buraco_textura_min": [0.0]}
CACHE = []


def _carregar_cache(pastas, p):
    out = []
    for d in pastas:
        s = carregar(d)
        roi = s.mascaras["LANE"]
        pre = preprocessar(s.imagem, roi, p)
        out.append((pre["realcada"], pre["saturacao"], roi, s.mascaras["CRACK"], s.mascaras["POTHOLE"], pre["L"]))
    return out


def _init(pastas, pdict):
    global CACHE
    CACHE = _carregar_cache(pastas, Parametros(**pdict))


def _avaliar(args):
    pdict, alvo = args
    p = Parametros(**pdict)
    ct, cb, dt, db = [0] * 4, [0] * 4, ([], []), ([], [])
    for real, sat, roi, gt, gb, L in CACHE:
        if alvo in ("trinca", "ambos"):
            m = segmentar_trincas(real, roi, p, sat)["mascara"]
            ct = [a + b for a, b in zip(ct, confusao(m, gt, roi, 5))]
            dt[0].append(m.sum() > 60); dt[1].append(gt.sum() > 50)
        if alvo in ("buraco", "ambos"):
            m = segmentar_buracos(real, roi, p, sat=sat, L=L)["mascara"]
            cb = [a + b for a, b in zip(cb, confusao(m, gb, roi))]
            db[0].append(m.any()); db[1].append(gb.sum() > 300)
    return ct, cb, dt, db


def pontuar(ct, cb, dt, db, alvo):
    s = 0.0
    if alvo in ("trinca", "ambos"):
        s += resumir(*ct)["f1"] + 0.3 * deteccao_imagem(*dt)["f1"]
    if alvo in ("buraco", "ambos"):
        s += resumir(*cb)["f1"] + 0.5 * deteccao_imagem(*db)["f1"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imagens", type=int, default=60)
    ap.add_argument("--iter", type=int, default=30)
    ap.add_argument("--alvo", default="ambos", choices=["trinca", "buraco", "ambos"])
    ap.add_argument("--semente", type=int, default=None)
    a = ap.parse_args()
    estado_f = os.path.join(_caminho.SAIDAS, "ajuste_estado.json")
    params_f = os.path.join(_caminho.RAIZ, "params.json")
    estado = json.load(open(estado_f)) if os.path.exists(estado_f) else {}
    base = Parametros.carregar(params_f) if os.path.exists(params_f) else Parametros()
    pastas = [d for d in listar_amostras(_caminho.DATASET) if particao(os.path.basename(d)) == "treino"]
    pastas = pastas[:: len(pastas) // a.n_imagens][: a.n_imagens]
    rng = random.Random(a.semente)
    alvos = ["trinca", "buraco"] if a.alvo == "ambos" else [a.alvo]
    with Pool(os.cpu_count(), initializer=_init, initargs=(pastas, asdict(base))) as pool:
        for alvo in alvos:
            esp = ESPACO_T if alvo == "trinca" else ESPACO_B
            cands = [base] + [replace(base, **{k: rng.choice(v) for k, v in esp.items()}) for _ in range(a.iter)]
            # cada processo tem o cache completo: distribui candidatos
            res = pool.map(_avaliar, [(asdict(c), alvo) for c in cands])
            notas = [pontuar(*r, alvo) for r in res]
            i = int(np.argmax(notas))
            melhor_ant = estado.get(alvo, {}).get("nota", -1)
            print(f"[{alvo}] base={notas[0]:.4f} melhor={notas[i]:.4f} (anterior {melhor_ant:.4f})")
            if notas[i] > melhor_ant:
                base = replace(base, **{k: getattr(cands[i], k) for k in esp})
                estado[alvo] = {"nota": notas[i],
                                "params": {k: getattr(cands[i], k) for k in esp},
                                "metricas_px": resumir(*(res[i][0] if alvo == "trinca" else res[i][1]))}
    base.salvar(params_f)
    os.makedirs(_caminho.SAIDAS, exist_ok=True)
    json.dump(estado, open(estado_f, "w"), indent=2)
    print(json.dumps(estado, indent=2))


if __name__ == "__main__":
    main()
