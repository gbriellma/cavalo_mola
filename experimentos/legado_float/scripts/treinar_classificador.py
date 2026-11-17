"""Treina os classificadores de regiões (trinca / buraco) no conjunto de TREINO.

1. Gera candidatos com limiares mais permissivos (alta revocação).
2. Rotula cada candidato pela sobreposição com as máscaras do dataset.
3. Treina regressão logística (NumPy) e escolhe o limiar de probabilidade que
   maximiza o F1 (ponderado por pixels) no treino.
Saídas: modelos.json e params.json (com os limiares de candidatos usados).

Uso: python scripts/treinar_classificador.py --n-imagens 400
"""
import argparse, json, os
from dataclasses import asdict, replace
from multiprocessing import Pool

import cv2
import numpy as np

import _caminho  # noqa: F401
from pavimento.classificador import FEAT_BURACO, FEAT_TRINCA, RegressaoLogistica, salvar_modelos, vetor
from pavimento.dataset import carregar, listar_amostras, particao
from pavimento.parametros import Parametros
from pavimento.realce import preprocessar
from pavimento.segmentacao import segmentar_buracos, segmentar_trincas

P = None


def _init(pd):
    global P
    P = Parametros(**pd)


def extrair(pasta):
    s = carregar(pasta)
    roi, gt, gb = s.mascaras["LANE"], s.mascaras["CRACK"], s.mascaras["POTHOLE"]
    pre = preprocessar(s.imagem, roi, P)
    gt_d = cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    out = {"trinca": [], "buraco": [], "gt_trinca": int(gt.sum()), "gt_buraco": int(gb.sum())}
    for nome, fn, ref, feats in (("trinca", segmentar_trincas, gt_d, FEAT_TRINCA),
                                 ("buraco", segmentar_buracos, gb, FEAT_BURACO)):
        kw = {"L": pre["L"]} if nome == "buraco" else {}
        r = fn(pre["realcada"], roi, P, sat=pre["saturacao"], candidatos=True, **kw)
        for d in r["candidatos"]:
            x, y, w, h = d["bbox"]
            sobre = int((ref[y:y + h, x:x + w] & d["mascara"]).sum())
            out[nome].append((vetor(d, feats).tolist(), d["area"], sobre))
    return out


def escolher_limiar(prob, area, sobre, gt_total):
    melhor = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 37):
        acc = prob >= t
        tp = sobre[acc].sum()
        fp = (area[acc] - sobre[acc]).sum()
        prec = tp / max(tp + fp, 1)
        rec = min(tp / max(gt_total, 1), 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > melhor[1]:
            melhor = (float(t), float(f1))
    return melhor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imagens", type=int, default=400)
    a = ap.parse_args()
    params_f = os.path.join(_caminho.RAIZ, "params.json")
    p = Parametros.carregar(params_f) if os.path.exists(params_f) else Parametros()
    # candidatos permissivos
    p = replace(p, trinca_k_std=max(p.trinca_k_std - 1.0, 1.5), buraco_k_std=max(p.buraco_k_std - 0.5, 1.0),
                buraco_area_min=min(p.buraco_area_min, 300))
    pastas = [d for d in listar_amostras(_caminho.DATASET) if particao(os.path.basename(d)) == "treino"]
    pastas = pastas[:: max(1, len(pastas) // a.n_imagens)][: a.n_imagens]
    with Pool(os.cpu_count(), initializer=_init, initargs=(asdict(p),)) as pool:
        res = pool.map(extrair, pastas, chunksize=4)
    modelos, rel = {}, {}
    for nome, frac_pos in (("trinca", 0.5), ("buraco", 0.3)):
        X = np.array([c[0] for r in res for c in r[nome]])
        area = np.array([c[1] for r in res for c in r[nome]], float)
        sobre = np.array([c[2] for r in res for c in r[nome]], float)
        y = (sobre / area >= frac_pos).astype(float)
        gt_total = sum(r["gt_" + nome] for r in res)
        m = RegressaoLogistica().treinar(X, y)
        t, f1 = escolher_limiar(m.prob(X), area, sobre, gt_total)
        m.limiar = t
        modelos[nome] = m
        feats = FEAT_TRINCA if nome == "trinca" else FEAT_BURACO
        rel[nome] = {"candidatos": int(len(y)), "positivos": int(y.sum()), "limiar": round(t, 3),
                     "f1_pixel_treino_aprox": round(f1, 4),
                     "pesos": {f: round(float(w), 3) for f, w in zip(feats, m.w)}}
    salvar_modelos(modelos, os.path.join(_caminho.RAIZ, "modelos.json"))
    p.salvar(params_f)
    os.makedirs(_caminho.SAIDAS, exist_ok=True)
    json.dump(rel, open(os.path.join(_caminho.SAIDAS, "treino_classificador.json"), "w"), indent=2)
    print(json.dumps(rel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
