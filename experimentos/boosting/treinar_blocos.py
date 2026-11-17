"""Treina os classificadores de blocos (trinca × resto, buraco × resto) com boosting do zero.

Divisão: imagens de TESTE (hash do nome) ficam de fora; das imagens de treino, 1 em cada 5
vira validação (parada antecipada e escolha do limiar). Blocos normais são subamostrados
(1 em cada 3, determinístico) no treino. Saída: modelos_blocos.json + outputs/avaliacao_blocos.json
Uso: python scripts/treinar_blocos.py
"""
import json, os
from multiprocessing import Pool

import numpy as np

import _caminho
from pavimento.blocos import NOMES
from pavimento.boosting import Boosting, salvar


def f1(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precisao": round(p, 4), "revocacao": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "tp": tp, "fp": fp, "fn": fn}


def _treinar(args):
    classe, X, y, img, teste = args
    alvo = (y == classe)
    val = (~teste) & (img % 5 == 0)
    tr = (~teste) & ~val
    neg = tr & ~alvo
    neg_idx = np.flatnonzero(neg)[::3]
    idx = np.sort(np.concatenate([np.flatnonzero(tr & alvo), neg_idx]))
    m = Boosting(n_arvores=300, profundidade=3, taxa=0.15).treinar(
        X[idx], alvo[idx].astype(float), X[val], alvo[val].astype(float), paciencia=30)
    sv = m.pontuar(X[val])
    # limiar inteiro que maximiza F1 na validação
    cands = np.unique(np.percentile(sv, np.linspace(50, 99.9, 200)).astype(np.int64))
    best = max(cands, key=lambda t: f1(sv >= t, alvo[val])["f1"])
    st = m.pontuar(X[teste])
    return classe, m, int(best), f1(sv >= best, alvo[val]), f1(st >= best, alvo[teste]), st


def main():
    d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
    X, y, img, teste = d["X"], d["y"], d["img"], d["teste"]
    with Pool(2) as pool:
        res = pool.map(_treinar, [(1, X, y, img, teste), (2, X, y, img, teste)])
    modelos, rel, pred = {}, {}, {}
    for classe, m, lim, fv, ft, st in res:
        nome = {1: "trinca", 2: "buraco"}[classe]
        modelos[nome] = m; modelos[nome + "_limiar"] = lim
        rel[nome] = {"arvores": len(m.arvores), "limiar": lim, "validacao": fv, "teste": ft}
        pred[nome] = st >= lim
    dano_pred = pred["trinca"] | pred["buraco"]
    rel["dano_qualquer_teste"] = f1(dano_pred, y[teste] > 0)
    rel["referencia"] = "Fernandes, Passos et al. (2023), mesmo dataset: F=75,16% (blocos com trinca/buraco)"
    # importância: quantas vezes cada descritor é usado nas divisões
    uso = {}
    for nome in ("trinca", "buraco"):
        for arv in modelos[nome].arvores:
            for no in arv:
                if "f" in no:
                    uso[NOMES[no["f"]]] = uso.get(NOMES[no["f"]], 0) + 1
    rel["descritores_mais_usados"] = dict(sorted(uso.items(), key=lambda x: -x[1])[:12])
    salvar(modelos, os.path.join(_caminho.RAIZ, "modelos_blocos.json"))
    json.dump(rel, open(os.path.join(_caminho.SAIDAS, "avaliacao_blocos.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
