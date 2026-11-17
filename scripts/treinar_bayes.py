"""Classificador CLÁSSICO de blocos: Bayes gaussiano + seleção de descritores para frente (validação).
Uso: python scripts/treinar_bayes.py  -> modelo_bayes.json, outputs/avaliacao_bayes.json"""
import itertools, json, os, time

import numpy as np

import _caminho
from pavimento.prototipos import CLASSES, DESCRITORES, BayesGaussiano

Q = BayesGaussiano.Q


def f1(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precisao": round(p, 4), "revocacao": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def ajustar_vies(S, y):
    best = (-1, 0, 0)
    for bt, bb in itertools.product(np.arange(-4, 8.01, 0.5), np.arange(-4, 8.01, 0.5)):
        p = (S + np.array([0, bt * Q, bb * Q])).argmax(1)
        n = f1(p == 1, y == 1)["f1"] + f1(p == 2, y == 2)["f1"]
        if n > best[0]:
            best = (n, int(bt * Q), int(bb * Q))
    return best


def main():
    d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
    X, y, img, teste = d["X"], d["y"], d["img"], d["teste"]
    val = (~teste) & (img % 5 == 0); tr = (~teste) & ~val
    sel, resto, atual = [], DESCRITORES[:], 0.0
    t0 = time.time()
    while resto and time.time() - t0 < 140:
        cand = []
        for c in resto:
            m = BayesGaussiano(sel + [c]).treinar(X[tr], y[tr])
            cand.append((ajustar_vies(m.escores(X[val]), y[val])[0], c))
        n, c = max(cand)
        if n <= atual + 0.003:
            break
        atual = n; sel.append(c); resto.remove(c)
        print(f" + {c}: {n:.4f}", flush=True)
    m = BayesGaussiano(sel).treinar(X[tr], y[tr])
    _, bt, bb = ajustar_vies(m.escores(X[val]), y[val])
    m.vies = {"normal": 0, "trinca": bt, "buraco": bb}
    ct = m.classificar(X[teste]); cv = m.classificar(X[val])
    rel = {"descritores": sel, "vies_2^12": m.vies,
           "validacao": {"trinca": f1(cv == 1, y[val] == 1), "buraco": f1(cv == 2, y[val] == 2)},
           "teste": {"trinca": f1(ct == 1, y[teste] == 1), "buraco": f1(ct == 2, y[teste] == 2), "dano": f1(ct > 0, y[teste] > 0)}}
    json.dump(m.para_dict(), open(os.path.join(_caminho.RAIZ, "modelo_bayes.json"), "w"), indent=1)
    json.dump(rel, open(os.path.join(_caminho.SAIDAS, "avaliacao_bayes.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
