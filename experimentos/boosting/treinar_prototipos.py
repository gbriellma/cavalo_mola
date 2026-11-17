"""Treina o classificador CLÁSSICO de blocos (K-means + distância mínima) e ajusta, na validação,
o número de protótipos e os pesos das classes. Avalia no teste e compara com o boosting.
Uso: python scripts/treinar_prototipos.py   -> modelo_prototipos.json, outputs/avaliacao_prototipos.json
"""
import itertools, json, os

import numpy as np

import _caminho
from pavimento.prototipos import CLASSES, DistanciaMinima


def f1(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precisao": round(p, 4), "revocacao": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def avaliar(m, X, y):
    c = m.classificar(X)
    return {"trinca": f1(c == 1, y == 1), "buraco": f1(c == 2, y == 2), "dano": f1(c > 0, y > 0)}


def main():
    d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
    X, y, img, teste = d["X"], d["y"], d["img"], d["teste"]
    val = (~teste) & (img % 5 == 0); tr = (~teste) & ~val
    melhor = None
    for k in [(4, 3, 2), (8, 6, 4), (16, 8, 4), (24, 12, 6)]:
        m = DistanciaMinima().treinar(X[tr], y[tr], k=k)
        Dv = m.distancias(X[val])
        for pt, pb in itertools.product(range(40, 161, 10), range(40, 161, 10)):
            pred = (Dv * np.array([100, pt, pb])[None, :]).argmin(1)
            nota = f1(pred == 1, y[val] == 1)["f1"] + f1(pred == 2, y[val] == 2)["f1"]
            if melhor is None or nota > melhor[0]:
                melhor = (nota, k, pt, pb)
        print("k", k, "melhor até agora", melhor, flush=True)
    _, k, pt, pb = melhor
    m = DistanciaMinima().treinar(X[tr], y[tr], k=k)
    m.peso = {"normal": 100, "trinca": pt, "buraco": pb}
    rel = {"k_prototipos": dict(zip(CLASSES, k)), "pesos": m.peso,
           "validacao": avaliar(m, X[val], y[val]), "teste": avaliar(m, X[teste], y[teste])}
    try:
        rel["comparacao_boosting_teste"] = {k2: json.load(open(os.path.join(_caminho.SAIDAS, "avaliacao_blocos.json")))[k2]["teste"]["f1"]
                                            for k2 in ("trinca", "buraco")}
    except Exception:
        pass
    json.dump(m.para_dict(), open(os.path.join(_caminho.RAIZ, "modelo_prototipos.json"), "w"))
    json.dump(rel, open(os.path.join(_caminho.SAIDAS, "avaliacao_prototipos.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
