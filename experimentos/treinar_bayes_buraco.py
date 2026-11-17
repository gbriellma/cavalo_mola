"""Bayes gaussiano DEDICADO a buraco × resto, com seleção de descritores própria (validação).
Classes: 0 = sem buraco (normal ou trinca), 1 = buraco. Viés ajustado para o F1 de buraco.
Uso: python scripts/treinar_bayes_buraco.py -> modelo_bayes_buraco.json"""
import json, os, time

import numpy as np

import _caminho
from pavimento.prototipos import DESCRITORES, BayesGaussiano

Q = BayesGaussiano.Q


def f1(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precisao": round(p, 4), "revocacao": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


class BayesBinario(BayesGaussiano):
    """Mesma regra de decisão, com 2 classes: classes 'normal' (sem buraco) e 'buraco'."""
    def treinar(self, X, y):
        yy = np.where(y == 2, 2, 0)
        sel = X[:, self.idx]
        self.mu = np.round(sel.mean(0)).astype(np.int64); self.sd = np.maximum(np.round(sel.std(0)), 1).astype(np.int64)
        Z = self.padronizar(X).astype(np.float64); n = len(self.idx)
        for c, nome in ((0, "normal"), (2, "buraco")):
            Zc = Z[yy == c]; C = np.cov(Zc.T).reshape(n, n) + 1e-3 * 64 * 64 * np.eye(n)
            self.m[nome] = np.round(Zc.mean(0)).astype(np.int64)
            self.Ci[nome] = np.round(np.linalg.inv(C) * self.Q).astype(np.int64)
            self.ld[nome] = int(round(0.5 * np.linalg.slogdet(C)[1] * self.Q))
        self.m["trinca"], self.Ci["trinca"], self.ld["trinca"] = self.m["normal"], self.Ci["normal"], self.ld["normal"]
        self.vies["trinca"] = -(1 << 40)          # classe trinca desativada neste modelo
        return self


def melhor_vies(m, X, y):
    S = m.escores(X)
    marg = S[:, 2] - S[:, 0]
    cands = np.unique(np.percentile(marg, np.linspace(80, 99.9, 120)).astype(np.int64))
    return max(((f1(marg >= t, y == 2)["f1"], int(t)) for t in cands))


def main():
    d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
    X, y, img, teste = d["X"], d["y"], d["img"], d["teste"]
    val = (~teste) & (img % 5 == 0); tr = (~teste) & ~val
    sel, resto, atual, t0 = [], DESCRITORES[:], 0.0, time.time()
    while resto and time.time() - t0 < 140:
        cand = [(melhor_vies(BayesBinario(sel + [c]).treinar(X[tr], y[tr]), X[val], y[val])[0], c) for c in resto]
        n, c = max(cand)
        if n <= atual + 0.003:
            break
        atual = n; sel.append(c); resto.remove(c); print(f" + {c}: {n:.4f}", flush=True)
    m = BayesBinario(sel).treinar(X[tr], y[tr])
    _, t = melhor_vies(m, X[val], y[val])
    m.vies["buraco"] = -t                      # decisão: escore_buraco - escore_normal > 0
    S = m.escores(X[teste]); pred = (S[:, 2] - S[:, 0]) > 0
    rel = {"descritores": sel, "teste_buraco": f1(pred, y[teste] == 2)}
    json.dump(m.para_dict(), open(os.path.join(_caminho.RAIZ, "modelo_bayes_buraco.json"), "w"), indent=1)
    print(json.dumps(rel, ensure_ascii=False))


if __name__ == "__main__":
    main()
