"""Gradient boosting de árvores de decisão implementado do zero (NumPy) — classificação de padrões.

Por que árvores com boosting:
  * classificam vetores de descritores heterogêneos (estatística, gradiente, textura, contexto)
    sem normalização;
  * inferência = comparações inteiras  x[f] <= limiar  + soma de folhas inteiras → determinística
    e trivial de portar para C na ESP32 (sem float);
  * na literatura de pavimentos, árvores (CrackForest, LightGBM) estão entre os melhores
    classificadores sobre descritores manuais.
Referências: Friedman (2001) "Greedy function approximation: a gradient boosting machine";
Chen & Guestrin (2016) XGBoost (ganho de segunda ordem); Ke et al. (2017) LightGBM (histogramas).

Treino determinístico: sem amostragem aleatória; desempates pela ordem (descritor, limiar).
"""
from __future__ import annotations

import json

import numpy as np

ESCALA = 1024  # folhas quantizadas: valor_inteiro = round(folha * ESCALA)


def limiares_quantis(X: np.ndarray, nbins: int = 32) -> list[np.ndarray]:
    """Limiares inteiros candidatos por descritor (quantis únicos dos dados de treino)."""
    lim = []
    q = np.linspace(0, 100, nbins + 1)[1:-1]
    for j in range(X.shape[1]):
        t = np.unique(np.percentile(X[:, j], q, method="lower").astype(np.int64))
        lim.append(t)
    return lim


def binarizar(X: np.ndarray, lim: list[np.ndarray]) -> np.ndarray:
    """bin = nº de limiares < x  (x <= lim[b] ⇔ bin <= b)."""
    return np.stack([np.searchsorted(lim[j], X[:, j], side="left") for j in range(X.shape[1])], axis=1).astype(np.int32)


class Boosting:
    def __init__(self, n_arvores=200, profundidade=3, taxa=0.1, lam=1.0, min_hess=5.0):
        self.n_arvores, self.prof, self.taxa, self.lam, self.min_hess = n_arvores, profundidade, taxa, lam, min_hess
        self.arvores = []   # cada árvore: lista de nós {f, t, esq, dir} ou {folha}
        self.base = 0
        self.lim = None

    def _histograma(self, Xb, idx, grad, hess, nb):
        F = Xb.shape[1]
        cod = (Xb[idx] + (np.arange(F) * nb)[None, :]).ravel()
        G = np.bincount(cod, weights=np.repeat(grad[idx], F), minlength=F * nb).reshape(F, nb)
        H = np.bincount(cod, weights=np.repeat(hess[idx], F), minlength=F * nb).reshape(F, nb)
        return G, H

    def _construir(self, Xb, idx, grad, hess, prof, nos, nb, hist=None):
        g, h = grad[idx].sum(), hess[idx].sum()
        folha = -g / (h + self.lam)
        if prof == self.prof or len(idx) < 2:
            nos.append({"folha": int(round(self.taxa * folha * ESCALA))})
            return len(nos) - 1
        G, H = hist if hist is not None else self._histograma(Xb, idx, grad, hess, nb)
        GL, HL = np.cumsum(G, axis=1), np.cumsum(H, axis=1)
        GR, HR = g - GL, h - HL
        ganho = GL ** 2 / (HL + self.lam) + GR ** 2 / (HR + self.lam) - g ** 2 / (h + self.lam)
        ganho[(HL < self.min_hess) | (HR < self.min_hess)] = -np.inf
        ganho[self._invalido] = -np.inf                  # só limiares existentes
        k = int(np.argmax(ganho))
        if not np.isfinite(ganho.flat[k]) or ganho.flat[k] <= 1e-9:
            nos.append({"folha": int(round(self.taxa * folha * ESCALA))})
            return len(nos) - 1
        f, b = divmod(k, nb)
        no = {"f": int(f), "t": int(self.lim[f][b])}
        nos.append(no)
        pos = len(nos) - 1
        m = Xb[idx, f] <= b
        ie, idir = idx[m], idx[~m]
        hist_e = hist_d = None
        if prof + 1 < self.prof:   # histograma só do filho menor; o outro por subtração
            if len(ie) <= len(idir):
                hist_e = self._histograma(Xb, ie, grad, hess, nb); hist_d = (G - hist_e[0], H - hist_e[1])
            else:
                hist_d = self._histograma(Xb, idir, grad, hess, nb); hist_e = (G - hist_d[0], H - hist_d[1])
        no["esq"] = self._construir(Xb, ie, grad, hess, prof + 1, nos, nb, hist_e)
        no["dir"] = self._construir(Xb, idir, grad, hess, prof + 1, nos, nb, hist_d)
        return pos

    def treinar(self, X, y, X_val=None, y_val=None, paciencia=30, log=None, tempo_max=None, estado=None):
        """Treina (ou CONTINUA, se 'estado' vier de uma chamada anterior) até n_arvores,
        parada antecipada pela validação ou 'tempo_max' segundos. Retorna o estado para retomar."""
        import time
        t0 = time.time()
        if estado is None:
            self.lim = limiares_quantis(X)
            p0 = min(max(y.mean(), 1e-4), 1 - 1e-4)
            self.base = int(round(np.log(p0 / (1 - p0)) * ESCALA))
            estado = {"F": np.full(len(y), self.base, np.int64),
                      "Fv": np.full(len(y_val), self.base, np.int64) if X_val is not None else None,
                      "melhor": np.inf, "melhor_n": 0, "fim": False}
        nb = max(len(l) for l in self.lim) + 1
        self._invalido = np.zeros((X.shape[1], nb), bool)
        for j in range(X.shape[1]):
            self._invalido[j, len(self.lim[j]):] = True
        Xb = binarizar(X, self.lim)
        F, Fv = estado["F"], estado["Fv"]
        idx_all = np.arange(len(y))
        while len(self.arvores) < self.n_arvores:
            if tempo_max and time.time() - t0 > tempo_max:
                return estado
            pr = 1 / (1 + np.exp(-F / ESCALA))
            grad, hess = pr - y, np.maximum(pr * (1 - pr), 1e-6)
            nos = []
            self._construir(Xb, idx_all, grad, hess, 0, nos, nb)
            self.arvores.append(nos)
            F += self._pontuar_arvore(nos, X)
            t = len(self.arvores)
            if X_val is not None:
                Fv += self._pontuar_arvore(nos, X_val)
                pv = np.clip(1 / (1 + np.exp(-Fv / ESCALA)), 1e-7, 1 - 1e-7)
                perda = -np.mean(y_val * np.log(pv) + (1 - y_val) * np.log(1 - pv))
                if perda < estado["melhor"] - 1e-6:
                    estado["melhor"], estado["melhor_n"] = perda, t
                elif t - estado["melhor_n"] >= paciencia:
                    break
                if log and t % 25 == 0:
                    log(f"  árvore {t}: perda val {perda:.4f}")
        if X_val is not None and estado["melhor_n"]:
            self.arvores = self.arvores[:estado["melhor_n"]]
        estado["fim"] = True
        return estado

    @staticmethod
    def _pontuar_arvore(nos, X):
        atual = np.zeros(len(X), np.int64)
        folha = np.array([("folha" in n) for n in nos])
        val = np.array([n.get("folha", 0) for n in nos], np.int64)
        f = np.array([n.get("f", 0) for n in nos], np.int64)
        t = np.array([n.get("t", 0) for n in nos], np.int64)
        e = np.array([n.get("esq", 0) for n in nos], np.int64)
        d = np.array([n.get("dir", 0) for n in nos], np.int64)
        lin = np.arange(len(X))
        for _ in range(64):
            ativo = ~folha[atual]
            if not ativo.any():
                break
            a = atual[ativo]
            vai_esq = X[lin[ativo], f[a]] <= t[a]
            atual[ativo] = np.where(vai_esq, e[a], d[a])
        return val[atual]

    def pontuar(self, X) -> np.ndarray:
        """Escore inteiro (log-odds × ESCALA)."""
        F = np.full(len(X), self.base, np.int64)
        for nos in self.arvores:
            F += self._pontuar_arvore(nos, X)
        return F

    def para_dict(self):
        return {"base": self.base, "escala": ESCALA, "arvores": self.arvores}

    @classmethod
    def de_dict(cls, d):
        m = cls(); m.base = d["base"]; m.arvores = d["arvores"]; return m


def salvar(modelos: dict, caminho: str):
    json.dump({k: (m.para_dict() if hasattr(m, "para_dict") else m) for k, m in modelos.items()},
              open(caminho, "w"), indent=1)


def carregar(caminho: str) -> dict:
    d = json.load(open(caminho))
    return {k: (Boosting.de_dict(v) if isinstance(v, dict) and "arvores" in v else v) for k, v in d.items()}
