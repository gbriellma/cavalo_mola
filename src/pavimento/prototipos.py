"""Classificação de padrões CLÁSSICA para os blocos: protótipos por K-means + distância mínima.

* Descritores do bloco: só os que vêm do material da disciplina (ver blocos.py), padronizados
  em inteiros: z = (x - média)·64 // desvio.
* Protótipos: K-means (algoritmo de Lloyd) por classe — o mesmo agrupamento usado pelo professor
  em codigos_7_cores/Cores.ipynb (segmentação por cor em Lab) —, implementado do zero.
* Classificador de distância mínima: o bloco recebe a classe do protótipo mais próximo pela
  distância Euclidiana [2_2 sl.20-22] (comparada ao quadrado, sem raiz). Um peso inteiro por
  classe (ajustado na validação) desloca a fronteira entre precisão e revocação.
  Referência: Gonzalez & Woods, 3ª ed., seção 12.2.1 ("classificador de distância mínima").
* Coerência espacial: filtro de MEDIANA 3×3 na grade de decisões [4_3 sl.54-58] (opcional).
Tudo determinístico (inicialização determinística do K-means) e em inteiros na inferência.
"""
from __future__ import annotations

import json

import numpy as np

from .blocos import NOMES

DESCRITORES = ["media", "desvio", "desvio_local_medio", "frac_cand_trinca", "frac_escuro_forte", "grad_medio",
               "lapl_medio", "frac_textura", "minimo", "maximo", "contraste_contexto", "desvio_contexto", "linha",
               "esqueleto", "frac_cand_buraco", "ctx_frac_cand_trinca", "ctx_grad", "ctx_esqueleto", "ctx_textura"]
IDX = [NOMES.index(n) for n in DESCRITORES]
CLASSES = ("normal", "trinca", "buraco")


def kmeans(Z: np.ndarray, k: int, iters: int = 50) -> np.ndarray:
    """K-means de Lloyd com inicialização DETERMINÍSTICA por ponto mais distante
    (começa pela média; depois sempre o ponto mais distante dos centros já escolhidos)."""
    Z = Z.astype(np.float64)
    c = [Z.mean(0)]
    d = ((Z - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        c.append(Z[int(np.argmax(d))]); d = np.minimum(d, ((Z - c[-1]) ** 2).sum(1))
    C = np.array(c)
    for _ in range(iters):
        dist = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = dist.argmin(1)
        novo = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(novo, C):
            break
        C = novo
    return C


class DistanciaMinima:
    def __init__(self):
        self.mu = self.sd = None
        self.prot = {}          # classe -> matriz de protótipos (int)
        self.peso = {"normal": 100, "trinca": 100, "buraco": 100}

    def padronizar(self, X: np.ndarray) -> np.ndarray:
        Z = ((X[:, IDX].astype(np.int64) - self.mu) * 64) // self.sd
        return np.clip(Z, -512, 512)

    def treinar(self, X, y, k=(8, 6, 4), max_por_classe=40000):
        sel = X[:, IDX]
        self.mu = np.round(sel.mean(0)).astype(np.int64)
        self.sd = np.maximum(np.round(sel.std(0)), 1).astype(np.int64)
        Z = self.padronizar(X)
        for c, kc in zip(range(3), k):
            Zc = Z[y == c]
            if len(Zc) > max_por_classe:                     # subamostragem determinística
                Zc = Zc[:: len(Zc) // max_por_classe + 1]
            self.prot[CLASSES[c]] = np.round(kmeans(Zc, kc)).astype(np.int64)
        return self

    def distancias(self, X) -> np.ndarray:
        """Distância (ao quadrado) ao protótipo mais próximo de cada classe — inteiros."""
        Z = self.padronizar(X)
        return np.stack([((Z[:, None, :] - self.prot[c][None]) ** 2).sum(2).min(1) for c in CLASSES], axis=1)

    def classificar(self, X) -> np.ndarray:
        D = self.distancias(X) * np.array([self.peso[c] for c in CLASSES])[None, :]
        return D.argmin(1)

    def para_dict(self):
        return {"descritores": DESCRITORES, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "prot": {c: v.tolist() for c, v in self.prot.items()}, "peso": self.peso}

    @classmethod
    def de_dict(cls, d):
        m = cls(); m.mu = np.array(d["mu"]); m.sd = np.array(d["sd"])
        m.prot = {c: np.array(v) for c, v in d["prot"].items()}; m.peso = d["peso"]
        return m


def filtro_mediana_grade(dec: np.ndarray, valido: np.ndarray) -> np.ndarray:
    """Mediana 3×3 de uma grade binária = maioria entre os vizinhos válidos [4_3 sl.54-58]."""
    gh, gw = dec.shape
    p = np.pad(dec.astype(np.int32), 1); v = np.pad(valido.astype(np.int32), 1)
    s = sum(p[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
    n = sum(v[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
    return (2 * s > n) & valido


class BayesGaussiano:
    """Classificador de Bayes para classes gaussianas (Gonzalez & Woods, 3ª ed., seção 12.2.2):
        d_c(z) = -½ (z-m_c)ᵀ C_c⁻¹ (z-m_c) - ½ ln|C_c| + viés_c ,   classe = argmax_c d_c
    O viés (ajustado na validação) faz o papel do ln P(c) e controla precisão × revocação.
    Inferência 100% INTEIRA: z = (x-μ)·64//σ;  C⁻¹ quantizada em 2^12/64²;  termos em 2^12.
    """
    Q = 1 << 12

    def __init__(self, descritores=None):
        self.descritores = descritores or DESCRITORES
        self.idx = [NOMES.index(n) for n in self.descritores]
        self.mu = self.sd = None
        self.m, self.Ci, self.ld = {}, {}, {}
        self.vies = {"normal": 0, "trinca": 0, "buraco": 0}   # em unidades de 2^12

    def padronizar(self, X):
        return np.clip(((X[:, self.idx].astype(np.int64) - self.mu) * 64) // self.sd, -512, 512)

    def treinar(self, X, y):
        sel = X[:, self.idx]
        self.mu = np.round(sel.mean(0)).astype(np.int64)
        self.sd = np.maximum(np.round(sel.std(0)), 1).astype(np.int64)
        Z = self.padronizar(X).astype(np.float64)
        n = len(self.idx)
        for c, nome in enumerate(CLASSES):
            Zc = Z[y == c]
            C = np.cov(Zc.T).reshape(n, n) + 1e-3 * 64 * 64 * np.eye(n)
            self.m[nome] = np.round(Zc.mean(0)).astype(np.int64)
            self.Ci[nome] = np.round(np.linalg.inv(C) * self.Q).astype(np.int64)
            self.ld[nome] = int(round(0.5 * np.linalg.slogdet(C)[1] * self.Q))
        return self

    def escores(self, X) -> np.ndarray:
        Z = self.padronizar(X)
        S = []
        for nome in CLASSES:
            D = Z - self.m[nome]
            quad = np.einsum("ij,jk,ik->i", D, self.Ci[nome], D)          # ≈ (z-m)ᵀC⁻¹(z-m)·2^12
            S.append(-(quad // 2) - self.ld[nome] + self.vies[nome])
        return np.stack(S, axis=1)

    def classificar(self, X) -> np.ndarray:
        return self.escores(X).argmax(1)

    def para_dict(self):
        return {"tipo": "bayes", "descritores": self.descritores, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "m": {c: v.tolist() for c, v in self.m.items()}, "Ci": {c: v.tolist() for c, v in self.Ci.items()},
                "ld": self.ld, "vies": self.vies}

    @classmethod
    def de_dict(cls, d):
        b = cls(d["descritores"]); b.mu = np.array(d["mu"]); b.sd = np.array(d["sd"])
        b.m = {c: np.array(v) for c, v in d["m"].items()}; b.Ci = {c: np.array(v) for c, v in d["Ci"].items()}
        b.ld = d["ld"]; b.vies = d["vies"]
        return b
