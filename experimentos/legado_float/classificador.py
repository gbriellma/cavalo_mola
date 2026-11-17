"""Classificador de regiões candidatas: regressão logística implementada do zero (NumPy).

Cada componente segmentado vira um vetor de características (forma, contraste,
textura, cor, posição). O classificador decide se é dano verdadeiro ou falso
positivo (manchas, sombras, juntas, sinalização).
"""
from __future__ import annotations

import json

import numpy as np

FEAT_TRINCA = ["log_area", "alongamento", "log_comprimento", "largura_media", "bif_por_100px",
               "contraste", "solidez", "angulo_norm", "saturacao", "y_rel"]
FEAT_BURACO = ["log_area", "solidez", "alongamento", "circularidade", "textura", "contraste",
               "intensidade_rel", "saturacao", "y_rel", "preenchimento_bbox"]


def vetor(d: dict, nomes: list[str]) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in nomes], np.float64)


class RegressaoLogistica:
    def __init__(self, lr=0.1, epocas=3000, l2=1e-3):
        self.lr, self.epocas, self.l2 = lr, epocas, l2
        self.w = self.b = self.mu = self.sd = None
        self.limiar = 0.5

    @staticmethod
    def _sig(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def treinar(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        n, k = Z.shape
        self.w, self.b = np.zeros(k), 0.0
        # pesos de classe para lidar com desbalanceamento
        pos = max(y.mean(), 1e-6)
        sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-6))
        for _ in range(self.epocas):
            pr = self._sig(Z @ self.w + self.b)
            g = (pr - y) * sw
            self.w -= self.lr * (Z.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def prob(self, X):
        X = np.atleast_2d(X)
        return self._sig(((X - self.mu) / self.sd) @ self.w + self.b)

    def para_dict(self):
        return {"w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "limiar": self.limiar}

    @classmethod
    def de_dict(cls, d):
        m = cls()
        m.w, m.b = np.array(d["w"]), d["b"]
        m.mu, m.sd, m.limiar = np.array(d["mu"]), np.array(d["sd"]), d["limiar"]
        return m


def carregar_modelos(caminho: str) -> dict:
    with open(caminho, encoding="utf-8") as f:
        d = json.load(f)
    return {k: RegressaoLogistica.de_dict(v) for k, v in d.items()}


def salvar_modelos(modelos: dict, caminho: str):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({k: m.para_dict() for k, m in modelos.items()}, f, indent=2)
