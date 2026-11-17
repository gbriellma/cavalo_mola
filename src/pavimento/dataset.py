"""Leitura do dataset (pastas com RAW.jpg + máscaras CRACK/POTHOLE/LANE)."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import cv2
import numpy as np

CLASSES = ("CRACK", "POTHOLE", "LANE")


@dataclass
class Amostra:
    nome: str
    imagem: np.ndarray            # BGR uint8
    mascaras: dict[str, np.ndarray]  # classe -> bool (H, W)


def listar_amostras(raiz: str) -> list[str]:
    """Retorna os caminhos das pastas de amostra (cada uma com 4 arquivos)."""
    return sorted(os.path.join(raiz, d) for d in os.listdir(raiz)
                  if os.path.isdir(os.path.join(raiz, d)))


def carregar(pasta: str) -> Amostra:
    nome = os.path.basename(pasta.rstrip("/\\"))
    img = cv2.imread(os.path.join(pasta, f"{nome}_RAW.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(pasta)
    masks = {}
    for c in CLASSES:
        m = cv2.imread(os.path.join(pasta, f"{nome}_{c}.png"), cv2.IMREAD_GRAYSCALE)
        masks[c] = (m > 127) if m is not None else np.zeros(img.shape[:2], bool)
    return Amostra(nome, img, masks)


def particao(nome: str, frac_teste: float = 0.3) -> str:
    """Divisão determinística treino/teste pelo hash do nome (reprodutível)."""
    h = int(hashlib.md5(nome.encode()).hexdigest(), 16) % 1000
    return "teste" if h < frac_teste * 1000 else "treino"
