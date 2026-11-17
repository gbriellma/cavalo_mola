"""Leitura do RTK (Rateke et al.): imagem RGB + rótulo com 13 classes (codes.txt)."""
import os

import cv2
import numpy as np

import _caminho

RTK = os.path.join(_caminho.RAIZ, "dataset", "Road Traversing Knowledge", "RTK")
CLASSES = ["background", "roadAsphalt", "roadPaved", "roadUnpaved", "roadMarking", "speedBump", "catsEye",
           "stormDrain", "manholeCover", "patchs", "waterPuddle", "pothole", "craks"]


def listar(particao="train"):
    arq = os.path.join(RTK, "trainpaths.txt" if particao == "train" else "valpaths.txt")
    return [os.path.basename(l.strip()) for l in open(arq) if l.strip()]


def carregar(nome):
    bgr = cv2.imread(os.path.join(RTK, "image", nome))
    lab = cv2.imread(os.path.join(RTK, "label", nome), cv2.IMREAD_UNCHANGED)
    if lab.ndim == 3:
        lab = lab[..., 0]
    return bgr, lab


def superficie(lab):
    """Tipo de superfície dominante da estrada na imagem."""
    c = np.bincount(lab.ravel(), minlength=13)
    return ["asfalto", "calcamento", "terra"][int(np.argmax(c[1:4]))]
