"""Pré-processamento: fundamentos, realce, filtragem espacial e no domínio da frequência."""
from __future__ import annotations

import cv2
import numpy as np


def equalizar_histograma(gray: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    """Equalização de histograma implementada do zero (CDF), opcionalmente só na ROI."""
    vals = gray[mascara] if mascara is not None else gray.ravel()
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.clip(np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255), 0, 255).astype(np.uint8)
    return lut[gray]


def filtro_homomorfico(gray: np.ndarray, d0: float = 20.0, gamma_l: float = 0.5,
                       gamma_h: float = 1.4, c: float = 1.0) -> np.ndarray:
    """Filtro homomórfico (FFT) para atenuar variações de iluminação/sombras.

    I = L * R  ->  log I = log L + log R. A iluminação L é de baixa frequência e a
    refletância R (textura, trincas) de alta frequência; um passa-alta gaussiano
    com ganhos gamma_l < 1 < gamma_h comprime L e realça R.
    """
    f = np.log1p(gray.astype(np.float32))
    rows, cols = f.shape
    P, Q = cv2.getOptimalDFTSize(2 * rows), cv2.getOptimalDFTSize(2 * cols)
    pad = np.zeros((P, Q), np.float32)
    pad[:rows, :cols] = f
    F = np.fft.fftshift(np.fft.fft2(pad))
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    V, U = np.meshgrid(v, u)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (U ** 2 + V ** 2) / (d0 ** 2))) + gamma_l
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))[:rows, :cols]
    out = np.expm1(g)
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def espectro(gray: np.ndarray) -> np.ndarray:
    """Magnitude do espectro (log) para visualização."""
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log1p(np.abs(F))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocessar(img_bgr: np.ndarray, roi: np.ndarray, p) -> dict:
    """Pipeline de pré-processamento. Retorna as imagens intermediárias."""
    # Espaço de cor: o canal L do Lab separa luminância de cromaticidade
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    homo = filtro_homomorfico(L, d0=p.homo_d0, gamma_l=p.homo_gl, gamma_h=p.homo_gh)
    # Suavização que preserva bordas (bilateral) para reduzir textura do asfalto
    suave = cv2.bilateralFilter(homo, p.bilat_d, p.bilat_sc, p.bilat_ss)
    # Realce de contraste local (CLAHE) restrito à pista
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(suave)
    clahe[~roi] = int(np.median(clahe[roi])) if roi.any() else 0
    # Saturação (HSV) ajuda a separar sinalização/vegetação do asfalto cinza
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    return {"L": L, "homomorfico": homo, "suavizada": suave, "realcada": clahe, "saturacao": sat}
