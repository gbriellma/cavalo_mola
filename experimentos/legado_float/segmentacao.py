"""Segmentação de trincas e buracos (limiarização + morfologia + seleção de candidatos)."""
from __future__ import annotations

import cv2
import numpy as np

from .caracteristicas import componentes, descrever
from .classificador import FEAT_BURACO, FEAT_TRINCA, vetor
from .morfologia import elemento, preencher_buracos


def _limiar_estatistico(resp: np.ndarray, roi: np.ndarray, k: float) -> float:
    """Limiar = mediana + k * desvio robusto (MAD) da resposta dentro da pista."""
    v = resp[roi].astype(np.float32)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return float(med + k * mad)


def _contexto(d, rec, bbox, bh, sat, realcada, roi_med, H):
    x, y, w, h = bbox
    d["contraste"] = float(bh[y:y + h, x:x + w][rec].mean())
    d["saturacao"] = float(sat[y:y + h, x:x + w][rec].mean()) if sat is not None else 0.0
    d["intensidade_rel"] = float(realcada[y:y + h, x:x + w][rec].mean() - roi_med)
    d["y_rel"] = (y + h / 2) / H
    d["preenchimento_bbox"] = d["area"] / max(w * h, 1)
    d["log_area"] = float(np.log(d["area"]))
    d["bbox"] = bbox


def segmentar_trincas(realcada, roi, p, sat=None, modelo=None, candidatos=False):
    """Trincas: estruturas escuras, finas e alongadas.

    1. Black-hat (fechamento - imagem) com EE pequeno realça vales escuros estreitos.
    2. Limiarização estatística (mediana + k·MAD) dentro da ROI da pista.
    3. Fechamento leve para religar trechos.
    4. Seleção de componentes por forma (alongamento, comprimento do esqueleto) e,
       se houver modelo treinado, por regressão logística.
    """
    bh = cv2.morphologyEx(realcada, cv2.MORPH_BLACKHAT, elemento(p.trinca_bh_k))
    bh = cv2.GaussianBlur(bh, (3, 3), 0)
    t = _limiar_estatistico(bh, roi, p.trinca_k_std)
    bruta = (bh > t) & roi
    m = cv2.morphologyEx(bruta.astype(np.uint8), cv2.MORPH_CLOSE, elemento(3)) > 0
    roi_med = float(np.median(realcada[roi])) if roi.any() else 0.0
    final = np.zeros_like(m)
    regioes, cands = [], []
    for rec, bbox, _, _ in componentes(m, p.trinca_area_min):
        d = descrever(rec, com_esqueleto=True)
        if d["comprimento"] < p.trinca_comprimento_min:
            continue
        _contexto(d, rec, bbox, bh, sat, realcada, roi_med, roi.shape[0])
        d["log_comprimento"] = float(np.log(d["comprimento"]))
        d["bif_por_100px"] = 100 * d["bifurcacoes"] / d["comprimento"]
        d["angulo_norm"] = d["angulo"] / 90
        d["mascara"] = rec
        if candidatos:
            cands.append(d)
            continue
        if modelo is not None:
            ok = modelo.prob(vetor(d, FEAT_TRINCA))[0] >= modelo.limiar
        else:
            ok = d["alongamento"] >= p.trinca_alongamento_min or d["comprimento"] >= 4 * p.trinca_comprimento_min
            if sat is not None:
                ok = ok and d["saturacao"] < p.saturacao_max
        if ok:
            x, y, w, h = bbox
            final[y:y + h, x:x + w] |= rec
            regioes.append(d)
    return {"blackhat": bh, "bruta": bruta, "mascara": final, "regioes": regioes,
            "candidatos": cands, "limiar": t}


def textura_local(gray: np.ndarray, k: int = 15) -> np.ndarray:
    """Desvio padrão local (descritor estatístico de textura) via filtros de média."""
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def segmentar_buracos(realcada, roi, p, sat=None, modelo=None, candidatos=False, L=None):
    """Buracos: regiões de textura rugosa (agregado exposto) dentro da pista.

    Análise do dataset mostrou que o desvio padrão local da luminância separa
    buraco/asfalto muito melhor que a intensidade (AUC ~0.93 vs ~0.73).
    1. Mapa de textura = desvio padrão local (janela k) do canal L.
    2. Limiar estatístico (mediana + k·MAD) na pista.
    3. Abertura (remove trincas finas) + fechamento + preenchimento de buracos.
    4. Seleção por forma/textura ou pelo classificador treinado.
    """
    base = L if L is not None else realcada
    tex = textura_local(cv2.GaussianBlur(base, (3, 3), 0), p.buraco_tex_k)
    t = _limiar_estatistico(tex, roi, p.buraco_k_std)
    bruta = (tex > t) & roi
    m = cv2.morphologyEx(bruta.astype(np.uint8), cv2.MORPH_OPEN, elemento(p.buraco_abertura))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, elemento(15))
    m = preencher_buracos(m) & roi
    bh = cv2.morphologyEx(realcada, cv2.MORPH_BLACKHAT, elemento(p.buraco_bh_k))
    roi_med = float(np.median(realcada[roi])) if roi.any() else 0.0
    final = np.zeros_like(m)
    regioes, cands = [], []
    for rec, bbox, _, _ in componentes(m, p.buraco_area_min):
        d = descrever(rec, com_esqueleto=False)
        x, y, w, h = bbox
        d["textura"] = float(tex[y:y + h, x:x + w][rec].mean())
        _contexto(d, rec, bbox, bh, sat, realcada, roi_med, roi.shape[0])
        d["mascara"] = rec
        if candidatos:
            cands.append(d)
            continue
        if modelo is not None:
            ok = modelo.prob(vetor(d, FEAT_BURACO))[0] >= modelo.limiar
        else:
            ok = (d["solidez"] >= p.buraco_solidez_min and d["alongamento"] <= p.buraco_alongamento_max
                  and d["textura"] >= p.buraco_textura_min
                  and (sat is None or d["saturacao"] < p.saturacao_max))
        if ok:
            final[y:y + h, x:x + w] |= rec
            regioes.append(d)
    return {"blackhat": bh, "textura": tex, "bruta": bruta, "mascara": final, "regioes": regioes,
            "candidatos": cands, "limiar": t}
