"""Descritores por BLOCO (sub-região B×B) para classificação de padrões — só inteiros.

Ideia (Fernandes, Passos et al., 2023, no mesmo dataset do DNIT): dividir a imagem em blocos,
descrever a textura de cada bloco e classificar o bloco (trinca / buraco / asfalto normal).
O classificador corta os falsos positivos da textura do asfalto; a segmentação clássica
dentro dos blocos aprovados dá o contorno exato para a quantificação.

Referências por descritor (material do Prof. Lucas Cambuim):
  média, desvio, mín., máx. do bloco ........ [4_2] sl.7 (medidas estatísticas do histograma)
  sub-regiões e desvio padrão (textura) ...... [7_1S] sl.19-24
  candidatos por limiar variável ............. [7_1S] sl.96-100 (núcleo clássico)
  gradiente (1ª derivada) e coerência ........ [4_3] sl.59-62 (passa-alta de 1ª ordem); [7_1S] sl.57-68 (bordas)
  Laplaciano (2ª derivada) ................... [4_3] sl.62-65
  comprimento do esqueleto no bloco .......... [9_2] sl.49 (comprimento = nº de pixels) + Zhang-Suen (complemento)
  contexto (blocos vizinhos) ................. [2_2] sl.2-5 (vizinhança)
  LBP uniforme (8 vizinhos) .................. COMPLEMENTO: Ojala, Pietikäinen & Mäenpää (2002), usado por
                                               Fernandes et al. (2023) neste dataset
"""
from __future__ import annotations

import numpy as np

from .nucleo import abertura, afinar, estatisticas_locais, fechamento, mediana_mad, raiz_inteira
from .parametros import Parametros

B = 16  # lado do bloco na resolução de processamento

NOMES = (["media", "desvio", "desvio_local_medio", "frac_cand_trinca", "frac_escuro_forte",
          "grad_medio", "coerencia_1000", "lapl_medio"] + [f"lbp{i}" for i in range(10)] +
         ["frac_textura", "minimo", "maximo", "contraste_contexto", "desvio_contexto", "linha",
          "frac_roi", "esqueleto", "frac_cand_buraco",
          "ctx_frac_cand_trinca", "ctx_grad", "ctx_coerencia", "ctx_esqueleto", "ctx_textura"])


def _soma_blocos(a: np.ndarray, gh: int, gw: int) -> np.ndarray:
    return a[:gh * B, :gw * B].astype(np.int64).reshape(gh, B, gw, B).sum(axis=(1, 3))


def _lbp_uniforme(g: np.ndarray) -> np.ndarray:
    """LBP de 8 vizinhos (raio 1); 0..8 = padrões uniformes por nº de bits 1; 9 = não uniforme."""
    p = np.pad(g.astype(np.int32), 1, mode="edge")
    c = p[1:-1, 1:-1]
    viz = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    bits = [(v >= c).astype(np.int32) for v in viz]
    uns = sum(bits)
    trans = sum((bits[i] != bits[(i + 1) % 8]).astype(np.int32) for i in range(8))
    return np.where(trans <= 2, uns, 9).astype(np.int32)


def _contexto(f: np.ndarray) -> np.ndarray:
    """Média inteira de um descritor sobre os 3×3 blocos vizinhos (borda recortada)."""
    gh, gw = f.shape
    p = np.pad(f.astype(np.int64), 1)
    c = np.pad(np.ones_like(f, dtype=np.int64), 1)
    s = sum(p[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
    n = sum(c[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
    return s // n


def descritores(g: np.ndarray, roi: np.ndarray, p: Parametros) -> tuple[np.ndarray, dict]:
    """Retorna matriz (gh, gw, n_desc) de inteiros e mapas auxiliares do núcleo clássico."""
    h, w = g.shape
    gh, gw = h // B, w // B
    n = B * B
    gi = g.astype(np.int32)
    media, var = estatisticas_locais(g, p.raio)
    desvio = raiz_inteira(var)
    # candidatos de trinca (limiar variável) em dois níveis
    d16 = 16 * (media - gi) - p.trinca_b16
    d2 = d16.astype(np.int64) ** 2
    cand = (d16 > 0) & (d2 > (p.trinca_a16 ** 2) * var.astype(np.int64)) & roi
    forte = (d16 > 0) & (d2 > (4 * p.trinca_a16 ** 2) * var.astype(np.int64)) & roi
    if p.trinca_fecho > 1:
        cand = fechamento(cand, p.trinca_fecho) & roi
    esq = afinar(cand)
    # candidatos de buraco (textura)
    med, mad = mediana_mad(desvio[roi].astype(np.int64))
    tex = (desvio > med + (p.buraco_k4 * mad) // 4) & roi
    cand_b = abertura(tex, p.buraco_abertura) & roi
    # gradiente (diferenças centrais, inteiro) e tensor de estrutura
    pg = np.pad(gi, 1, mode="edge")
    gx = pg[1:-1, 2:] - pg[1:-1, :-2]
    gy = pg[2:, 1:-1] - pg[:-2, 1:-1]
    lap = np.abs(4 * gi - pg[:-2, 1:-1] - pg[2:, 1:-1] - pg[1:-1, :-2] - pg[1:-1, 2:])
    sxx = _soma_blocos(gx * gx, gh, gw); syy = _soma_blocos(gy * gy, gh, gw); sxy = _soma_blocos(gx * gy, gh, gw)
    coer = (1000 * ((sxx - syy) ** 2 + 4 * sxy * sxy)) // ((sxx + syy) ** 2 + 1)
    lbp = _lbp_uniforme(g)
    s1 = _soma_blocos(gi, gh, gw); s2 = _soma_blocos(gi * gi, gh, gw)
    mb = s1 // n
    db = raiz_inteira((n * s2 - s1 * s1) // (n * n))
    blk = gi[:gh * B, :gw * B].reshape(gh, B, gw, B)
    f = {
        "media": mb, "desvio": db,
        "desvio_local_medio": _soma_blocos(desvio, gh, gw) // n,
        "frac_cand_trinca": (1000 * _soma_blocos(cand, gh, gw)) // n,
        "frac_escuro_forte": (1000 * _soma_blocos(forte, gh, gw)) // n,
        "grad_medio": _soma_blocos(np.abs(gx) + np.abs(gy), gh, gw) // n,
        "coerencia_1000": coer,
        "lapl_medio": _soma_blocos(lap, gh, gw) // n,
        "frac_textura": (1000 * _soma_blocos(tex, gh, gw)) // n,
        "minimo": blk.min(axis=(1, 3)), "maximo": blk.max(axis=(1, 3)),
        "linha": np.repeat(np.arange(gh)[:, None], gw, axis=1),
        "frac_roi": (1000 * _soma_blocos(roi, gh, gw)) // n,
        "esqueleto": _soma_blocos(esq, gh, gw),
        "frac_cand_buraco": (1000 * _soma_blocos(cand_b, gh, gw)) // n,
    }
    for i in range(10):
        f[f"lbp{i}"] = _soma_blocos(lbp == i, gh, gw)
    f["contraste_contexto"] = mb - _contexto(mb)
    f["desvio_contexto"] = db - _contexto(db)
    f["ctx_frac_cand_trinca"] = _contexto(f["frac_cand_trinca"])
    f["ctx_grad"] = _contexto(f["grad_medio"])
    f["ctx_coerencia"] = _contexto(f["coerencia_1000"])
    f["ctx_esqueleto"] = _contexto(f["esqueleto"])
    f["ctx_textura"] = _contexto(f["frac_textura"])
    X = np.stack([np.asarray(f[k], dtype=np.int64) for k in NOMES], axis=-1)
    return X, {"cand_trinca": cand, "esqueleto": esq, "textura": tex, "cand_buraco": cand_b,
               "media": media, "var": var, "desvio": desvio}


def rotulos_blocos(gt: np.ndarray, gb: np.ndarray, roi: np.ndarray, min_trinca: int = 6, min_buraco: int = 64):
    """Rótulo do bloco: 2 = buraco (>=25% do bloco), 1 = trinca (>= min_trinca px), 0 = normal.
    Blocos com menos de 50% de pista são ignorados (-1)."""
    gh, gw = gt.shape[0] // B, gt.shape[1] // B
    t = _soma_blocos(gt, gh, gw); b = _soma_blocos(gb, gh, gw); r = _soma_blocos(roi, gh, gw)
    y = np.zeros((gh, gw), np.int64)
    y[t >= min_trinca] = 1
    y[b >= min_buraco] = 2
    y[2 * r < B * B] = -1
    return y
