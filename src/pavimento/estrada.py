"""Detecção automática da ÁREA DA ESTRADA por crescimento de regiões (só inteiros).

Tudo o que estiver fora da estrada é descartado antes da análise de danos.

Referências (material do Prof. Lucas Cambuim):
  [7_2]  "7_2 - segmentacao.pdf" sl.4-14: crescimento de regiões — sementes, critério de
         similaridade, conectividade (sl.13) e regra de parada (sl.14)
  [7_1C] "7_1 - processamento de imagens em cores.pdf" sl.60-73: modelo HSI;
         conversão RGB->HSI sl.70-71 (I = (R+G+B)/3, S = 1 - 3·min(R,G,B)/(R+G+B))
  [7_1S] "7_1 - segmentacao.pdf" sl.19-24: textura pelo desvio padrão em sub-regiões
  [6_1]  fechamento sl.40-42, preenchimento de buracos sl.61-66
Para velocidade (e para a ESP32), o crescimento é feito sobre BLOCOS b×b (sub-regiões,
como no slide 7_1S sl.23), cada bloco descrito por média de I, média de S e desvio de I.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nucleo import fechamento, preencher_buracos, raiz_inteira


@dataclass(frozen=True)
class ParamEstrada:
    bloco: int = 8          # lado do bloco (px)
    tol_i: int = 28         # |I_bloco - I_semente| máximo
    tol_s: int = 30         # |S_bloco - S_semente| máximo (S em 0..255)
    tol_d: int = 10         # desvio do bloco <= desvio_semente + tol_d
    semente_altura: int = 2 # nº de linhas de blocos na base da imagem usadas como semente
    semente_larg_pct: int = 50  # % central da largura usada como semente
    fecho: int = 3          # fechamento no grid de blocos


def saturacao_hsi(rgb: np.ndarray) -> np.ndarray:
    """S = 1 - 3·min/(R+G+B) [7_1C sl.70-71], em 0..255 inteiro: 255 - 765·min // soma."""
    a = rgb.astype(np.int32)
    soma = a[..., 0] + a[..., 1] + a[..., 2]
    mn = a.min(axis=2)
    s = 255 - (765 * mn) // np.maximum(soma, 1)
    s[soma == 0] = 0
    return s.astype(np.uint8)


def _blocos(img: np.ndarray, b: int):
    h, w = img.shape[0] // b * b, img.shape[1] // b * b
    a = img[:h, :w].astype(np.int64).reshape(h // b, b, w // b, b)
    return a.sum(axis=(1, 3)), (a * a).sum(axis=(1, 3))


def descritores_blocos(i_img: np.ndarray, s_img: np.ndarray, b: int):
    """Média de I, média de S e desvio de I por bloco (divisões inteiras)."""
    n = b * b
    si, si2 = _blocos(i_img, b)
    ss, _ = _blocos(s_img, b)
    mi = si // n
    ms = ss // n
    var = (n * si2 - si * si) // (n * n)
    return mi.astype(np.int32), ms.astype(np.int32), raiz_inteira(var)


def _mediana_inferior(v: np.ndarray) -> int:
    v = np.sort(v.ravel())
    return int(v[(v.size - 1) // 2]) if v.size else 0


def crescer_estrada(i_img: np.ndarray, s_img: np.ndarray, p: ParamEstrada = ParamEstrada()):
    """Retorna (máscara da estrada em pixels, dados intermediários)."""
    b = p.bloco
    mi, ms, md = descritores_blocos(i_img, s_img, b)
    gh, gw = mi.shape
    # Sementes: base central da imagem (a câmera olha a estrada à frente) [7_2 sl.5, 10-11]
    c0 = gw * (100 - p.semente_larg_pct) // 200
    c1 = gw - c0
    r0 = max(0, gh - p.semente_altura)
    ref_i = _mediana_inferior(mi[r0:, c0:c1])
    ref_s = _mediana_inferior(ms[r0:, c0:c1])
    ref_d = _mediana_inferior(md[r0:, c0:c1])
    # Critério de similaridade com a semente [7_2 sl.6-8, 12]
    ok = (np.abs(mi - ref_i) <= p.tol_i) & (np.abs(ms - ref_s) <= p.tol_s) & (md <= ref_d + p.tol_d)
    # Crescimento com conectividade-4 a partir das sementes que satisfazem o critério [7_2 sl.13];
    # para quando nenhum bloco vizinho satisfaz o critério [7_2 sl.14]
    reg = np.zeros_like(ok)
    reg[r0:, c0:c1] = ok[r0:, c0:c1]
    while True:
        viz = reg.copy()
        viz[1:] |= reg[:-1]; viz[:-1] |= reg[1:]
        viz[:, 1:] |= reg[:, :-1]; viz[:, :-1] |= reg[:, 1:]
        nova = viz & ok
        if np.array_equal(nova, reg):
            break
        reg = nova
    # Pós-processamento morfológico no grid de blocos [6_1]
    if p.fecho > 1:
        reg = fechamento(reg, p.fecho)
    reg = preencher_buracos(reg)
    # De volta à resolução de pixels (cada bloco vira b×b); sobras da borda ficam fora
    mask = np.zeros(i_img.shape, bool)
    mask[:gh * b, :gw * b] = np.repeat(np.repeat(reg, b, axis=0), b, axis=1)
    return mask, {"blocos_ok": ok, "blocos_estrada": reg, "ref": (ref_i, ref_s, ref_d)}
