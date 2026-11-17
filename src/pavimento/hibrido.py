"""Pipeline HÍBRIDO: classificador de blocos (onde há dano) + segmentação clássica (contorno exato).

1. Descritores por bloco 16×16 (blocos.py) -> boosting estágio 1 -> contexto -> estágio 2.
2. Blocos de trinca aprovados (+1 bloco de margem) liberam os candidatos de trinca do limiar
   variável [7_1S sl.96-100]; componentes/esqueleto como no núcleo.
3. Blocos de buraco aprovados: candidatos de textura dentro da região, fechamento e
   preenchimento [6_1 sl.40-42, 61-66].
4. Quantificação idêntica à do núcleo (comprimento pelo esqueleto, área, índice, severidade).
"""
from __future__ import annotations

import numpy as np

from .blocos import B, NOMES, descritores
from .nucleo import (TIPOS, afinar, contar_vizinhos8, dilatar, fechamento, preencher_buracos, quantificar,
                     rotular)
from dataclasses import dataclass

from .parametros import Parametros


def _contexto_grade(esc: np.ndarray, valido: np.ndarray):
    """Escore próprio, média e máximo dos 8 vizinhos válidos (mesma regra do treino)."""
    gh, gw = esc.shape
    p = np.zeros((gh + 2, gw + 2), np.int64); t = np.zeros((gh + 2, gw + 2), bool)
    p[1:-1, 1:-1] = esc; t[1:-1, 1:-1] = valido
    soma = np.zeros_like(esc); cnt = np.zeros_like(esc); mx = esc.copy(); prim = np.ones(esc.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            v = p[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw]; ok = t[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw]
            soma += np.where(ok, v, 0); cnt += ok
            mx = np.where(ok & (prim | (v > mx)), v, mx); prim &= ~ok
    return np.stack([esc, soma // np.maximum(cnt, 1), mx], axis=-1)


def carregar_classificador(raiz: str, tipo: str = "bayes"):
    """'bayes' (padrão, CLÁSSICO: Bayes gaussiano, Gonzalez & Woods 12.2.2) ou
    'boosting' (experimento de comparação, experimentos/)."""
    import json, os
    if tipo == "bayes":
        from .prototipos import BayesGaussiano
        return BayesGaussiano.de_dict(json.load(open(os.path.join(raiz, "modelo_bayes.json"))))
    from .boosting import carregar
    return carregar(os.path.join(raiz, "modelos_blocos.json"))


def classificar_blocos(X: np.ndarray, valido: np.ndarray, modelos, mediana: bool = False):
    gh, gw, F = X.shape
    if not isinstance(modelos, dict):                      # Bayes gaussiano (clássico)
        from .prototipos import filtro_mediana_grade
        S = modelos.escores(X.reshape(-1, F)).reshape(gh, gw, 3)
        out = {}
        for k, nome in ((1, "trinca"), (2, "buraco")):
            outros = np.max(np.delete(S, k, axis=2), axis=2)
            esc = S[..., k] - outros                        # > 0  <=>  classe vencedora
            dec = (esc > 0) & valido
            if mediana:
                dec = filtro_mediana_grade(dec, valido)
            out[nome], out[nome + "_escore"], out[nome + "_limiar"] = dec, esc, 0
        return out
    Xf = X.reshape(-1, F)
    ctx = []
    for nome in ("trinca", "buraco"):
        esc = modelos[nome + "_e1"].pontuar(Xf).reshape(gh, gw)
        ctx.append(_contexto_grade(esc, valido))
    X2 = np.concatenate([X] + ctx, axis=-1).reshape(gh * gw, -1)
    out = {}
    for nome in ("trinca", "buraco"):
        s = modelos[nome + "_e2"].pontuar(X2).reshape(gh, gw)
        out[nome] = (s >= modelos[nome + "_limiar"]) & valido
        out[nome + "_escore"] = s
        out[nome + "_limiar"] = modelos[nome + "_limiar"]
    return out


def _expandir(blocos: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    gh, gw = blocos.shape
    m[:gh * B, :gw * B] = np.repeat(np.repeat(blocos, B, axis=0), B, axis=1)
    return m


@dataclass(frozen=True)
class ParamPixel:
    """Parâmetros do estágio de PIXELS (separados dos descritores, que alimentam o classificador)."""
    a16: int = 16            # limiar variável dentro dos blocos aprovados: I < m - (a·s + b)
    b16: int = 32
    fecho: int = 3
    area_min: int = 25
    comp_min: int = 14
    larg_max16: int = 48
    margem: int = 3          # dilatação (em blocos) da zona de trinca
    buraco_fecho: int = 5
    buraco_area_min: int = 40
    buraco_modo: int = 1     # 0 = textura refinada na zona, 1 = zona inteira dos blocos, 2 = escore interpolado
    buraco_desloc: int = 0   # deslocamento do limiar do escore interpolado (modo 2)
    trinca_desloc: int = -100000  # componente de trinca só é aceito se o escore médio interpolado
                                  # dos blocos sobre ele for >= limiar_trinca + desloc (-100000 = desligado)
    mediana: int = 0         # 1 = filtro de mediana 3×3 nas decisões dos blocos [4_3 sl.54-58]
    sev_media: int = 0       # limiares de severidade do índice PREVISTO (0 = usar os de Parametros);
    sev_alta: int = 0        # calibrados na validação por scripts/calibrar_severidade_hibrido.py


def interpolar_bilinear(grade: np.ndarray, shape) -> np.ndarray:
    """Interpolação bilinear INTEIRA do escore dos blocos para os pixels [2_1 sl.71-88].
    Centros dos blocos em (B·i + B/2); pesos inteiros em 1/B (separável: x depois y);
    fora dos centros extremos o valor é replicado (bordas)."""
    gh, gw = grade.shape
    h, w = shape
    def eixo(n_pix, n_blk):
        pos2 = 2 * np.arange(n_pix) + 1 - B           # 2·(posição - centro do bloco 0)
        i0 = np.clip(pos2 // (2 * B), 0, n_blk - 1)
        resto = pos2 - i0 * 2 * B                       # 0..2B-1 dentro do intervalo
        i1 = np.clip(i0 + 1, 0, n_blk - 1)
        wgt = np.clip(resto, 0, 2 * B)                  # peso do vizinho seguinte, em 1/(2B)
        wgt = np.where(pos2 < 0, 0, wgt); wgt = np.where(i0 == n_blk - 1, 0, wgt)
        return i0, i1, wgt
    y0, y1, wy = eixo(h, gh); x0, x1, wx = eixo(w, gw)
    g64 = grade.astype(np.int64)
    linha = g64[:, x0] * (2 * B - wx)[None, :] + g64[:, x1] * wx[None, :]          # escala 2B
    out = linha[y0, :] * (2 * B - wy)[:, None] + linha[y1, :] * wy[:, None]       # escala 4B²
    return out // (4 * B * B)


def analisar_hibrido(g: np.ndarray, roi: np.ndarray, p: Parametros, modelos: dict,
                     pp: "ParamPixel" = None) -> dict:
    pp = pp or ParamPixel()
    roi = roi.astype(bool)
    X, aux = descritores(g, roi, p)
    gh, gw = X.shape[:2]
    frac_roi = X[..., NOMES.index("frac_roi")]  # x1000
    valido = frac_roi >= 500
    cls = classificar_blocos(X, valido, modelos, pp.mediana == 1)
    # ---- buracos
    zona_b = _expandir(cls["buraco"], g.shape) & roi
    if pp.buraco_modo == 2:
        esc = np.where(valido, cls["buraco_escore"], -(1 << 20))
        bur = (interpolar_bilinear(esc, g.shape) >= cls["buraco_limiar"] + pp.buraco_desloc) & roi
    elif pp.buraco_modo == 1:
        bur = zona_b.copy()
    else:
        bur = (aux["textura"] | aux["cand_buraco"]) & zona_b
        bur = fechamento(bur, pp.buraco_fecho)
        bur = preencher_buracos(bur) & zona_b
    lab_b, nb = rotular(bur)
    reg_b = []
    if nb:
        area = np.bincount(lab_b.ravel(), minlength=nb + 1)
        for i in range(1, nb + 1):
            if area[i] >= pp.buraco_area_min:
                reg_b.append({"area": int(area[i])})
            else:
                bur[lab_b == i] = False
    # ---- trincas
    zona_t = _expandir(dilatar(cls["trinca"], 2 * pp.margem + 1), g.shape) & roi & ~bur
    # candidatos de pixel com os limiares do estágio de pixels (média/variância já calculadas)
    d16 = 16 * (aux["media"] - g.astype(np.int32)) - pp.b16
    cand = (d16 > 0) & (d16.astype(np.int64) ** 2 > (pp.a16 ** 2) * aux["var"].astype(np.int64)) & zona_t
    if pp.fecho > 1:
        cand = fechamento(cand, pp.fecho) & zona_t
    lab_t, nt = rotular(cand)
    trincas = np.zeros_like(cand); esq_total = np.zeros_like(cand); reg_t = []
    if nt:
        area = np.bincount(lab_t.ravel(), minlength=nt + 1)
        esq = afinar(cand)
        comp = np.bincount(lab_t[esq], minlength=nt + 1)
        bif = np.bincount(lab_t[esq & (contar_vizinhos8(esq) >= 3)], minlength=nt + 1)
        if pp.trinca_desloc > -100000:
            esc_t = interpolar_bilinear(np.where(valido, cls["trinca_escore"], -(1 << 20)), g.shape)
            soma_esc = np.bincount(lab_t.ravel(), weights=esc_t.ravel().astype(np.float64), minlength=nt + 1).astype(np.int64)
            lim_t = cls["trinca_limiar"] + pp.trinca_desloc
        ys, xs = np.nonzero(lab_t); ls = lab_t[ys, xs]
        x0 = np.full(nt + 1, 1 << 30); x1 = np.full(nt + 1, -1); y0 = np.full(nt + 1, 1 << 30); y1 = np.full(nt + 1, -1)
        np.minimum.at(x0, ls, xs); np.maximum.at(x1, ls, xs); np.minimum.at(y0, ls, ys); np.maximum.at(y1, ls, ys)
        for i in range(1, nt + 1):
            a_i, c_i = int(area[i]), int(comp[i])
            if a_i < pp.area_min or c_i < pp.comp_min or (16 * a_i) // c_i > pp.larg_max16:
                continue
            if pp.trinca_desloc > -100000 and int(soma_esc[i]) < lim_t * a_i:   # média >= limiar (sem divisão)
                continue
            w, h = int(x1[i] - x0[i] + 1), int(y1[i] - y0[i] + 1)
            if 100 * int(bif[i]) >= p.jacare_bif_100px * c_i and c_i >= p.jacare_comp_min:
                tipo = 3
            elif h >= 2 * w:
                tipo = 0
            elif w >= 2 * h:
                tipo = 1
            else:
                tipo = 2
            sel = lab_t == i
            trincas |= sel; esq_total |= esq & sel
            reg_t.append({"area": a_i, "comprimento": c_i, "largura16": (16 * a_i) // c_i,
                          "bifurcacoes": int(bif[i]), "bbox": (int(x0[i]), int(y0[i]), w, h), "tipo": TIPOS[tipo]})
    if pp.sev_media and pp.sev_alta:
        from dataclasses import replace as _rep
        p = _rep(p, sev_media_1000=pp.sev_media, sev_alta_1000=pp.sev_alta)
    q = quantificar(reg_t, reg_b, int(roi.sum()), p)
    return {"trincas": trincas, "esqueleto": esq_total, "buracos": bur, "blocos": cls,
            "regioes_trinca": reg_t, "regioes_buraco": reg_b, "quant": q, "aux": aux}
