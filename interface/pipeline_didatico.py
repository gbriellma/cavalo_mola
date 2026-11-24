"""Instrumentação visual do núcleo clássico (`pavimento.nucleo`).

Executa exclusivamente as funções reais e determinísticas de `pavimento.nucleo`,
capturando os mapas intermediários em NumPy, histogramas, componentes aceitos e
rejeitados, métricas contra o gabarito DNIT e imagens BGR prontas para exibição
nativa na interface gráfica Python.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pavimento.metricas import confusao, resumir
from pavimento.nucleo import (
    TIPOS,
    abertura,
    afinar,
    analisar,
    borda,
    contar_vizinhos8,
    erodir,
    estatisticas_locais,
    fechamento,
    histerese_componentes,
    intensidade_hsi,
    mediana_mad,
    morfologia_direcional,
    otsu_inteiro,
    preencher_buracos,
    raiz_inteira,
    reduzir_2x,
    reduzir_mascara_2x,
    rotular,
)
from pavimento.parametros import Parametros
from pavimento.visualizacao import comparar, legenda, painel, sobrepor


def _cinza_para_bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _desenhar_contorno_roi(
    bgr: np.ndarray,
    roi: np.ndarray,
    cor: Tuple[int, int, int] = (235, 195, 50),
    espessura: int = 1,
) -> np.ndarray:
    out = bgr.copy()
    cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, cor, espessura)
    return out


def _desenhar_sobreposicao_mascara(
    base_bgr: np.ndarray,
    mascara: np.ndarray,
    cor_bgr: Tuple[int, int, int],
    alpha: float = 0.65,
) -> np.ndarray:
    out = base_bgr.copy()
    if mascara.any():
        cor = np.array(cor_bgr, dtype=np.float32)
        out[mascara] = np.clip(
            out[mascara].astype(np.float32) * (1.0 - alpha) + cor * alpha, 0, 255
        ).astype(np.uint8)
    return out


def _motivo_mais_frequente(rejeitados: List[dict]) -> str:
    """Retorna o motivo de rejeição mais comum entre componentes descartados."""
    motivos = [m for r in rejeitados for m in r.get("motivos", [])]
    if not motivos:
        return "Nenhum rejeitado"
    return Counter(motivos).most_common(1)[0][0]


def executar_pipeline_didatico(
    img_bgr: np.ndarray,
    roi_full: np.ndarray,
    p: Parametros,
    aplicar_reducao_2x: bool = True,
    gt_trinca_full: Optional[np.ndarray] = None,
    gt_buraco_full: Optional[np.ndarray] = None,
) -> dict:
    """Executa o pipeline completo instrumentado com paridade bit a bit ao núcleo oficial."""
    h0, w0 = img_bgr.shape[:2]

    # ------------------------------------------------------------------ #
    # ETAPA 1: Aquisição e Pré-processamento                             #
    # ------------------------------------------------------------------ #
    hsi_full = intensidade_hsi(img_bgr)
    reduzida = bool(aplicar_reducao_2x and h0 >= 32 and w0 >= 32)
    if reduzida:
        g = reduzir_2x(hsi_full)
        roi = reduzir_mascara_2x(roi_full, "maioria")
        orig_proc = cv2.resize(img_bgr, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        gt_t = reduzir_mascara_2x(gt_trinca_full, "qualquer") if gt_trinca_full is not None else None
        gt_b = reduzir_mascara_2x(gt_buraco_full, "qualquer") if gt_buraco_full is not None else None
    else:
        g = hsi_full
        roi = roi_full.astype(bool)
        orig_proc = img_bgr.copy()
        gt_t = gt_trinca_full.astype(bool) if gt_trinca_full is not None else None
        gt_b = gt_buraco_full.astype(bool) if gt_buraco_full is not None else None

    h, w = g.shape
    n_total = h * w
    n_roi = int(roi.sum())

    # Cronometra o núcleo OFICIAL puro (exatamente como roda na avaliação)
    t0 = time.perf_counter()
    res_oficial = analisar(g, roi, p)
    tempo_nucleo_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # ------------------------------------------------------------------ #
    # ETAPA 2: Filtragem Espacial (m_xy e s_xy em O(1))                  #
    # ------------------------------------------------------------------ #
    media, var = estatisticas_locais(g, p.raio)
    desvio = raiz_inteira(var)
    if p.raio_buraco == p.raio:
        desvio_b = desvio
    else:
        _, var_b = estatisticas_locais(g, p.raio_buraco)
        desvio_b = raiz_inteira(var_b)

    contraste_local = np.clip(media - g.astype(np.int32), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # ETAPA 3B & 4 (Buracos): Textura Mediana+MAD -> Morfologia -> Forma #
    # ------------------------------------------------------------------ #
    med_tex, mad_tex = mediana_mad(desvio_b[roi].astype(np.int64))
    t_tex = med_tex + (p.buraco_k4 * mad_tex) // 4
    g_med, g_mad = mediana_mad(g[roi].astype(np.int64))
    t_otsu = otsu_inteiro(g[roi])
    t_fraco = g_med - p.buraco_margem_fraca
    t_forte = g_med - max(p.buraco_margem_forte, p.buraco_margem_fraca)
    if p.buraco_usar_otsu:
        t_fraco, t_forte = min(t_fraco, t_otsu), min(t_forte, t_otsu)

    escura = (g.astype(np.int32) <= t_fraco) & roi
    fraca_b_esc = (media <= t_fraco) & escura

    if p.buraco_modo == 1:
        cand_b = (desvio_b > t_tex) & roi
        b_aberto = abertura(cand_b, p.buraco_abertura) & roi
        b_fechado = fechamento(b_aberto, p.buraco_fecho) & roi
        b_preenchido = preencher_buracos(b_fechado) & roi
    else:
        cand_b = (desvio_b > t_tex) & (media <= t_forte) & fraca_b_esc
        b_aberto = abertura(fraca_b_esc, p.buraco_abertura) & roi
        b_hist = histerese_componentes(cand_b, b_aberto)
        b_fechado = fechamento(b_hist, p.buraco_fecho) & roi
        b_preenchido = preencher_buracos(b_fechado) & roi

    lab_b, nb = rotular(b_preenchido)
    buracos = np.zeros_like(b_preenchido)
    buracos_rejeitados = np.zeros_like(b_preenchido)
    reg_b: List[dict] = []
    rej_b: List[dict] = []
    faixa_borda = (
        roi & ~erodir(roi, 2 * p.buraco_borda_margem + 1)
        if p.buraco_borda_margem > 0
        else np.zeros_like(roi)
    )
    if nb:
        area = np.bincount(lab_b.ravel(), minlength=nb + 1)
        per = np.bincount(lab_b[borda(b_preenchido)], minlength=nb + 1)
        soma_g = np.zeros(nb + 1, dtype=np.int64)
        np.add.at(soma_g, lab_b.ravel(), g.ravel().astype(np.int64))
        area_escura = np.bincount(lab_b[escura], minlength=nb + 1)
        toca_borda = np.bincount(lab_b[faixa_borda], minlength=nb + 1)
        ys, xs = np.nonzero(b_preenchido)
        ls = lab_b[ys, xs]
        x0 = np.full(nb + 1, w, dtype=np.int64)
        y0 = np.full(nb + 1, h, dtype=np.int64)
        x1 = np.full(nb + 1, -1, dtype=np.int64)
        y1 = np.full(nb + 1, -1, dtype=np.int64)
        np.minimum.at(x0, ls, xs)
        np.maximum.at(x1, ls, xs)
        np.minimum.at(y0, ls, ys)
        np.maximum.at(y1, ls, ys)
        for i in range(1, nb + 1):
            a_i, p_i = int(area[i]), int(per[i])
            circ = (12566 * a_i) // max(p_i * p_i, 1)
            esc_diff = a_i * g_med - int(soma_g[i])
            escuro_ok = esc_diff >= a_i * p.buraco_escuro_min
            bw, bh = int(x1[i] - x0[i] + 1), int(y1[i] - y0[i] + 1)
            extensao = (1000 * a_i) // max(bw * bh, 1)
            aspecto = (1000 * max(bw, bh)) // max(min(bw, bh), 1)
            fracao_escura = (1000 * int(area_escura[i])) // max(a_i, 1)

            motivos = []
            if a_i < p.buraco_area_min:
                motivos.append(f"Área {a_i} < {p.buraco_area_min} px")
            if circ < p.buraco_circ_min_1000:
                motivos.append(f"Circ. {circ/1000:.2f} < {p.buraco_circ_min_1000/1000:.2f}")
            if not escuro_ok:
                motivos.append(f"Escurec. {esc_diff/max(a_i,1):.1f} < {p.buraco_escuro_min}")
            if extensao < p.buraco_extensao_min_1000:
                motivos.append(f"Extensão {extensao/1000:.2f} < {p.buraco_extensao_min_1000/1000:.2f}")
            if aspecto > p.buraco_aspecto_max_1000:
                motivos.append(f"Aspecto {aspecto/1000:.2f} > {p.buraco_aspecto_max_1000/1000:.2f}")
            if fracao_escura < p.buraco_fracao_escura_min_1000:
                motivos.append(f"Fração esc. {fracao_escura/1000:.2f} < {p.buraco_fracao_escura_min_1000/1000:.2f}")
            if toca_borda[i] != 0:
                motivos.append("Toca borda da ROI")

            info_reg = {
                "id": len(reg_b) + 1 if not motivos else i,
                "area": a_i,
                "perimetro": p_i,
                "circularidade_1000": circ,
                "escurecimento_medio": round(esc_diff / max(a_i, 1), 1),
                "bbox": [int(x0[i]), int(y0[i]), bw, bh],
                "extensao_1000": extensao,
                "aspecto_1000": aspecto,
                "fracao_escura_1000": fracao_escura,
            }
            if not motivos:
                buracos |= lab_b == i
                reg_b.append(info_reg)
            else:
                buracos_rejeitados |= lab_b == i
                info_reg["motivos"] = motivos
                rej_b.append(info_reg)

    # ------------------------------------------------------------------ #
    # ETAPA 3A & 4 & 5 (Trincas): Limiar Variável -> Fecho Dir -> Hist   #
    # ------------------------------------------------------------------ #
    permitido = roi & ~buracos

    def candidato(a16: int, b16: int) -> np.ndarray:
        d = 16 * (media - g.astype(np.int32)) - b16
        return permitido & (d > 0) & (d.astype(np.int64) ** 2 > a16 ** 2 * var.astype(np.int64))

    forte_t = candidato(p.trinca_a16, p.trinca_b16)
    fraca_bruta_t = candidato(
        min(p.trinca_a16, p.trinca_fraca_a16),
        min(p.trinca_b16, p.trinca_fraca_b16),
    ) | forte_t
    fraca_t = morfologia_direcional(fraca_bruta_t, p.trinca_fecho, fechar=True) & permitido
    suporte_t = morfologia_direcional(fraca_t, p.trinca_abertura_linha)
    cand_t = histerese_componentes(forte_t, fraca_t)

    lab_t, nt = rotular(cand_t)
    trincas = np.zeros_like(cand_t)
    trincas_rejeitadas = np.zeros_like(cand_t)
    esq_total = np.zeros_like(cand_t)
    bif_total = np.zeros_like(cand_t)
    reg_t: List[dict] = []
    rej_t: List[dict] = []

    if nt:
        area_t = np.bincount(lab_t.ravel(), minlength=nt + 1)
        suporte = np.bincount(lab_t[suporte_t & cand_t], minlength=nt + 1)
        esq = afinar(cand_t)
        viz8 = contar_vizinhos8(esq)
        bif_mask = esq & (viz8 >= 3)
        comp = np.bincount(lab_t[esq], minlength=nt + 1)
        bif = np.bincount(lab_t[bif_mask], minlength=nt + 1)
        ys, xs = np.nonzero(lab_t)
        ls = lab_t[ys, xs]
        x0 = np.full(nt + 1, 1 << 30)
        x1 = np.full(nt + 1, -1)
        y0 = np.full(nt + 1, 1 << 30)
        y1 = np.full(nt + 1, -1)
        np.minimum.at(x0, ls, xs)
        np.maximum.at(x1, ls, xs)
        np.minimum.at(y0, ls, ys)
        np.maximum.at(y1, ls, ys)
        for i in range(1, nt + 1):
            a_i, c_i = int(area_t[i]), int(comp[i])
            tw, th = int(x1[i] - x0[i] + 1), int(y1[i] - y0[i] + 1)
            larg16 = (16 * a_i) // max(c_i, 1)
            motivos = []
            if a_i < p.trinca_area_min:
                motivos.append(f"Área {a_i} < {p.trinca_area_min} px")
            if c_i < p.trinca_comp_min:
                motivos.append(f"Comp. {c_i} < {p.trinca_comp_min} px")
            if suporte[i] == 0 and c_i < p.trinca_abertura_linha:
                motivos.append(f"Sem suporte linear ({p.trinca_abertura_linha} px)")
            if c_i > 0 and larg16 > p.trinca_larg_max16:
                motivos.append(f"Largura {larg16/16:.1f} > {p.trinca_larg_max16/16:.1f} px")
            if c_i > 0 and 16 * c_i * c_i < p.trinca_along_min16 * a_i:
                along = (c_i * c_i) / max(a_i, 1)
                motivos.append(f"Along. {along:.1f} < {p.trinca_along_min16/16:.1f}")

            sel = lab_t == i
            if not motivos:
                if 100 * int(bif[i]) >= p.jacare_bif_100px * c_i and c_i >= p.jacare_comp_min:
                    tipo_idx = 3
                elif th >= 2 * tw:
                    tipo_idx = 0
                elif tw >= 2 * th:
                    tipo_idx = 1
                else:
                    tipo_idx = 2
                trincas |= sel
                esq_total |= esq & sel
                bif_total |= bif_mask & sel
                reg_t.append({
                    "id": len(reg_t) + 1,
                    "area": a_i,
                    "comprimento": c_i,
                    "largura16": larg16,
                    "largura_px": round(larg16 / 16.0, 2),
                    "bifurcacoes": int(bif[i]),
                    "bbox": [int(x0[i]), int(y0[i]), tw, th],
                    "tipo": TIPOS[tipo_idx],
                })
            else:
                trincas_rejeitadas |= sel
                rej_t.append({
                    "id": i,
                    "area": a_i,
                    "comprimento": c_i,
                    "largura_px": round(larg16 / 16.0, 2) if c_i else 0.0,
                    "bbox": [int(x0[i]), int(y0[i]), tw, th],
                    "motivos": motivos,
                })

    # Garantia absoluta de paridade bit a bit com nucleo.analisar()
    assert np.array_equal(trincas, res_oficial["trincas"]), "Divergência em trincas vs nucleo.analisar"
    assert np.array_equal(buracos, res_oficial["buracos"]), "Divergência em buracos vs nucleo.analisar"
    assert np.array_equal(esq_total, res_oficial["esqueleto"]), "Divergência em esqueleto vs nucleo.analisar"

    q = res_oficial["quant"]

    # ------------------------------------------------------------------ #
    # RENDERIZAÇÃO VISUAL DE TODAS AS ETAPAS (BGR nativo)                #
    # ------------------------------------------------------------------ #
    g_dim = _cinza_para_bgr(np.where(roi, g // 3, g // 6))
    otsu_mask = (g.astype(np.int32) <= t_otsu) & roi

    # Etapa 1
    img_1_orig = orig_proc
    img_1_hsi = _cinza_para_bgr(g)
    img_1_roi = _desenhar_contorno_roi(_cinza_para_bgr(np.where(roi, g, g // 3)), roi, (235, 195, 50), 1)

    # Etapa 2
    img_2_media = _desenhar_contorno_roi(_cinza_para_bgr(media), roi, (160, 140, 60), 1)
    sd_vis = np.clip(desvio.astype(np.int32) * 6, 0, 255).astype(np.uint8)
    sd_b_vis = np.clip(desvio_b.astype(np.int32) * 6, 0, 255).astype(np.uint8)
    img_2_desvio = _desenhar_contorno_roi(_cinza_para_bgr(sd_vis), roi, (160, 140, 60), 1)
    img_2_desvio_b = _desenhar_contorno_roi(_cinza_para_bgr(sd_b_vis), roi, (160, 140, 60), 1)
    img_2_contraste = _desenhar_contorno_roi(
        _cinza_para_bgr(np.clip(contraste_local.astype(np.int32) * 4, 0, 255)), roi, (160, 140, 60), 1
    )

    # Etapa 3A — Trincas
    img_3a_sementes = _desenhar_contorno_roi(g_dim, roi, (120, 100, 40), 1)
    img_3a_sementes[fraca_bruta_t] = (230, 185, 55)     # ciano = candidata fraca
    img_3a_sementes[forte_t] = (25, 195, 255)           # âmbar = semente forte

    fecho_adicionados = fraca_t & ~fraca_bruta_t
    img_3a_fecho_dir = _desenhar_contorno_roi(g_dim, roi, (120, 100, 40), 1)
    img_3a_fecho_dir[fraca_bruta_t] = (205, 155, 55)
    img_3a_fecho_dir[fecho_adicionados] = (80, 235, 95)  # verde = religado em 4 ângulos
    img_3a_fecho_dir[forte_t] = (25, 195, 255)

    descartados_hist = fraca_t & ~cand_t
    img_3a_histerese = _desenhar_contorno_roi(g_dim, roi, (120, 100, 40), 1)
    img_3a_histerese = _desenhar_sobreposicao_mascara(img_3a_histerese, descartados_hist, (175, 85, 155), 0.72)
    img_3a_histerese[cand_t] = (65, 90, 245)             # vermelho = mantido pela histerese
    img_3a_histerese[forte_t & cand_t] = (30, 215, 255)  # amarelo = semente forte

    img_3a_otsu = _desenhar_contorno_roi(_cinza_para_bgr(np.where(roi, g // 2, g // 5)), roi, (160, 140, 60), 1)
    img_3a_otsu = _desenhar_sobreposicao_mascara(img_3a_otsu, otsu_mask, (70, 95, 240), 0.65)

    # Etapa 3B — Buracos (Textura + Mediana/MAD)
    img_3b_textura = _desenhar_contorno_roi(_cinza_para_bgr(sd_b_vis), roi, (160, 140, 60), 1)
    img_3b_textura = _desenhar_sobreposicao_mascara(img_3b_textura, cand_b, (20, 165, 250), 0.55)
    img_3b_cand = _desenhar_contorno_roi(np.zeros((h, w, 3), dtype=np.uint8), roi, (140, 120, 45), 1)
    img_3b_cand[cand_b] = (245, 245, 245)

    # Etapa 4 — Morfologia do Zero
    removido_abertura = cand_b & ~b_aberto
    img_4_abertura = _desenhar_contorno_roi(g_dim, roi, (120, 100, 40), 1)
    img_4_abertura = _desenhar_sobreposicao_mascara(img_4_abertura, removido_abertura, (75, 75, 235), 0.80)
    img_4_abertura[b_aberto] = (245, 245, 245)

    adicionado_fecho = b_fechado & ~b_aberto
    adicionado_preench = b_preenchido & ~b_fechado
    img_4_fecho_preench = _desenhar_contorno_roi(g_dim, roi, (120, 100, 40), 1)
    img_4_fecho_preench[b_aberto] = (235, 235, 235)
    img_4_fecho_preench[adicionado_fecho] = (80, 235, 110)     # verde = fechamento
    img_4_fecho_preench[adicionado_preench] = (245, 195, 45)   # ciano = preenchimento condicional

    # Etapa 5 — Componentes, Esqueleto e Quantificação
    img_5_buracos_forma = _desenhar_contorno_roi(
        _cinza_para_bgr(np.where(roi, g // 2, g // 5)), roi, (160, 140, 60), 1
    )
    img_5_buracos_forma = _desenhar_sobreposicao_mascara(img_5_buracos_forma, buracos_rejeitados, (75, 80, 220), 0.60)
    img_5_buracos_forma = _desenhar_sobreposicao_mascara(img_5_buracos_forma, buracos, (230, 80, 220), 0.55)
    cnts_b, _ = cv2.findContours(buracos.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_5_buracos_forma, cnts_b, -1, (255, 120, 255), 2)

    img_5_esqueleto = _desenhar_contorno_roi(np.zeros((h, w, 3), dtype=np.uint8), roi, (180, 150, 45), 1)
    img_5_esqueleto[trincas_rejeitadas] = (110, 75, 115)       # roxo apagado = trinca rejeitada
    img_5_esqueleto[trincas] = (40, 40, 220)                   # vermelho = trinca aceita
    img_5_esqueleto[esq_total] = (25, 240, 255)                # amarelo = esqueleto 1 px
    if bif_total.any():
        by_s, bx_s = np.nonzero(bif_total)
        for bx, by in zip(bx_s, by_s):
            cv2.circle(img_5_esqueleto, (int(bx), int(by)), 2, (255, 235, 40), -1)

    img_5_final = sobrepor(g, roi, trincas, esq_total, buracos)

    # Comparação com o gabarito real (quando a amostra vem do dataset DNIT)
    metricas_gt = None
    imagens_extra_gt: Dict[str, np.ndarray] = {}
    if gt_t is not None and gt_b is not None:
        mt_tol2 = resumir(*confusao(trincas, gt_t, roi, 2))
        mt_exato = resumir(*confusao(trincas, gt_t, roi, 0))
        mb = resumir(*confusao(buracos, gt_b, roi, 0))
        metricas_gt = {
            "px_gt_trinca": int((gt_t & roi).sum()),
            "px_gt_buraco": int((gt_b & roi).sum()),
            "trinca_tol2": mt_tol2,
            "trinca_exato": mt_exato,
            "buraco": mb,
        }
        img_gt_cmp = _desenhar_contorno_roi(comparar(g, trincas, buracos, gt_t, gt_b), roi, (235, 195, 50), 1)
        img_gt_rot = _desenhar_contorno_roi(_cinza_para_bgr(np.where(roi, g, g // 3)), roi, (235, 195, 50), 1)
        img_gt_rot = _desenhar_sobreposicao_mascara(img_gt_rot, gt_t & roi, (30, 175, 255), 0.78)
        img_gt_rot = _desenhar_sobreposicao_mascara(img_gt_rot, gt_b & roi, (70, 230, 80), 0.65)
        cnts_gtb, _ = cv2.findContours((gt_b & roi).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_gt_rot, cnts_gtb, -1, (40, 235, 255), 2)
        imagens_extra_gt["5_gt_comparar"] = img_gt_cmp
        imagens_extra_gt["5_gt_rotulo"] = img_gt_rot

    # Histogramas para o gráfico nativo da Etapa 3B
    hist_tex_raw = np.bincount(np.clip(desvio_b[roi], 0, 127).astype(np.int32), minlength=128)
    max_tex_bin = max(int(t_tex) + 12, int(np.max(np.nonzero(hist_tex_raw)[0])) + 2) if n_roi else 64
    max_tex_bin = min(128, max(32, max_tex_bin))
    hist_g_raw = np.bincount(g[roi].astype(np.int32), minlength=256) if n_roi else np.zeros(256, dtype=np.int64)

    comp = q["comprimento_trincas_px"]
    comp_jacare = sum(r["comprimento"] for r in reg_t if r["tipo"] == "couro_de_jacare")
    comp_pond = comp + comp_jacare
    n_roi_safe = max(n_roi, 1)
    termo_trinca = (p.dano_peso_trinca * 1000 * comp_pond) // n_roi_safe
    termo_buraco = (p.dano_peso_buraco * q["area_buracos_pct_100"]) // 10

    imagens_bgr: Dict[str, np.ndarray] = {
        "1_original": img_1_orig,
        "1_hsi": img_1_hsi,
        "1_roi": img_1_roi,
        "2_media": img_2_media,
        "2_desvio": img_2_desvio,
        "2_desvio_b": img_2_desvio_b,
        "2_contraste": img_2_contraste,
        "3a_sementes": img_3a_sementes,
        "3a_fecho_dir": img_3a_fecho_dir,
        "3a_histerese": img_3a_histerese,
        "3a_otsu": img_3a_otsu,
        "3b_textura": img_3b_textura,
        "3b_cand": img_3b_cand,
        "4_abertura": img_4_abertura,
        "4_fecho_preench": img_4_fecho_preench,
        "5_buracos_forma": img_5_buracos_forma,
        "5_esqueleto": img_5_esqueleto,
        "5_final": img_5_final,
        **imagens_extra_gt,
    }

    mapas = {
        "g": g,
        "media": media,
        "desvio": desvio,
        "desvio_b": desvio_b,
        "roi": roi,
        "forte_t": forte_t,
        "fraca_bruta_t": fraca_bruta_t,
        "fraca_t": fraca_t,
        "cand_t": cand_t,
        "trincas": trincas,
        "esq_total": esq_total,
        "bif_total": bif_total,
        "cand_b": cand_b,
        "b_aberto": b_aberto,
        "b_fechado": b_fechado,
        "b_preenchido": b_preenchido,
        "buracos": buracos,
        "otsu_mask": otsu_mask,
    }

    return {
        "dimensoes": {
            "original": [w0, h0],
            "processamento": [w, h],
            "reduzida_2x": reduzida,
            "pixels_total": n_total,
            "pixels_roi": n_roi,
            "roi_pct": round(100.0 * n_roi / max(n_total, 1), 1),
        },
        "tempo_nucleo_ms": tempo_nucleo_ms,
        "params": asdict(p),
        "estatisticas": {
            "janela_trinca": 2 * p.raio + 1,
            "janela_buraco": 2 * p.raio_buraco + 1,
            "media_roi": round(float(g[roi].mean()), 1) if n_roi else 0.0,
            "mediana_g": int(g_med),
            "mad_g": int(g_mad),
            "mediana_textura": int(med_tex),
            "mad_textura": int(mad_tex),
            "limiar_textura": int(t_tex),
            "limiar_otsu": int(t_otsu),
            "limiar_escuro": int(t_fraco),
            "limiar_escuro_forte": int(t_forte),
            "px_trinca_forte": int(forte_t.sum()),
            "px_trinca_fraca_bruta": int(fraca_bruta_t.sum()),
            "px_trinca_fecho_adicionados": int(fecho_adicionados.sum()),
            "px_trinca_fraca_pos_fecho": int(fraca_t.sum()),
            "px_trinca_pos_histerese": int(cand_t.sum()),
            "px_trinca_descartados_histerese": int(descartados_hist.sum()),
            "px_trinca_final": int(trincas.sum()),
            "px_esqueleto": int(esq_total.sum()),
            "px_bifurcacoes": int(bif_total.sum()),
            "px_otsu": int(otsu_mask.sum()),
            "px_buraco_cand": int(cand_b.sum()),
            "px_buraco_pos_abertura": int(b_aberto.sum()),
            "px_buraco_removido_abertura": int(removido_abertura.sum()),
            "px_buraco_adicionado_fecho": int(adicionado_fecho.sum()),
            "px_buraco_adicionado_preench": int(adicionado_preench.sum()),
            "px_buraco_pos_preench": int(b_preenchido.sum()),
            "px_buraco_final": int(buracos.sum()),
            "n_comp_trinca_brutos": int(nt),
            "n_comp_trinca_aceitos": len(reg_t),
            "n_comp_trinca_rejeitados": len(rej_t),
            "n_comp_buraco_brutos": int(nb),
            "n_comp_buraco_aceitos": len(reg_b),
            "n_comp_buraco_rejeitados": len(rej_b),
        },
        "histogramas": {
            "textura": hist_tex_raw[:max_tex_bin].tolist(),
            "intensidade": hist_g_raw.tolist(),
        },
        "quant": q,
        "formula_dano": {
            "comp_total": comp,
            "comp_jacare": comp_jacare,
            "comp_ponderado": comp_pond,
            "termo_trinca_1000": int(termo_trinca),
            "termo_buraco_1000": int(termo_buraco),
            "soma_bruta_1000": int(termo_trinca + termo_buraco),
            "indice_final_1000": int(q["indice_dano_1000"]),
        },
        "regioes_trinca": reg_t,
        "rejeitados_trinca": rej_t[:60],
        "regioes_buraco": reg_b,
        "rejeitados_buraco": rej_b[:60],
        "metricas_gt": metricas_gt,
        "imagens_bgr": imagens_bgr,
        "mapas": mapas,
        "resumo_didatico": {
            "etapa1": {
                "pct_fora_roi": round(100.0 - (100.0 * n_roi / max(n_total, 1)), 1),
                "fator_pixels": 4 if reduzida else 1,
                "pixels_fora_roi": int(n_total - n_roi),
            },
            "etapa2": {
                "s_xy_mediana_roi": int(np.median(desvio[roi])) if n_roi else 0,
                "s_xy_p95_roi": int(np.percentile(desvio[roi], 95)) if n_roi else 0,
                "tempo_nucleo_ms": tempo_nucleo_ms,
            },
            "etapa3a": {
                "px_otsu": int(otsu_mask.sum()),
                "px_histerese": int(cand_t.sum()),
                "pct_ruido_removido": round(100.0 * int(descartados_hist.sum()) / max(int(fraca_t.sum()), 1), 1),
                "otsu_vs_histerese_x": round(int(otsu_mask.sum()) / max(int(cand_t.sum()), 1), 1),
            },
            "etapa3b": {
                "pct_pista_cand": round(100.0 * int(cand_b.sum()) / max(n_roi, 1), 2),
                "mediana_textura": int(med_tex),
                "mad_textura": int(mad_tex),
                "limiar_textura": int(t_tex),
            },
            "etapa4": {
                "pct_removido_abertura": round(100.0 * int(removido_abertura.sum()) / max(int(cand_b.sum()), 1), 1),
                "px_removido_abertura": int(removido_abertura.sum()),
                "px_preenchido_miolo": int(adicionado_preench.sum()),
                "px_religado_trincas": int(fecho_adicionados.sum()),
            },
            "etapa5": {
                "trincas_rejeitadas_pct": round(100.0 * len(rej_t) / max(int(nt), 1), 1),
                "buracos_rejeitados_pct": round(100.0 * len(rej_b) / max(int(nb), 1), 1),
                "motivo_trinca_top": _motivo_mais_frequente(rej_t),
                "motivo_buraco_top": _motivo_mais_frequente(rej_b),
                "severidade_label": {
                    "nenhum": "Nenhum",
                    "baixa": "Baixa",
                    "media": "Média",
                    "alta": "Alta",
                }.get(q["severidade"], q["severidade"]),
            },
        },
    }


def inspecionar_pixel(resultado: dict, x: int, y: int) -> Optional[dict]:
    """Consulta os valores inteiros exatos do pipeline no pixel (x, y)."""
    mapas = resultado["mapas"]
    g = mapas["g"]
    h, w = g.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    iv = int(g[y, x])
    mv = int(mapas["media"][y, x])
    sv = int(mapas["desvio"][y, x])
    sb = int(mapas["desvio_b"][y, x])
    em_roi = bool(mapas["roi"][y, x])
    p = resultado["params"]
    t_tex = int(resultado["estatisticas"]["limiar_textura"])

    queda = mv - iv
    lim_forte = (p["trinca_a16"] * sv + p["trinca_b16"]) / 16.0
    lim_fraco = (
        min(p["trinca_a16"], p["trinca_fraca_a16"]) * sv
        + min(p["trinca_b16"], p["trinca_fraca_b16"])
    ) / 16.0

    if not em_roi:
        estado = "Fora da ROI (ignorado)"
    elif mapas["buracos"][y, x]:
        estado = "Buraco aceito"
    elif mapas["bif_total"][y, x]:
        estado = "Trinca (bifurcação)"
    elif mapas["esq_total"][y, x]:
        estado = "Trinca (esqueleto 1 px)"
    elif mapas["trincas"][y, x]:
        estado = "Trinca aceita"
    elif mapas["cand_t"][y, x]:
        estado = "Cand. trinca (rejeitado)"
    elif mapas["fraca_t"][y, x]:
        estado = "Ruído fraco (sem semente)"
    elif mapas["cand_b"][y, x]:
        estado = "Cand. buraco (filtrado)"
    else:
        estado = "Pista íntegra"

    if bool(mapas["forte_t"][y, x]):
        status_trinca = f"m−I={queda:+d} > {lim_forte:.1f} (Forte)"
    elif bool(mapas["fraca_bruta_t"][y, x]):
        status_trinca = f"m−I={queda:+d} > {lim_fraco:.1f} (Fraca)"
    else:
        status_trinca = f"m−I={queda:+d} ≤ {lim_fraco:.1f}"

    status_buraco = f"s_b={sb} {'>' if sb > t_tex else '≤'} t_tex={t_tex}"

    return {
        "x": x,
        "y": y,
        "intensidade": iv,
        "media": mv,
        "desvio": sv,
        "desvio_b": sb,
        "em_roi": em_roi,
        "status_trinca": status_trinca,
        "status_buraco": status_buraco,
        "estado": estado,
    }


def gerar_painel_oficial_bgr(
    img_bgr: np.ndarray,
    roi_full: np.ndarray,
    p: Parametros,
    aplicar_reducao_2x: bool = True,
    gt_trinca_full: Optional[np.ndarray] = None,
    gt_buraco_full: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gera o painel oficial de 9 ou 10 etapas (mesmo formato de scripts/processar_imagem.py)."""
    h0, w0 = img_bgr.shape[:2]
    hsi = intensidade_hsi(img_bgr)
    if aplicar_reducao_2x and h0 >= 32 and w0 >= 32:
        g = reduzir_2x(hsi)
        roi = reduzir_mascara_2x(roi_full, "maioria")
        gt_t = reduzir_mascara_2x(gt_trinca_full, "qualquer") if gt_trinca_full is not None else None
        gt_b = reduzir_mascara_2x(gt_buraco_full, "qualquer") if gt_buraco_full is not None else None
    else:
        g = hsi
        roi = roi_full.astype(bool)
        gt_t = gt_trinca_full.astype(bool) if gt_trinca_full is not None else None
        gt_b = gt_buraco_full.astype(bool) if gt_buraco_full is not None else None
    r = analisar(g, roi, p)
    sd = np.clip(r["desvio"].astype(np.int32) * 6, 0, 255).astype(np.uint8)
    etapas = [
        ("1. Original", cv2.resize(img_bgr, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)),
        ("2. Intensidade HSI + ROI", np.where(roi, g, g // 3).astype(np.uint8)),
        ("3. Media local m_xy", r["media"].astype(np.uint8)),
        ("4. Desvio local s_xy (textura)", sd),
        ("5. Limiar variavel (trincas)", r["cand_trinca"]),
        ("6. Limiar textura (buracos)", r["cand_buraco"]),
        ("7. Trincas filtradas + esqueleto", sobrepor(np.zeros_like(g), roi, r["trincas"], r["esqueleto"], np.zeros_like(roi))),
        ("8. Buracos (abertura+preench.)", r["buracos"]),
        ("9. Resultado quantificado", legenda(sobrepor(g, roi, r["trincas"], r["esqueleto"], r["buracos"]), r["quant"])),
    ]
    if gt_t is not None and gt_b is not None:
        etapas.append(("10. vs rotulo (verde=nao detect.)", comparar(g, r["trincas"], r["buracos"], gt_t, gt_b)))
    return painel(etapas, colunas=3)


def gerar_painel_oficial_png(
    img_bgr: np.ndarray,
    roi_full: np.ndarray,
    p: Parametros,
    aplicar_reducao_2x: bool = True,
    gt_trinca_full: Optional[np.ndarray] = None,
    gt_buraco_full: Optional[np.ndarray] = None,
) -> bytes:
    """Codifica o painel oficial em bytes PNG."""
    out = gerar_painel_oficial_bgr(img_bgr, roi_full, p, aplicar_reducao_2x, gt_trinca_full, gt_buraco_full)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("Falha ao codificar painel PNG")
    return buf.tobytes()
