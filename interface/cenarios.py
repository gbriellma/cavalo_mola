"""Cenários prontos (fotos reais da apresentação + cenários didáticos sintéticos).

Todos os cenários sintéticos são 100% determinísticos (aritmética inteira sem RNG)
e construídos para evidenciar cada etapa do pipeline clássico em `pavimento.nucleo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PASTA_EXEMPLOS = Path(__file__).resolve().parent / "exemplos"


@dataclass(frozen=True)
class CenarioInfo:
    id: str
    titulo: str
    categoria: str  # "real" | "didatico"
    subtitulo: str
    descricao: str
    reduzir_padrao: bool
    roi_poligono: Tuple[Tuple[float, float], ...]  # pontos normalizados (x, y) em [0..1]


CENARIOS: Tuple[CenarioInfo, ...] = (
    CenarioInfo(
        id="rodovia_dnit_01",
        titulo="Rodovia Real (DNIT · Slides 2, 5–7)",
        categoria="real",
        subtitulo="512×293 RGB · Trincas + faixa branca + textura lateral",
        descricao=(
            "Foto real da base DNIT usada nos Slides 2, 5, 6 e 7 da apresentação. "
            "Permite ver o pipeline completo em asfalto real: trincas no trecho direito, "
            "faixa branca lateral removida pela abertura e acostamento/vegetação ignorados pela ROI."
        ),
        reduzir_padrao=False,
        roi_poligono=((0.295, 0.00), (0.656, 0.00), (1.00, 0.80), (1.00, 1.00), (0.00, 1.00), (0.00, 0.24)),
    ),
    CenarioInfo(
        id="rodovia_df251",
        titulo="Rodovia Real — DF-251 (Slide 8)",
        categoria="real",
        subtitulo="512×227 RGB · Fissuração severa (couro de jacaré)",
        descricao=(
            "Foto real da rodovia DF-251 apresentada no Slide 8. "
            "Pavimento fortemente degradado em couro de jacaré — excelente para "
            "observar o esqueleto Zhang-Suen, as bifurcações e o índice de severidade alta."
        ),
        reduzir_padrao=False,
        roi_poligono=((0.576, 0.00), (0.758, 0.00), (1.00, 0.90), (1.00, 1.00), (0.016, 1.00)),
    ),
    CenarioInfo(
        id="didatico_sombra_trinca",
        titulo="Didático: Trincas sob Sombra + Histerese",
        categoria="didatico",
        subtitulo="1024×640 RGB → 512×320 · Sol/sombra + pontas fracas + ruído",
        descricao=(
            "Cenário controlado com sombra projetada na metade direita da pista, "
            "trincas que atravessam do sol para a sombra (com pontas mais claras) e "
            "manchas fracas isoladas. Mostra por que o limiar variável vence o limiar global "
            "e como a histerese preserva as pontas eliminando ruído solto."
        ),
        reduzir_padrao=True,
        roi_poligono=((0.24, 0.00), (0.76, 0.00), (0.95, 1.00), (0.05, 1.00)),
    ),
    CenarioInfo(
        id="didatico_buraco_faixa",
        titulo="Didático: Buraco (Miolo Liso) + Faixa Pintada",
        categoria="didatico",
        subtitulo="1024×640 RGB → 512×320 · Textura MAD + abertura + preenchimento",
        descricao=(
            "Cenário projetado para o ramo de buracos: contém um buraco áspero com "
            "miolo liso (poça/poeira), uma faixa branca lateral de alto contraste e "
            "um risco alongado áspero. Evidencia abertura, preenchimento de buracos e "
            "rejeição por circularidade/aspecto."
        ),
        reduzir_padrao=True,
        roi_poligono=((0.22, 0.00), (0.78, 0.00), (0.95, 1.00), (0.05, 1.00)),
    ),
    CenarioInfo(
        id="didatico_completo",
        titulo="Didático: Todos os Tipos de Trinca + Buraco",
        categoria="didatico",
        subtitulo="1024×640 RGB → 512×320 · Long., Transv., Diag., Jacaré + Buraco",
        descricao=(
            "Mostruário completo da quantificação na Etapa 5: reúne trinca longitudinal, "
            "transversal, diagonal, rede em couro de jacaré (com bifurcações, peso 2×) "
            "e um buraco áspero na mesma pista."
        ),
        reduzir_padrao=True,
        roi_poligono=((0.22, 0.00), (0.78, 0.00), (0.95, 1.00), (0.05, 1.00)),
    ),
)

CENARIOS_POR_ID = {c.id: c for c in CENARIOS}


def poligono_para_mascara(shape: Tuple[int, int], poligono: List[Tuple[float, float]]) -> np.ndarray:
    """Converte polígono normalizado [(x, y) em 0..1] em máscara booleana (H, W)."""
    h, w = shape
    if not poligono or len(poligono) < 3:
        return np.ones((h, w), dtype=bool)
    pts = np.array(
        [[int(round(px * (w - 1))), int(round(py * (h - 1)))] for px, py in poligono],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask > 0


def _base_estrada_rgb(h: int = 640, w: int = 1024, poligono=((0.22, 0.0), (0.78, 0.0), (0.95, 1.0), (0.05, 1.0))):
    """Gera uma estrada sintética RGB determinística com asfalto, acostamento e vegetação."""
    y, x = np.indices((h, w), dtype=np.int32)
    # Grão multiescala em blocos 2×2 (sobrevive à redução 2× com mediana ~7 e MAD ~2 -> t_tex ~21)
    bx, by = x >> 1, y >> 1
    grao_fino = ((bx * 37 + by * 61 + (bx * by) * 7 + (bx >> 1) * 19) % 23) - 11
    # Modulação de aspereza natural do pavimento em faixas amplas (garante MAD = 2 realista)
    zona = ((bx >> 5) + (by >> 5) * 3) % 4
    amp = np.where(zona == 0, 2, np.where(zona == 1, 3, np.where(zona == 2, 5, 3)))
    grao = (grao_fino * amp) // 3 + (((x * 17 + y * 29) % 5) - 2)
    base_i = np.clip(156 - (y * 12) // h + grao, 0, 255).astype(np.int32)

    # BGR do asfalto (cinza realista)
    bgr = np.empty((h, w, 3), dtype=np.uint8)
    bgr[..., 0] = np.clip(base_i + 2, 0, 255)
    bgr[..., 1] = np.clip(base_i, 0, 255)
    bgr[..., 2] = np.clip(base_i - 2, 0, 255)

    # Acostamento avermelhado/escuro fora da ROI (sem degrau abrupto na borda) + vegetação nas margens
    roi = poligono_para_mascara((h, w), list(poligono))
    dist_fora = cv2.distanceTransform((~roi).astype(np.uint8), cv2.DIST_L2, 5)
    acost = (~roi) & (dist_fora <= 75)
    grama = (~roi) & (dist_fora > 75)
    bgr[acost, 0] = np.clip(base_i[acost] - 10, 30, 220)
    bgr[acost, 1] = np.clip(base_i[acost] - 8, 30, 220)
    bgr[acost, 2] = np.clip(base_i[acost] + 10, 30, 235)
    grama_var = ((bx * 23 + by * 41 + bx * by * 3) % 29) - 14
    bgr[grama, 0] = np.clip(58 + grama_var[grama], 20, 120)
    bgr[grama, 1] = np.clip(112 + grama_var[grama], 50, 180)
    bgr[grama, 2] = np.clip(64 + grama_var[grama], 25, 130)

    return bgr, roi


def _desenhar_trinca_variavel(bgr: np.ndarray, pontos: List[Tuple[int, int]],
                              contraste_centro: int = 62, espessura: int = 2,
                              contraste_ponta: int | None = None, fracao_ponta: float = 0.24):
    """Escurece uma polilinha de trinca de forma relativa ao asfalto local (funciona no sol e na sombra)."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(pontos, dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=espessura, lineType=cv2.LINE_8)
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return
    if contraste_ponta is None:
        queda = np.full(ys.size, contraste_centro, dtype=np.int32)
    else:
        p0, p1 = np.array(pontos[0], dtype=np.float64), np.array(pontos[-1], dtype=np.float64)
        v = p1 - p0
        denom = max(float(v @ v), 1.0)
        t = ((xs - p0[0]) * v[0] + (ys - p0[1]) * v[1]) / denom
        t = np.clip(t, 0.0, 1.0)
        dist_extremo = np.minimum(t, 1.0 - t)
        eh_ponta = dist_extremo < fracao_ponta
        queda = np.where(eh_ponta, contraste_ponta, contraste_centro).astype(np.int32)
    for c in range(3):
        bgr[ys, xs, c] = np.clip(bgr[ys, xs, c].astype(np.int32) - queda, 12, 255).astype(np.uint8)


def _desenhar_buraco_sintetico(bgr: np.ndarray, cx: int, cy: int, rx: int, ry: int,
                               centro_liso: bool = False):
    """Desenha um buraco oval áspero (com opcional miolo liso tipo poça/poeira)."""
    h, w = bgr.shape[:2]
    y, x = np.indices((h, w), dtype=np.int32)
    bx, by = x >> 1, y >> 1
    ang_mod = ((bx * 7 + by * 11) % 7) - 3
    dist = ((x - cx) * 100 // max(rx, 1)) ** 2 + ((y - cy) * 100 // max(ry, 1)) ** 2
    alvo = dist <= (100 + ang_mod) ** 2
    # Textura intensa em blocos 2×2 (desvio local ~42 >> t_tex ~21)
    tex = ((bx * 29 + by * 43 + (bx * by) * 13) % 156) - 78
    val = np.clip(102 + tex, 20, 225).astype(np.int32)

    if centro_liso:
        miolo = dist <= 52 ** 2
        val = np.where(miolo, 96 + (((bx * 11 + by * 17) % 5) - 2), val)

    for c in range(3):
        canal = bgr[..., c].astype(np.int32)
        tom = val + (-4 if c == 0 else (0 if c == 1 else 6))
        bgr[..., c] = np.where(alvo, np.clip(tom, 15, 230), canal).astype(np.uint8)


def _gerar_sombra_trinca() -> Tuple[np.ndarray, np.ndarray]:
    info = CENARIOS_POR_ID["didatico_sombra_trinca"]
    bgr, roi = _base_estrada_rgb(640, 1024, info.roi_poligono)
    h, w = bgr.shape[:2]
    y, x = np.indices((h, w), dtype=np.int32)

    # Sombra diagonal suave na metade direita (queda de 58 níveis em rampa de 80 px)
    dist_s = (x + (y * 2) // 5) - 560
    queda = np.clip((dist_s * 58) // 80, 0, 58)
    for c in range(3):
        bgr[..., c] = np.clip(bgr[..., c].astype(np.int32) - queda, 28, 255).astype(np.uint8)

    # 1) Trinca diagonal longa atravessando do sol até a sombra (pontas fracas salvas pela histerese!)
    pts_diag = [(310, 130), (390, 205), (475, 285), (565, 365), (665, 445), (755, 515)]
    _desenhar_trinca_variavel(bgr, pts_diag, contraste_centro=68, espessura=2,
                              contraste_ponta=22, fracao_ponta=0.22)

    # 2) Trinca longitudinal na área iluminada (esquerda) com pequena lacuna (religada pelo fecho direcional!)
    _desenhar_trinca_variavel(bgr, [(335, 260), (328, 345), (322, 412)], contraste_centro=66, espessura=2)
    _desenhar_trinca_variavel(bgr, [(322, 416), (314, 495), (308, 565)], contraste_centro=66, espessura=2)

    # 3) Trinca dentro da área de sombra (escuro sobre escuro -> Otsu global falha, limiar variável pega!)
    _desenhar_trinca_variavel(bgr, [(690, 145), (705, 225), (722, 305)], contraste_centro=64, espessura=2)

    # Riscos fracos isolados (contraste ~18: entram em fraca_t mas NÃO em forte_t -> histerese descarta!)
    for (x0, y0, x1, y1) in [
        (410, 120, 470, 132),
        (230, 460, 290, 475),
        (485, 520, 550, 528),
        (590, 195, 645, 210),
        (425, 385, 475, 402),
    ]:
        _desenhar_trinca_variavel(bgr, [(x0, y0), (x1, y1)], contraste_centro=19, espessura=2)

    return bgr, roi


def _gerar_buraco_faixa() -> Tuple[np.ndarray, np.ndarray]:
    info = CENARIOS_POR_ID["didatico_buraco_faixa"]
    bgr, roi = _base_estrada_rgb(640, 1024, info.roi_poligono)
    h, w = bgr.shape[:2]

    # Faixas brancas laterais pintadas DENTRO da ROI (acendem em cand_b como faixas estreitas,
    # e são APAGADAS pela ABERTURA de tamanho 8 na Etapa 4!)
    cv2.line(bgr, (760, 10), (925, 630), (208, 208, 208), 3, cv2.LINE_AA)
    cv2.line(bgr, (260, 10), (95, 630), (208, 208, 208), 3, cv2.LINE_AA)

    # Buraco principal grande com MIOLO LISO (borda áspera acende em cand_b;
    # abertura preserva; preenchimento de buracos fecha o interior liso!)
    _desenhar_buraco_sintetico(bgr, cx=565, cy=360, rx=104, ry=78, centro_liso=True)

    # Faixa clara áspera muito alongada (sobrevive textura e abertura, mas rejeitada por ASPECTO/CIRCULARIDADE!)
    y, x = np.indices((h, w), dtype=np.int32)
    bx, by = x >> 1, y >> 1
    faixa_aspera = (x >= 285) & (x <= 325) & (y >= 150) & (y <= 550)
    tex = ((bx * 31 + by * 47 + (bx * by) * 11) % 130) - 55
    for c in range(3):
        bgr[..., c] = np.where(faixa_aspera, np.clip(175 + tex, 125, 245), bgr[..., c]).astype(np.uint8)

    # Pequena aspereza isolada (sobrevive textura, mas área < 125 px -> rejeitada na Etapa 5!)
    alvo_peq = ((x - 425) ** 2 + (y - 180) ** 2) <= 10 ** 2
    tex_p = ((bx * 29 + by * 43) % 110) - 40
    for c in range(3):
        bgr[..., c] = np.where(alvo_peq, np.clip(165 + tex_p, 115, 235), bgr[..., c]).astype(np.uint8)

    return bgr, roi


def _gerar_completo() -> Tuple[np.ndarray, np.ndarray]:
    info = CENARIOS_POR_ID["didatico_completo"]
    bgr, roi = _base_estrada_rgb(640, 1024, info.roi_poligono)

    # 1) Trinca longitudinal (vertical: h >= 2w)
    _desenhar_trinca_variavel(bgr, [(280, 150), (276, 250), (272, 360), (268, 470)],
                              contraste_centro=56, espessura=2)

    # 2) Trinca transversal (horizontal: w >= 2h)
    _desenhar_trinca_variavel(bgr, [(360, 145), (460, 149), (570, 147), (665, 151)],
                              contraste_centro=56, espessura=2)

    # 3) Trinca diagonal (separada da rede jacaré)
    _desenhar_trinca_variavel(bgr, [(380, 205), (430, 255), (480, 305), (525, 350)],
                              contraste_centro=56, espessura=2)

    # 4) Rede em Couro de Jacaré (contraste suficiente para trinca forte sem estourar textura de buraco!)
    rede = [
        [(330, 440), (420, 452), (510, 445), (590, 460)],
        [(340, 520), (430, 532), (520, 524), (605, 536)],
        [(360, 585), (450, 592), (545, 586)],
        [(370, 420), (366, 515), (376, 595)],
        [(445, 425), (440, 528), (450, 600)],
        [(525, 425), (520, 520), (530, 595)],
        [(405, 450), (482, 528)],
        [(485, 448), (410, 530)],
        [(440, 530), (525, 586)],
    ]
    for seg in rede:
        _desenhar_trinca_variavel(bgr, seg, contraste_centro=46, espessura=2)

    # 5) Buraco áspero na faixa direita (único componente compacto)
    _desenhar_buraco_sintetico(bgr, cx=725, cy=355, rx=84, ry=64, centro_liso=False)

    return bgr, roi


def carregar_cenario(cenario_id: str) -> Tuple[np.ndarray, np.ndarray, CenarioInfo]:
    """Retorna (imagem_bgr, roi_bool, CenarioInfo) para um cenário embutido."""
    if cenario_id not in CENARIOS_POR_ID:
        raise KeyError(f"Cenário desconhecido: {cenario_id}")
    info = CENARIOS_POR_ID[cenario_id]
    if cenario_id == "rodovia_dnit_01":
        bgr = cv2.imread(str(PASTA_EXEMPLOS / "rodovia_dnit_01.jpg"))
        roi = poligono_para_mascara(bgr.shape[:2], list(info.roi_poligono))
        return bgr, roi, info
    if cenario_id == "rodovia_df251":
        bgr = cv2.imread(str(PASTA_EXEMPLOS / "rodovia_df251.jpg"))
        roi = poligono_para_mascara(bgr.shape[:2], list(info.roi_poligono))
        return bgr, roi, info
    if cenario_id == "didatico_sombra_trinca":
        bgr, roi = _gerar_sombra_trinca()
        return bgr, roi, info
    if cenario_id == "didatico_buraco_faixa":
        bgr, roi = _gerar_buraco_faixa()
        return bgr, roi, info
    if cenario_id == "didatico_completo":
        bgr, roi = _gerar_completo()
        return bgr, roi, info
    raise KeyError(cenario_id)


def listar_cenarios() -> List[dict]:
    return [
        {
            "id": c.id,
            "titulo": c.titulo,
            "categoria": c.categoria,
            "subtitulo": c.subtitulo,
            "descricao": c.descricao,
            "reduzir_padrao": c.reduzir_padrao,
            "roi_poligono": [list(pt) for pt in c.roi_poligono],
        }
        for c in CENARIOS
    ]
