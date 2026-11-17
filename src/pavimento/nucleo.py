"""Núcleo determinístico de detecção e quantificação de danos em pavimentos.

Regras de projeto (diferencial do grupo: quantificação, velocidade, determinismo):
  * Somente aritmética INTEIRA (sem float, sem FFT, sem aleatoriedade) — o mesmo
    a etapa de buracos em Python evolui independentemente do firmware C.
  * Bordas tratadas sempre da mesma forma (janela recortada / padding com zeros).
  * Componentes conexos ordenados pelo primeiro pixel em varredura raster.

Referências — material do Prof. Lucas Cambuim (pasta materiais_PI_2026.2/):
  [4_3]  "4_3 - melhorias em imagens e filtragem.pdf"
  [6_1]  "6_1 - processamento de imagens morfologicas.pdf"
  [7_1S] "7_1 - segmentacao.pdf"
  [7_1C] "7_1 - processamento de imagens em cores.pdf"
  [9_2]  "9_2 - extracao de caracteristicas.pdf"
  [2_1], [2_2] "fundamentos da imagem digital" partes 1 e 2
  [GW]   Gonzalez & Woods, Processamento Digital de Imagens, 3ª ed. — usado
         SOMENTE onde o slide não cobre (marcado como "complemento").
"""
from __future__ import annotations

import numpy as np
from math import isqrt

from .parametros import Parametros

# --------------------------------------------------------------------------- #
# 1. Aquisição / representação                                                #
# --------------------------------------------------------------------------- #

def intensidade_hsi(rgb: np.ndarray) -> np.ndarray:
    """I = (R+G+B)/3 — componente intensidade do modelo HSI [7_1C slides 60-73].
    Divisão inteira (piso). Entrada RGB ou BGR (a soma é a mesma)."""
    s = rgb[..., 0].astype(np.int32) + rgb[..., 1] + rgb[..., 2]
    return (s // 3).astype(np.uint8)


def reduzir_2x(img: np.ndarray) -> np.ndarray:
    """Reamostragem 2x por média de blocos 2x2 com arredondamento: (a+b+c+d+2)>>2.
    Reduz resolução espacial [2_1 slides 49-70] e o custo 4x (velocidade)."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    a = img[:h, :w].astype(np.int32)
    s = a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]
    return ((s + 2) >> 2).astype(np.uint8)


def reduzir_mascara_2x(m: np.ndarray, regra: str = "maioria") -> np.ndarray:
    """Máscara binária 2x menor: 'maioria' (>=2 de 4) para ROI, 'qualquer' para rótulos."""
    h, w = m.shape[0] // 2 * 2, m.shape[1] // 2 * 2
    a = (m[:h, :w] > 0).astype(np.int32)
    s = a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]
    return s >= 2 if regra == "maioria" else s >= 1


# --------------------------------------------------------------------------- #
# 2. Filtragem espacial: estatísticas locais com filtro de caixa              #
# --------------------------------------------------------------------------- #

def _soma_janela_1d(a: np.ndarray, r: int, eixo: int) -> np.ndarray:
    """Soma em janela [i-r, i+r] recortada nas bordas (somas acumuladas)."""
    n = a.shape[eixo]
    c = np.cumsum(a, axis=eixo, dtype=np.int64)
    zero_shape = list(a.shape); zero_shape[eixo] = 1
    c = np.concatenate([np.zeros(zero_shape, np.int64), c], axis=eixo)
    i = np.arange(n)
    hi, lo = np.minimum(i + r + 1, n), np.maximum(i - r, 0)
    return np.take(c, hi, axis=eixo) - np.take(c, lo, axis=eixo)


def estatisticas_locais(g: np.ndarray, r: int):
    """Média m_xy e variância s²_xy na vizinhança S_xy (2r+1)² — filtro de caixa
    [4_3 slides 33-36]; m_xy e s_xy como em [7_1S slides 98-99].
    Bordas: janela recortada à imagem (estratégia de limites [2_2 slides 60-62]).
      m   = floor(S1 / n)
      s²  = floor((n*S2 - S1²) / n²)
    """
    h, w = g.shape
    a = g.astype(np.int64)
    s1 = _soma_janela_1d(_soma_janela_1d(a, r, 1), r, 0)
    s2 = _soma_janela_1d(_soma_janela_1d(a * a, r, 1), r, 0)
    ny = np.minimum(np.arange(h) + r + 1, h) - np.maximum(np.arange(h) - r, 0)
    nx = np.minimum(np.arange(w) + r + 1, w) - np.maximum(np.arange(w) - r, 0)
    n = ny[:, None] * nx[None, :]
    media = s1 // n
    var = (n * s2 - s1 * s1) // (n * n)
    return media.astype(np.int32), var.astype(np.int32)


def raiz_inteira(v: np.ndarray) -> np.ndarray:
    """floor(sqrt(v)) exato para inteiros (igual ao isqrt do firmware)."""
    return np.fromiter((isqrt(int(x)) for x in v.flat), dtype=np.int32,
                       count=v.size).reshape(v.shape)


# --------------------------------------------------------------------------- #
# 3. Limiarização                                                              #
# --------------------------------------------------------------------------- #

def mediana_mad(valores: np.ndarray, nbins: int = 256):
    """Mediana inferior e MAD (mediana dos desvios absolutos) via histograma
    [4_2 slides 2-9; sl.7 medidas estatísticas do histograma]. Determinístico e O(n)."""
    n = valores.size
    if n == 0:
        return 0, 0
    h = np.bincount(valores, minlength=nbins)
    med = int(np.searchsorted(np.cumsum(h), (n + 1) // 2))
    d = np.abs(valores.astype(np.int32) - med)
    hd = np.bincount(d, minlength=nbins)
    mad = int(np.searchsorted(np.cumsum(hd), (n + 1) // 2))
    return med, mad


# --------------------------------------------------------------------------- #
# 4. Morfologia matemática binária [6_1]                                      #
# --------------------------------------------------------------------------- #

def otsu_inteiro(valores: np.ndarray) -> int:
    """Otsu [7_1S, limiarização global], sem normalização em ponto flutuante.

    Maximiza (N*S0 - n0*S)^2 / (n0*(N-n0)) por produtos cruzados
    com inteiros Python (sem overflow). Empates escolhem o menor limiar.
    Histograma vazio retorna 0; constante retorna a única intensidade.
    """
    hist = np.bincount(valores.astype(np.int64), minlength=256)
    total = int(hist.sum())
    soma = sum(i * int(n) for i, n in enumerate(hist))
    n0 = s0 = melhor_num = 0
    melhor_den = 1
    limiar = int(np.flatnonzero(hist)[0]) if total else 0
    for t, n in enumerate(hist):
        n0 += int(n)
        s0 += t * int(n)
        if n0 == 0 or n0 == total:
            continue
        num = (total * s0 - n0 * soma) ** 2
        den = n0 * (total - n0)
        if num * melhor_den > melhor_num * den:
            melhor_num, melhor_den, limiar = num, den, t
    return limiar


def histerese_componentes(forte: np.ndarray, fraca: np.ndarray) -> np.ndarray:
    """Regiões fracas 8-conexas contendo sementes fortes [7_2, 16–21;
    6_1, 67–73]. Equivale a crescer sementes sob o predicado da região fraca.
    """
    lab, n = rotular(fraca)
    manter = np.zeros(n + 1, dtype=bool)
    manter[lab[forte & fraca]] = True
    manter[0] = False
    return manter[lab]


def _desloc(m, dy, dx):
    """Translação (A)_z com preenchimento por zeros [6_1 slides 13, 25]."""
    h, w = m.shape
    out = np.zeros_like(m)
    if abs(dy) >= h or abs(dx) >= w:
        return out
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = m[ys, xs]
    return out


def dilatar(m: np.ndarray, k: int) -> np.ndarray:
    """Dilatação por EE quadrado k×k (separável) [6_1 slides 28-33]."""
    if k <= 1:
        return m.copy()
    r = k // 2
    t = m.copy()
    for d in range(1, r + 1):
        t = t | _desloc(m, 0, d) | _desloc(m, 0, -d)
    out = t.copy()
    for d in range(1, r + 1):
        out = out | _desloc(t, d, 0) | _desloc(t, -d, 0)
    return out


def erodir(m: np.ndarray, k: int) -> np.ndarray:
    """Erosão por EE quadrado k×k com padding de zeros [6_1 slides 22-27]."""
    if k <= 1:
        return m.copy()
    r = k // 2
    t = m.copy()
    for d in range(1, r + 1):
        t = t & _desloc(m, 0, d) & _desloc(m, 0, -d)
    out = t.copy()
    for d in range(1, r + 1):
        out = out & _desloc(t, d, 0) & _desloc(t, -d, 0)
    return out


def abertura(m, k):
    """A∘B = (A⊖B)⊕B — remove estruturas mais estreitas que B [6_1 slides 36-39]."""
    return dilatar(erodir(m, k), k)


def fechamento(m, k):
    """A•B = (A⊕B)⊖B — funde separações estreitas [6_1 slides 40-42]."""
    return erodir(dilatar(m, k), k)


def morfologia_direcional(m, k, fechar=False):
    """União de aberturas/fechamentos por linhas H/V/diagonais [6_1].

    Uma abertura quadrada apagaria a trinca fina. EEs ímpares lineares
    preservam segmentos alinhados e o fechamento liga lacunas na sua direção.
    """
    if k <= 1:
        return m.copy()
    out = np.zeros_like(m)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        pontos = tuple((d * dy, d * dx) for d in range(-(k // 2), k // 2 + 1))

        def operar(a, dilatacao):
            r = a.copy()
            for y, x in pontos:
                if dilatacao:
                    r |= _desloc(a, y, x)
                else:
                    r &= _desloc(a, y, x)
            return r

        out |= operar(operar(m, fechar), not fechar)
    return out


def borda(m):
    """β(A) = A − (A ⊖ B), B 3x3 [6_1 slides 59-60]; |β(A)| = perímetro [9_2 slide 59]."""
    return m & ~erodir(m, 3)


def preencher_buracos(m: np.ndarray) -> np.ndarray:
    """Preenchimento de buracos [6_1 slides 61-66]. Em vez de semear cada buraco,
    o fundo conectado à borda da imagem é obtido por dilatação condicional
    X_k = (X_{k-1} ⊕ B) ∩ A^c (B = cruz, conectividade-4) a partir da moldura;
    o que sobra do complemento são buracos. (Variante automática: [GW] 9.5.9.)"""
    fundo = ~m
    x = np.zeros_like(m)
    x[0, :], x[-1, :], x[:, 0], x[:, -1] = fundo[0, :], fundo[-1, :], fundo[:, 0], fundo[:, -1]
    while True:
        xn = (x | _desloc(x, 1, 0) | _desloc(x, -1, 0) | _desloc(x, 0, 1) | _desloc(x, 0, -1)) & fundo
        if np.array_equal(xn, x):
            break
        x = xn
    return m | (fundo & ~x)


def rotular(m: np.ndarray):
    """Componentes conexos com adjacência-8 [2_2 slides 6-16; 6_1 slides 67-73].
    Rótulos 1..N ordenados pelo primeiro pixel em varredura raster (igual ao firmware).
    Implementação: OpenCV só para a rotulagem (velocidade); a ordem é normalizada."""
    import cv2
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return lab.astype(np.int32), 0
    flat = lab.ravel()
    primeiro = np.full(n, flat.size, np.int64)
    idx = np.flatnonzero(flat)
    np.minimum.at(primeiro, flat[idx], idx)
    ordem = np.argsort(primeiro[1:], kind="stable") + 1
    mapa = np.zeros(n, np.int32)
    mapa[ordem] = np.arange(1, n, dtype=np.int32)
    return mapa[lab], n - 1


def _vizinhos8(e):
    p = np.pad(e, 1)
    return [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]  # P2..P9


def afinar(m: np.ndarray) -> np.ndarray:
    """Esqueletização de Zhang & Suen (1984) — COMPLEMENTO: não está nos slides
    ([GW] 9.5.7 "afinamento"). Usada para medir o COMPRIMENTO da trinca, base da
    quantificação. Cada subiteração decide sobre uma cópia (paralelo) — o firmware
    faz exatamente o mesmo."""
    img = m.astype(np.uint8)
    while True:
        mudou = False
        for passo in (0, 1):
            P = _vizinhos8(img)
            B = sum(p.astype(np.int32) for p in P)
            seq = P + [P[0]]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            P2, P3, P4, P5, P6, P7, P8, P9 = P
            if passo == 0:
                c = (P2 & P4 & P6) | (P4 & P6 & P8)
            else:
                c = (P2 & P4 & P8) | (P2 & P6 & P8)
            rem = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c == 0)
            if rem.any():
                img[rem] = 0
                mudou = True
        if not mudou:
            return img.astype(bool)


def contar_vizinhos8(e: np.ndarray) -> np.ndarray:
    return sum(p.astype(np.int32) for p in _vizinhos8(e.astype(np.uint8)))


# --------------------------------------------------------------------------- #
# 5. Pipeline                                                                  #
# --------------------------------------------------------------------------- #

TIPOS = ("longitudinal", "transversal", "diagonal", "couro_de_jacare")


def analisar(g: np.ndarray, roi: np.ndarray, p: Parametros = Parametros()) -> dict:
    """g: uint8 (H,W) intensidade; roi: bool (H,W) região da pista.
    Retorna máscaras, regiões e quantificação (tudo inteiro)."""
    roi = roi.astype(bool)
    n_roi = int(roi.sum())

    # (a) estatísticas locais [4_3 33-36], [7_1S 98-99]
    media, var = estatisticas_locais(g, p.raio)
    desvio = raiz_inteira(var)
    # Janela própria da textura de buracos: trincas pedem janela pequena; buracos, grande.
    if p.raio_buraco == p.raio:
        desvio_b = desvio
    else:
        _, var_b = estatisticas_locais(g, p.raio_buraco)
        desvio_b = raiz_inteira(var_b)

    # (b) BURACOS: textura E escurecimento nas sementes; crescimento conectado
    # pelo escurecimento recupera interiores lisos [7_2, 16–21]. A média local
    # [4_3, 31–36; 7_1S, 96–100] reduz ruído sem exigir textura em toda a região.
    med, mad = mediana_mad(desvio_b[roi].astype(np.int64))
    t_tex = med + (p.buraco_k4 * mad) // 4
    g_med, _ = mediana_mad(g[roi].astype(np.int64))
    t_otsu = otsu_inteiro(g[roi])
    t_fraco = g_med - p.buraco_margem_fraca
    t_forte = g_med - max(p.buraco_margem_forte, p.buraco_margem_fraca)
    if p.buraco_usar_otsu:
        t_fraco, t_forte = min(t_fraco, t_otsu), min(t_forte, t_otsu)
    # Exigir também intensidade escura evita o halo claro criado pela média
    # junto ao contorno, especialmente em buracos menores que a janela local.
    escura = (g.astype(np.int32) <= t_fraco) & roi
    fraca = (media <= t_fraco) & escura
    if p.buraco_modo == 1:
        # Modo textura [7_1 Seg 19–24; 4_2 2–9]: limiar global mediana + k·MAD sobre o
        # desvio local; abertura remove trincas finas [6_1 36–39], fechamento religa
        # [6_1 40–42] e preenchimento fecha o interior [6_1 61–66]. Sem float.
        cand_b = (desvio_b > t_tex) & roi
        fraca = abertura(cand_b, p.buraco_abertura) & roi
        b = fechamento(fraca, p.buraco_fecho) & roi
        b = preencher_buracos(b) & roi
    else:
        cand_b = (desvio_b > t_tex) & (media <= t_forte) & fraca
        # Abrir a região fraca antes de crescer evita pontes finas [6_1, 36–39].
        fraca = abertura(fraca, p.buraco_abertura) & roi
        b = histerese_componentes(cand_b, fraca)
        b = fechamento(b, p.buraco_fecho) & roi
        b = preencher_buracos(b) & roi
    lab_b, nb = rotular(b)
    buracos = np.zeros_like(b)
    reg_b = []
    faixa_borda = (roi & ~erodir(roi, 2 * p.buraco_borda_margem + 1)
                   if p.buraco_borda_margem > 0 else np.zeros_like(roi))
    if nb:
        area = np.bincount(lab_b.ravel(), minlength=nb + 1)
        per = np.bincount(lab_b[borda(b)], minlength=nb + 1)  # perímetro [9_2 59]
        soma_g = np.zeros(nb + 1, dtype=np.int64)
        np.add.at(soma_g, lab_b.ravel(), g.ravel().astype(np.int64))
        area_escura = np.bincount(lab_b[escura], minlength=nb + 1)
        toca_borda = np.bincount(lab_b[faixa_borda], minlength=nb + 1)
        ys, xs = np.nonzero(b)
        ls = lab_b[ys, xs]
        x0 = np.full(nb + 1, g.shape[1], dtype=np.int64)
        y0 = np.full(nb + 1, g.shape[0], dtype=np.int64)
        x1 = np.full(nb + 1, -1, dtype=np.int64)
        y1 = np.full(nb + 1, -1, dtype=np.int64)
        np.minimum.at(x0, ls, xs); np.maximum.at(x1, ls, xs)
        np.minimum.at(y0, ls, ys); np.maximum.at(y1, ls, ys)
        for i in range(1, nb + 1):
            a_i, p_i = int(area[i]), int(per[i])
            circ = (12566 * a_i) // max(p_i * p_i, 1)  # 4πA/P² x1000 [9_2 61]
            # intensidade média da região [9_2 55 (descritores de região)] vs. asfalto:
            # buraco é mais escuro => a·g_med - Σg >= a·escuro_min  (sem divisão)
            escuro_ok = a_i * g_med - int(soma_g[i]) >= a_i * p.buraco_escuro_min
            w, h = int(x1[i] - x0[i] + 1), int(y1[i] - y0[i] + 1)
            # Área e extensões da região [9_2, 55–58, 66–68]. Extensão usa
            # bbox, não a solidez relativa ao casco convexo.
            extensao = (1000 * a_i) // (w * h)
            aspecto = (1000 * max(w, h)) // min(w, h)
            fracao_escura = (1000 * int(area_escura[i])) // a_i
            forma_ok = (extensao >= p.buraco_extensao_min_1000
                        and aspecto <= p.buraco_aspecto_max_1000
                        and fracao_escura >= p.buraco_fracao_escura_min_1000
                        and toca_borda[i] == 0)
            if a_i >= p.buraco_area_min and circ >= p.buraco_circ_min_1000 and escuro_ok and forma_ok:
                buracos |= lab_b == i
                reg_b.append({"area": a_i, "perimetro": p_i, "circularidade_1000": circ,
                              "escurecimento_x_area": a_i * g_med - int(soma_g[i]),
                              "bbox": (int(x0[i]), int(y0[i]), w, h),
                              "extensao_1000": extensao, "aspecto_1000": aspecto,
                              "fracao_escura_1000": fracao_escura})

    # (c) O limiar único selecionava só centros escuros: máscara fina/dispersa,
    # lacunas em trechos claros e resíduos da textura. Dois limiares [7_1/7_2]
    # e reconstrução geodésica [6_1] recuperam largura sem dilatação irrestrita.
    permitido = roi & ~buracos

    def candidato(a16, b16):
        d = 16 * (media - g.astype(np.int32)) - b16
        return permitido & (d > 0) & (d.astype(np.int64) ** 2 > a16 ** 2 * var.astype(np.int64))

    forte_t = candidato(p.trinca_a16, p.trinca_b16)
    fraca_t = candidato(min(p.trinca_a16, p.trinca_fraca_a16),
                       min(p.trinca_b16, p.trinca_fraca_b16)) | forte_t
    # As pontes direcionais pertencem ao domínio de crescimento, nunca são
    # sementes novas. A ROI é reaplicada antes da reconstrução (sem atalhos).
    fraca_t = morfologia_direcional(fraca_t, p.trinca_fecho, fechar=True) & permitido
    suporte_t = morfologia_direcional(fraca_t, p.trinca_abertura_linha)
    # Componentes com semente equivalem à dilatação condicional até estabilidade.
    # A abertura não pode vetar sementes: uma linha digital de inclinação 1:2
    # não contém nenhum EE reto de 3 px. O suporte é verificado por componente,
    # aceitando também comprimento conectado suficiente, em qualquer ângulo.
    cand_t = histerese_componentes(forte_t, fraca_t)
    lab_t, nt = rotular(cand_t)
    trincas = np.zeros_like(cand_t)
    esq_total = np.zeros_like(cand_t)
    reg_t = []
    if nt:
        area = np.bincount(lab_t.ravel(), minlength=nt + 1)
        suporte = np.bincount(lab_t[suporte_t & cand_t], minlength=nt + 1)
        esq = afinar(cand_t)
        comp = np.bincount(lab_t[esq], minlength=nt + 1)
        bif = np.bincount(lab_t[esq & (contar_vizinhos8(esq) >= 3)], minlength=nt + 1)
        ys, xs = np.nonzero(lab_t)
        ls = lab_t[ys, xs]
        x0 = np.full(nt + 1, 1 << 30); x1 = np.full(nt + 1, -1)
        y0 = np.full(nt + 1, 1 << 30); y1 = np.full(nt + 1, -1)
        np.minimum.at(x0, ls, xs); np.maximum.at(x1, ls, xs)
        np.minimum.at(y0, ls, ys); np.maximum.at(y1, ls, ys)
        for i in range(1, nt + 1):
            a_i, c_i = int(area[i]), int(comp[i])
            if a_i < p.trinca_area_min or c_i < p.trinca_comp_min:
                continue
            if suporte[i] == 0 and c_i < p.trinca_abertura_linha:
                continue
            larg16 = (16 * a_i) // c_i                     # largura = área/comprimento
            if larg16 > p.trinca_larg_max16:
                continue
            if 16 * c_i * c_i < p.trinca_along_min16 * a_i:
                continue
            w, h = int(x1[i] - x0[i] + 1), int(y1[i] - y0[i] + 1)
            # tipo pelas projeções horizontal/vertical (extensão do objeto) [9_2 66]
            if 100 * int(bif[i]) >= p.jacare_bif_100px * c_i and c_i >= p.jacare_comp_min:
                tipo = 3
            elif h >= 2 * w:
                tipo = 0
            elif w >= 2 * h:
                tipo = 1
            else:
                tipo = 2
            sel = lab_t == i
            trincas |= sel
            esq_total |= esq & sel
            reg_t.append({"area": a_i, "comprimento": c_i, "largura16": larg16,
                          "bifurcacoes": int(bif[i]), "bbox": (int(x0[i]), int(y0[i]), w, h),
                          "tipo": TIPOS[tipo]})

    q = quantificar(reg_t, reg_b, n_roi, p)
    return {"media": media, "desvio": desvio, "cand_trinca": cand_t, "cand_buraco": cand_b,
            "trincas": trincas, "esqueleto": esq_total, "buracos": buracos,
            "regioes_trinca": reg_t, "regioes_buraco": reg_b, "limiar_textura": int(t_tex),
            "limiar_escuro": int(t_fraco), "limiar_escuro_forte": int(t_forte),
            "limiar_otsu": int(t_otsu), "buraco_fraca": fraca,
            "trinca_forte": forte_t, "trinca_fraca": fraca_t, "quant": q}


def indice_dano(comp_pond: int, area_b: int, n_roi: int,
                peso_trinca: int = 40, peso_buraco: int = 25) -> int:
    """Índice 0..1000 (heurístico, inspirado na lógica densidade x severidade do PCI —
    ASTM D6433; NÃO é o PCI oficial). Pesos explícitos separam referência/previsão."""
    n_roi = max(n_roi, 1)
    pct_b_100 = (10000 * area_b) // n_roi
    return min(1000, (peso_trinca * 1000 * comp_pond) // n_roi + (peso_buraco * pct_b_100) // 10)


def severidade(indice: int, p: Parametros) -> str:
    if indice < p.sev_min_1000:
        return "nenhum"
    return "baixa" if indice < p.sev_media_1000 else "media" if indice < p.sev_alta_1000 else "alta"


def quantificar(reg_t, reg_b, n_roi, p: Parametros = Parametros()) -> dict:
    """Quantificação inteira por imagem (unidades: pixels na resolução de processamento).
    Descritores de região: área [9_2 56], perímetro [9_2 59], comprimento [9_2 49]."""
    n_roi = max(n_roi, 1)
    comp = sum(r["comprimento"] for r in reg_t)
    area_t = sum(r["area"] for r in reg_t)
    area_b = sum(r["area"] for r in reg_b)
    tipos = {t: 0 for t in TIPOS}
    for r in reg_t:
        tipos[r["tipo"]] += 1
    dens_1000 = (1000 * comp) // n_roi                 # px de trinca por 1000 px de pista
    pct_b_100 = (10000 * area_b) // n_roi              # % da pista com buraco (x100)
    # Índice de dano: couro de jacaré pesa 2x no comprimento
    comp_pond = comp + sum(r["comprimento"] for r in reg_t if r["tipo"] == "couro_de_jacare")
    indice = indice_dano(comp_pond, area_b, n_roi, p.dano_peso_trinca, p.dano_peso_buraco)
    sev = severidade(indice, p)
    return {"n_trincas": len(reg_t), "comprimento_trincas_px": comp, "area_trincas_px": area_t,
            "largura_media_trincas_16": (16 * area_t) // comp if comp else 0,
            "densidade_trincas_1000": dens_1000, "tipos_trinca": tipos,
            "n_buracos": len(reg_b), "area_buracos_px": area_b,
            "area_buracos_pct_100": pct_b_100, "indice_dano_1000": indice, "severidade": sev}
