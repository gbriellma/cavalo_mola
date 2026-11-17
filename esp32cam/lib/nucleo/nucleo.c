/*
 * nucleo.c — porta exata de src/pavimento/nucleo.py (ver docstrings lá para a teoria).
 * Somente inteiros. Referências às aulas do Prof. Lucas Cambuim (materiais_PI_2026.2):
 *   [4_3] filtro de caixa sl.33-36 | [7_1S] média/desvio locais sl.98-99, limiar variável
 *   sl.97-100, textura por desvio padrão sl.19-24 | [6_1] erosão sl.22-27, dilatação sl.28-33,
 *   abertura/fechamento sl.36-42, borda sl.59-60, preenchimento sl.61-66, componentes sl.67-73
 *   | [2_2] adjacência-8 sl.6-16, limites da imagem sl.60-62 | [9_2] área sl.56, perímetro
 *   sl.59, circularidade sl.61, projeções sl.66 | complemento: Zhang-Suen ([GW] 9.5.7).
 */
#include "nucleo.h"
#include <string.h>

const char *PAV_TIPOS[4] = {"longitudinal", "transversal", "diagonal", "couro_de_jacare"};
const char *PAV_SEVERIDADE[4] = {"nenhum", "baixa", "media", "alta"};

#define ERR_ROTULOS 2
#define PAV_LARGURA_MAX 2048

static uint32_t (*g_relogio_us)(void) = 0;
void pav_definir_relogio(uint32_t (*f)(void)) { g_relogio_us = f; }
#define MARCA(i) do { if (g_relogio_us) { uint32_t _t = g_relogio_us(); q->t_etapa_us[i] = _t - t_ant; t_ant = _t; } } while (0)

typedef struct {
    uint8_t *media, *desvio, *cand_t, *bur, *tri, *esq, *tmpA, *tmpB;
    uint16_t *var, *lab;
    uint32_t *pilha;
    uint32_t *st_area, *st_a2, *st_a3;
    uint16_t *st_x0, *st_x1, *st_y0, *st_y1;
    uint32_t cap_rotulos;
} trab_t;

size_t pav_bytes_trabalho(int w, int h) {
    size_t n = (size_t)w * h;
    return 16 * n + 64;
}

static void particionar(trab_t *t, void *mem, int w, int h) {
    size_t n = (size_t)w * h;
    uint8_t *m = (uint8_t *)mem;
    /* região R1 (3n): media+var; depois reaproveitada para os rótulos (2n) */
    t->var = (uint16_t *)m;           /* 2n */
    t->media = m + 2 * n;             /* n  */
    t->lab = (uint16_t *)m;           /* reaproveita var depois que não é mais usada */
    m += 3 * n;
    t->desvio = m; m += n;
    t->cand_t = m; m += n;
    t->bur = m; m += n;
    t->tri = m; m += n;
    t->esq = m; m += n;
    t->tmpA = m; m += n;
    t->tmpB = m; m += n;
    t->pilha = (uint32_t *)m; m += 4 * n;
    /* estatísticas por rótulo: 20 bytes/rótulo dentro de 2n bytes */
    t->cap_rotulos = (uint32_t)((2 * n) / 20);
    t->st_area = (uint32_t *)m;
    t->st_a2 = t->st_area + t->cap_rotulos;
    t->st_a3 = t->st_a2 + t->cap_rotulos;
    t->st_x0 = (uint16_t *)(t->st_a3 + t->cap_rotulos);
    t->st_x1 = t->st_x0 + t->cap_rotulos;
    t->st_y0 = t->st_x1 + t->cap_rotulos;
    t->st_y1 = t->st_y0 + t->cap_rotulos;
}

/* ---------- estatísticas locais: filtro de caixa com janela recortada ---------- */
static void estatisticas_locais(const uint8_t *g, int w, int h, int r, uint8_t *media,
                                uint16_t *var, uint32_t *col1, uint32_t *col2) {
    /* col1/col2: somas verticais por coluna (tamanho w). Janela vertical [y-r, y+r]∩[0,h). */
    memset(col1, 0, sizeof(uint32_t) * w);
    memset(col2, 0, sizeof(uint32_t) * w);
    for (int y = 0; y <= r && y < h; y++)
        for (int x = 0; x < w; x++) { uint32_t v = g[y * w + x]; col1[x] += v; col2[x] += v * v; }
    for (int y = 0; y < h; y++) {
        int ylo = y - r < 0 ? 0 : y - r, yhi = y + r + 1 > h ? h : y + r + 1;
        int64_t ny = yhi - ylo;
        /* soma horizontal recortada sobre as colunas acumuladas */
        int64_t s1 = 0, s2 = 0;
        int xhi0 = r + 1 > w ? w : r + 1;
        for (int x = 0; x < xhi0; x++) { s1 += col1[x]; s2 += col2[x]; }
        for (int x = 0; x < w; x++) {
            int xlo = x - r < 0 ? 0 : x - r, xhi = x + r + 1 > w ? w : x + r + 1;
            if (x > 0) {
                if (x + r < w) { s1 += col1[x + r]; s2 += col2[x + r]; }
                if (x - r - 1 >= 0) { s1 -= col1[x - r - 1]; s2 -= col2[x - r - 1]; }
            }
            uint32_t n = (uint32_t)(ny * (xhi - xlo));
            uint32_t m = (uint32_t)s1 / n;                       /* s1 <= 361*255: cabe em 32 bits */
            uint64_t num = (uint64_t)n * (uint64_t)s2 - (uint64_t)s1 * (uint64_t)s1;
            uint32_t nn = n * n;
            uint32_t v = num < 0xFFFFFFFFull ? (uint32_t)num / nn : (uint32_t)(num / nn);
            media[y * w + x] = (uint8_t)m;
            var[y * w + x] = (uint16_t)v;
        }
        /* desliza a janela vertical para y+1 */
        if (y + r + 1 < h)
            for (int x = 0; x < w; x++) { uint32_t v = g[(y + r + 1) * w + x]; col1[x] += v; col2[x] += v * v; }
        if (y - r >= 0)
            for (int x = 0; x < w; x++) { uint32_t v = g[(y - r) * w + x]; col1[x] -= v; col2[x] -= v * v; }
    }
}

uint32_t pav_isqrt32(uint32_t v);
static uint32_t isqrt32(uint32_t v) {
    uint32_t r = 0, b = 1u << 30;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return r;
}

/* mediana inferior e MAD via histograma de 256 classes */
static void mediana_mad(const uint8_t *v, const uint8_t *roi, size_t n, int *med, int *mad) {
    uint32_t hst[256]; memset(hst, 0, sizeof hst);
    uint32_t cnt = 0;
    for (size_t i = 0; i < n; i++) if (roi[i]) { hst[v[i]]++; cnt++; }
    if (!cnt) { *med = 0; *mad = 0; return; }
    uint32_t alvo = (cnt + 1) / 2, acc = 0; int m = 0;
    for (m = 0; m < 256; m++) { acc += hst[m]; if (acc >= alvo) break; }
    memset(hst, 0, sizeof hst);
    for (size_t i = 0; i < n; i++) if (roi[i]) { int d = (int)v[i] - m; hst[d < 0 ? -d : d]++; }
    acc = 0; int a = 0;
    for (a = 0; a < 256; a++) { acc += hst[a]; if (acc >= alvo) break; }
    *med = m; *mad = a;
}

/* ---------- morfologia binária, EE quadrado k×k, padding com zeros ----------
   Implementação separável com CONTAGEM DESLIZANTE: para cada pixel conta quantos 1s
   há na janela [x-r, x+r] (fora da imagem conta como 0). Dilatação: conta > 0;
   erosão: conta == 2r+1 (qualquer posição fora da imagem já impede). Custo O(1)
   por pixel, independente de k — resultado idêntico à definição [6_1 sl.22-33]. */
void pav_morf(const uint8_t *src, uint8_t *dst, uint8_t *tmp, int w, int h, int k, int dilata);
static void morf(const uint8_t *src, uint8_t *dst, uint8_t *tmp, int w, int h, int k, int dilata) {
    size_t n = (size_t)w * h;
    if (k <= 1) { if (dst != src) memcpy(dst, src, n); return; }
    if (w > PAV_LARGURA_MAX) return;
    int r = k / 2, cheio = 2 * r + 1;
    for (int y = 0; y < h; y++) {
        const uint8_t *s = src + (size_t)y * w; uint8_t *t = tmp + (size_t)y * w;
        int c = 0;
        for (int x = 0; x <= r && x < w; x++) c += s[x] ? 1 : 0;
        for (int x = 0; x < w; x++) {
            t[x] = (uint8_t)(dilata ? (c > 0) : (c == cheio));
            if (x + r + 1 < w && s[x + r + 1]) c++;
            if (x - r >= 0 && s[x - r]) c--;
        }
    }
    /* passada vertical em ORDEM DE LINHAS (amigável ao cache da PSRAM na ESP32):
       cnt[x] = nº de 1s na coluna x dentro das linhas [y-r, y+r] */
    static int16_t cnt[PAV_LARGURA_MAX];
    for (int x = 0; x < w; x++) cnt[x] = 0;
    for (int y = 0; y <= r && y < h; y++) {
        const uint8_t *t = tmp + (size_t)y * w;
        for (int x = 0; x < w; x++) cnt[x] += t[x] ? 1 : 0;
    }
    for (int y = 0; y < h; y++) {
        uint8_t *d = dst + (size_t)y * w;
        if (dilata) for (int x = 0; x < w; x++) d[x] = (uint8_t)(cnt[x] > 0);
        else        for (int x = 0; x < w; x++) d[x] = (uint8_t)(cnt[x] == cheio);
        if (y + r + 1 < h) { const uint8_t *t = tmp + (size_t)(y + r + 1) * w; for (int x = 0; x < w; x++) cnt[x] += t[x] ? 1 : 0; }
        if (y - r >= 0)    { const uint8_t *t = tmp + (size_t)(y - r) * w;     for (int x = 0; x < w; x++) cnt[x] -= t[x] ? 1 : 0; }
    }
}

static void abertura(uint8_t *m, uint8_t *tmp, int w, int h, int k) { morf(m, m, tmp, w, h, k, 0); morf(m, m, tmp, w, h, k, 1); }
static void fechamento(uint8_t *m, uint8_t *tmp, int w, int h, int k) { morf(m, m, tmp, w, h, k, 1); morf(m, m, tmp, w, h, k, 0); }

/* preenchimento de buracos: fundo conectado (vizinhança-4) à moldura */
void pav_preencher(uint8_t *m, uint8_t *alc, uint32_t *pilha, int w, int h);
static void preencher(uint8_t *m, uint8_t *alc, uint32_t *pilha, int w, int h) {
    size_t n = (size_t)w * h, sp = 0;
    memset(alc, 0, n);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && !m[y * w + x]) {
                alc[y * w + x] = 1; pilha[sp++] = (uint32_t)(y * w + x);
            }
    while (sp) {
        uint32_t i = pilha[--sp]; int y = i / w, x = i % w;
        const int dy[4] = {-1, 1, 0, 0}, dx[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; k++) {
            int yy = y + dy[k], xx = x + dx[k];
            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
            uint32_t j = (uint32_t)(yy * w + xx);
            if (!m[j] && !alc[j]) { alc[j] = 1; pilha[sp++] = j; }
        }
    }
    for (size_t i = 0; i < n; i++) if (!m[i] && !alc[i]) m[i] = 1;
}

/* rotulagem 8-conexa; rótulos na ordem do primeiro pixel em varredura raster */
static int rotular(const uint8_t *m, uint16_t *lab, uint32_t *pilha, int w, int h, uint32_t cap, uint32_t *nlab) {
    size_t n = (size_t)w * h;
    memset(lab, 0, n * sizeof(uint16_t));
    uint32_t L = 0;
    for (size_t s = 0; s < n; s++) {
        if (!m[s] || lab[s]) continue;
        if (L + 1 >= cap || L + 1 >= 65535) return ERR_ROTULOS;
        L++;
        size_t sp = 0; lab[s] = (uint16_t)L; pilha[sp++] = (uint32_t)s;
        while (sp) {
            uint32_t i = pilha[--sp]; int y = i / w, x = i % w;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int yy = y + dy, xx = x + dx;
                    if ((!dy && !dx) || yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                    uint32_t j = (uint32_t)(yy * w + xx);
                    if (m[j] && !lab[j]) { lab[j] = (uint16_t)L; pilha[sp++] = j; }
                }
        }
    }
    *nlab = L;
    return 0;
}

static inline int px(const uint8_t *a, int w, int h, int y, int x) {
    return (y >= 0 && y < h && x >= 0 && x < w) ? a[y * w + x] : 0;
}

/* Zhang-Suen (complemento, [GW] 9.5.7); cada subiteração decide sobre uma cópia.
   Otimização: só visita a LISTA de pixels ainda acesos (lista em 'lst', capacidade n).
   A decisão de cada pixel depende só do estado no início da subiteração, então a
   ordem de visita não altera o resultado. */
static void afinar(uint8_t *img, uint8_t *rem, uint32_t *lst, int w, int h) {
    size_t n = (size_t)w * h, m = 0;
    (void)rem;
    for (size_t i = 0; i < n; i++) if (img[i]) lst[m++] = (uint32_t)i;
    for (;;) {
        int mudou = 0;
        for (int passo = 0; passo < 2; passo++) {
            size_t nrem = 0;
            /* marca: os índices a remover vão para o fim da lista (de trás para frente) */
            for (size_t k = 0; k < m; k++) {
                uint32_t i = lst[k]; int y = (int)(i / w), x = (int)(i % w);
                int P[9];
                P[0] = px(img, w, h, y - 1, x);     P[1] = px(img, w, h, y - 1, x + 1);
                P[2] = px(img, w, h, y, x + 1);     P[3] = px(img, w, h, y + 1, x + 1);
                P[4] = px(img, w, h, y + 1, x);     P[5] = px(img, w, h, y + 1, x - 1);
                P[6] = px(img, w, h, y, x - 1);     P[7] = px(img, w, h, y - 1, x - 1);
                P[8] = P[0];
                int B = 0, A = 0;
                for (int j = 0; j < 8; j++) { B += P[j]; if (!P[j] && P[j + 1]) A++; }
                int P2 = P[0], P4 = P[2], P6 = P[4], P8 = P[6];
                int c = passo == 0 ? ((P2 & P4 & P6) | (P4 & P6 & P8)) : ((P2 & P4 & P8) | (P2 & P6 & P8));
                if (B >= 2 && B <= 6 && A == 1 && c == 0) { lst[k] |= 0x80000000u; nrem++; }
            }
            if (nrem) {
                size_t j = 0;
                for (size_t k = 0; k < m; k++) {
                    if (lst[k] & 0x80000000u) img[lst[k] & 0x7FFFFFFFu] = 0;
                    else lst[j++] = lst[k];
                }
                m = j;
                mudou = 1;
            }
        }
        if (!mudou) return;
    }
}

static uint32_t fnv1a(const uint8_t *m, size_t n) {
    uint32_t hsh = 2166136261u;
    for (size_t i = 0; i < n; i++) { hsh ^= (uint8_t)(m[i] ? 1 : 0); hsh *= 16777619u; }
    return hsh;
}

int pav_analisar(const uint8_t *g, const uint8_t *roi, int w, int h, const pav_params_t *p,
                 void *trabalho, pav_quant_t *q, uint8_t *saida) {
    trab_t T; particionar(&T, trabalho, w, h);
    size_t n = (size_t)w * h;
    memset(q, 0, sizeof *q);
    uint32_t t_ant = g_relogio_us ? g_relogio_us() : 0;
    int64_t n_roi = 0;
    for (size_t i = 0; i < n; i++) n_roi += roi[i] ? 1 : 0;

    /* (a) estatísticas locais — col1/col2 usam a pilha como rascunho (2w uint32) */
    estatisticas_locais(g, w, h, p->raio, T.media, T.var, T.pilha, T.pilha + w);
    for (size_t i = 0; i < n; i++) T.desvio[i] = (uint8_t)isqrt32(T.var[i]);
    MARCA(0);  /* estatísticas locais (média, variância, desvio) */

    /* (c0) candidatos de trinca (limiar variável) — antes de liberar media/var */
    int64_t a2 = (int64_t)p->trinca_a16 * p->trinca_a16;
    for (size_t i = 0; i < n; i++) {
        int64_t d16 = 16 * ((int64_t)T.media[i] - g[i]) - p->trinca_b16;
        T.cand_t[i] = (roi[i] && d16 > 0 && d16 * d16 > a2 * (int64_t)T.var[i]) ? 1 : 0;
    }

    /* (b) buracos: textura = desvio local; limiar = mediana + k·MAD */
    int med, mad, g_med, lixo;
    mediana_mad(T.desvio, roi, n, &med, &mad);
    mediana_mad(g, roi, n, &g_med, &lixo);
    int t_tex = med + (p->buraco_k4 * mad) / 4;
    q->limiar_textura = t_tex;
    for (size_t i = 0; i < n; i++) T.bur[i] = (roi[i] && T.desvio[i] > t_tex) ? 1 : 0;
    abertura(T.bur, T.tmpA, w, h, p->buraco_abertura);
    fechamento(T.bur, T.tmpA, w, h, p->buraco_fecho);
    preencher(T.bur, T.tmpA, T.pilha, w, h);
    for (size_t i = 0; i < n; i++) T.bur[i] = T.bur[i] && roi[i];
    uint32_t nb = 0;
    if (rotular(T.bur, T.lab, T.pilha, w, h, T.cap_rotulos, &nb)) return ERR_ROTULOS;
    memset(T.st_area, 0, 4 * (nb + 1)); memset(T.st_a2, 0, 4 * (nb + 1)); memset(T.st_a3, 0, 4 * (nb + 1));
    morf(T.bur, T.tmpB, T.tmpA, w, h, 3, 0);                 /* erosão 3x3 -> borda */
    for (size_t i = 0; i < n; i++) {
        uint16_t L = T.lab[i]; if (!L) continue;
        T.st_area[L]++; T.st_a3[L] += g[i];
        if (!T.tmpB[i]) T.st_a2[L]++;                        /* pixel de borda */
    }
    memset(T.tmpB, 0, n);                                     /* tmpB = buracos aceitos (flag por rótulo) */
    for (uint32_t L = 1; L <= nb; L++) {
        int64_t a = T.st_area[L], per = T.st_a2[L];
        int64_t pp = per * per; if (pp < 1) pp = 1;
        int64_t circ = (12566 * a) / pp;
        int64_t esc = a * g_med - (int64_t)T.st_a3[L];
        int ok = a >= p->buraco_area_min && circ >= p->buraco_circ_min_1000 && esc >= a * p->buraco_escuro_min;
        T.st_x0[L] = (uint16_t)ok;                           /* reaproveita como flag */
        if (ok) { q->n_buracos++; q->area_buracos_px += a; }
    }
    for (size_t i = 0; i < n; i++) { uint16_t L = T.lab[i]; T.bur[i] = (L && T.st_x0[L]) ? 1 : 0; }

    MARCA(1);  /* buracos: limiar de textura, morfologia, preenchimento, rótulos, descritores */

    /* (c) trincas */
    for (size_t i = 0; i < n; i++) T.cand_t[i] = T.cand_t[i] && !T.bur[i];
    if (p->trinca_fecho > 1) {
        fechamento(T.cand_t, T.tmpA, w, h, p->trinca_fecho);
        for (size_t i = 0; i < n; i++) T.cand_t[i] = T.cand_t[i] && roi[i] && !T.bur[i];
    }
    uint32_t nt = 0;
    if (rotular(T.cand_t, T.lab, T.pilha, w, h, T.cap_rotulos, &nt)) return ERR_ROTULOS;
    MARCA(2);  /* trincas: limiar variável, fechamento, rótulos */
    memcpy(T.esq, T.cand_t, n);
    afinar(T.esq, T.tmpA, T.pilha, w, h);
    MARCA(3);  /* esqueleto (Zhang-Suen) */
    memset(T.st_area, 0, 4 * (nt + 1)); memset(T.st_a2, 0, 4 * (nt + 1)); memset(T.st_a3, 0, 4 * (nt + 1));
    for (uint32_t L = 0; L <= nt; L++) { T.st_x0[L] = 0xFFFF; T.st_x1[L] = 0; T.st_y0[L] = 0xFFFF; T.st_y1[L] = 0; }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x; uint16_t L = T.lab[i]; if (!L) continue;
            T.st_area[L]++;
            if (x < T.st_x0[L]) T.st_x0[L] = (uint16_t)x;
            if (x > T.st_x1[L]) T.st_x1[L] = (uint16_t)x;
            if (y < T.st_y0[L]) T.st_y0[L] = (uint16_t)y;
            if (y > T.st_y1[L]) T.st_y1[L] = (uint16_t)y;
            if (T.esq[i]) {
                T.st_a2[L]++;
                int viz = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        if ((dy || dx) && px(T.esq, w, h, y + dy, x + dx)) viz++;
                if (viz >= 3) T.st_a3[L]++;
            }
        }
    int64_t comp_pond = 0;
    memset(T.tri, 0, n);
    for (uint32_t L = 1; L <= nt; L++) {
        int64_t a = T.st_area[L], c = T.st_a2[L], bif = T.st_a3[L];
        int aceita = 1;
        if (a < p->trinca_area_min || c < p->trinca_comp_min) aceita = 0;
        else if ((16 * a) / c > p->trinca_larg_max16) aceita = 0;
        if (!aceita) { T.st_area[L] = 0; continue; }
        int64_t bw = T.st_x1[L] - T.st_x0[L] + 1, bh = T.st_y1[L] - T.st_y0[L] + 1;
        int tipo;
        if (100 * bif >= (int64_t)p->jacare_bif_100px * c && c >= p->jacare_comp_min) tipo = 3;
        else if (bh >= 2 * bw) tipo = 0;
        else if (bw >= 2 * bh) tipo = 1;
        else tipo = 2;
        q->tipos[tipo]++;
        q->n_trincas++;
        q->comprimento_trincas_px += c;
        q->area_trincas_px += a;
        comp_pond += c + (tipo == 3 ? c : 0);
        T.st_area[L] = 1;                                    /* flag de aceite */
    }
    for (size_t i = 0; i < n; i++) {
        uint16_t L = T.lab[i];
        T.tri[i] = (L && T.st_area[L] == 1 && T.st_a2[L] >= (uint32_t)p->trinca_comp_min) ? 1 : 0;
    }
    /* esqueleto final restrito às trincas aceitas */
    for (size_t i = 0; i < n; i++) T.esq[i] = T.esq[i] && T.tri[i];

    /* quantificação (idêntica a quantificar() em Python) */
    int64_t nr = n_roi > 0 ? n_roi : 1;
    int64_t comp = q->comprimento_trincas_px;
    q->largura_media_trincas_16 = comp ? (16 * q->area_trincas_px) / comp : 0;
    q->densidade_trincas_1000 = (1000 * comp) / nr;
    q->area_buracos_pct_100 = (10000 * q->area_buracos_px) / nr;
    int64_t ind = (40 * 1000 * comp_pond) / nr + (25 * q->area_buracos_pct_100) / 10;
    q->indice_dano_1000 = ind > 1000 ? 1000 : ind;
    q->severidade = q->indice_dano_1000 < p->sev_min_1000 ? 0 : q->indice_dano_1000 < p->sev_media_1000 ? 1
                  : q->indice_dano_1000 < p->sev_alta_1000 ? 2 : 3;
    q->hash_trincas = fnv1a(T.tri, n);
    q->hash_buracos = fnv1a(T.bur, n);
    if (saida)
        for (size_t i = 0; i < n; i++)
            saida[i] = T.bur[i] ? 255 : T.esq[i] ? 200 : T.tri[i] ? 128 : 0;
    MARCA(4);  /* medidas por componente, classificação e quantificação */
    return 0;
}

/* exportadas para estrada.c */
void pav_morf(const uint8_t *src, uint8_t *dst, uint8_t *tmp, int w, int h, int k, int dilata) { morf(src, dst, tmp, w, h, k, dilata); }
void pav_preencher(uint8_t *m, uint8_t *alc, uint32_t *pilha, int w, int h) { preencher(m, alc, pilha, w, h); }
uint32_t pav_isqrt32(uint32_t v) { return isqrt32(v); }
uint32_t pav_fnv1a(const uint8_t *m, size_t n) { return fnv1a(m, n); }
