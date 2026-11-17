/* estrada.c — porta exata de src/pavimento/estrada.py (só inteiros). */
#include "estrada.h"
#include "nucleo.h"
#include <string.h>

/* funções de nucleo.c reaproveitadas (morfologia e preenchimento) */
void pav_morf(const uint8_t *src, uint8_t *dst, uint8_t *tmp, int w, int h, int k, int dilata);
void pav_preencher(uint8_t *m, uint8_t *alc, uint32_t *pilha, int w, int h);
uint32_t pav_isqrt32(uint32_t v);

static int mediana_inferior(const int32_t *v, int gw, int r0, int r1, int c0, int c1) {
    uint32_t hst[256]; memset(hst, 0, sizeof hst);
    uint32_t n = 0;
    for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) { hst[v[r * gw + c]]++; n++; }
    if (!n) return 0;
    uint32_t alvo = (n + 1) / 2, acc = 0;
    for (int x = 0; x < 256; x++) { acc += hst[x]; if (acc >= alvo) return x; }
    return 255;
}

int pav_estrada(const uint8_t *I, const uint8_t *S, int w, int h,
                const pav_estrada_params_t *p, void *trabalho, uint8_t *roi) {
    int b = p->bloco, gh = h / b, gw = w / b, ng = gh * gw;
    if (gh <= 0 || gw <= 0) return 3;
    /* buffers do grid dentro da memória de trabalho */
    int32_t *mi = (int32_t *)trabalho, *ms = mi + ng, *md = ms + ng;
    uint8_t *ok = (uint8_t *)(md + ng), *reg = ok + ng, *tmp = reg + ng, *alc = tmp + ng;
    uint32_t *pilha = (uint32_t *)(alc + ((ng + 3) & ~3));
    int64_t n = (int64_t)b * b;
    for (int r = 0; r < gh; r++)
        for (int c = 0; c < gw; c++) {
            int64_t si = 0, si2 = 0, ss = 0;
            for (int y = r * b; y < r * b + b; y++)
                for (int x = c * b; x < c * b + b; x++) {
                    int64_t v = I[y * w + x]; si += v; si2 += v * v; ss += S[y * w + x];
                }
            mi[r * gw + c] = (int32_t)(si / n);
            ms[r * gw + c] = (int32_t)(ss / n);
            md[r * gw + c] = (int32_t)pav_isqrt32((uint32_t)((n * si2 - si * si) / (n * n)));
        }
    int c0 = gw * (100 - p->semente_larg_pct) / 200, c1 = gw - c0;
    int r0 = gh - p->semente_altura; if (r0 < 0) r0 = 0;
    int ri = mediana_inferior(mi, gw, r0, gh, c0, c1);
    int rs = mediana_inferior(ms, gw, r0, gh, c0, c1);
    int rd = mediana_inferior(md, gw, r0, gh, c0, c1);
    for (int k = 0; k < ng; k++) {
        int di = mi[k] - ri, ds = ms[k] - rs;
        ok[k] = (di <= p->tol_i && -di <= p->tol_i && ds <= p->tol_s && -ds <= p->tol_s && md[k] <= rd + p->tol_d);
    }
    /* crescimento 4-conexo a partir das sementes válidas */
    memset(reg, 0, ng);
    size_t sp = 0;
    for (int r = r0; r < gh; r++)
        for (int c = c0; c < c1; c++)
            if (ok[r * gw + c] && !reg[r * gw + c]) { reg[r * gw + c] = 1; pilha[sp++] = (uint32_t)(r * gw + c); }
    while (sp) {
        uint32_t k = pilha[--sp]; int r = k / gw, c = k % gw;
        const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int rr = r + dr[d], cc = c + dc[d];
            if (rr < 0 || rr >= gh || cc < 0 || cc >= gw) continue;
            int j = rr * gw + cc;
            if (ok[j] && !reg[j]) { reg[j] = 1; pilha[sp++] = (uint32_t)j; }
        }
    }
    if (p->fecho > 1) { pav_morf(reg, reg, tmp, gw, gh, p->fecho, 1); pav_morf(reg, reg, tmp, gw, gh, p->fecho, 0); }
    pav_preencher(reg, alc, pilha, gw, gh);
    memset(roi, 0, (size_t)w * h);
    for (int r = 0; r < gh; r++)
        for (int c = 0; c < gw; c++)
            if (reg[r * gw + c])
                for (int y = r * b; y < r * b + b; y++) memset(roi + (size_t)y * w + c * b, 1, b);
    return 0;
}
