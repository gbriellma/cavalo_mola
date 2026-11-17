/* Compila o MESMO nucleo.c no PC para verificar bit a bit contra o Python.
   Usado por scripts/comparar_esp32.py --pc
   ./teste_pc <pasta_com_pgms> <saida.csv>   (lê lista.txt da pasta) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nucleo.h"
#include "estrada.h"
#include "params.h"
#include "formato_csv.h"

static uint32_t relogio_pc(void) { return (uint32_t)((double)clock() * 1000000.0 / CLOCKS_PER_SEC); }

static uint8_t *ler_pgm(const char *f, int *w, int *h) {
    FILE *fp = fopen(f, "rb"); if (!fp) return NULL;
    int maxv; if (fscanf(fp, "P5 %d %d %d", w, h, &maxv) != 3) { fclose(fp); return NULL; }
    fgetc(fp);
    uint8_t *b = malloc((size_t)(*w) * (*h));
    if (fread(b, 1, (size_t)(*w) * (*h), fp) != (size_t)(*w) * (*h)) { fclose(fp); free(b); return NULL; }
    fclose(fp); return b;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s pasta saida.csv\n", argv[0]); return 1; }
    pav_params_t p = PAV_PARAMS_PADRAO;
    pav_definir_relogio(relogio_pc);
    char caminho[1024], nome[512];
    snprintf(caminho, sizeof caminho, "%s/lista.txt", argv[1]);
    FILE *lst = fopen(caminho, "r"); if (!lst) { perror("lista"); return 1; }
    FILE *out = fopen(argv[2], "w"); fprintf(out, "%s\n", PAV_CSV_CABECALHO);
    void *trab = NULL; size_t cap = 0;
    while (fscanf(lst, "%511s", nome) == 1) {
        int w, h, w2, h2;
        snprintf(caminho, sizeof caminho, "%s/%s.pgm", argv[1], nome);
        uint8_t *g = ler_pgm(caminho, &w, &h);
        snprintf(caminho, sizeof caminho, "%s/%s_roi.pgm", argv[1], nome);
        uint8_t *roi = ler_pgm(caminho, &w2, &h2);
        size_t need = pav_bytes_trabalho(w, h);
        if (need > cap) { free(trab); trab = malloc(need); cap = need; }
        uint32_t h_estrada = 0, t_estrada = 0;
        clock_t t0 = clock();
        if (!roi && g) {   /* sem máscara de pista: estrada automática a partir da saturação */
            snprintf(caminho, sizeof caminho, "%s/%s_sat.pgm", argv[1], nome);
            uint8_t *sat = ler_pgm(caminho, &w2, &h2);
            if (sat) {
                roi = malloc((size_t)w * h);
                pav_estrada_params_t pe = PAV_ESTRADA_PADRAO;
                pav_estrada(g, sat, w, h, &pe, trab, roi);
                h_estrada = pav_fnv1a(roi, (size_t)w * h);
                t_estrada = relogio_pc() - (uint32_t)((double)t0 * 1000000.0 / CLOCKS_PER_SEC);
                free(sat);
            }
        } else if (roi) { for (size_t i = 0; i < (size_t)w * h; i++) roi[i] = roi[i] ? 1 : 0; }
        if (!g || !roi) { fprintf(stderr, "falha lendo %s\n", nome); continue; }
        pav_quant_t q;
        int rc = pav_analisar(g, roi, w, h, &p, trab, &q, NULL);
        q.hash_estrada = h_estrada; q.t_estrada_us = t_estrada;
        long us = (long)((clock() - t0) * 1000000.0 / CLOCKS_PER_SEC);
        char linha[1024]; pav_csv_linha(linha, sizeof linha, nome, rc, &q, us);
        fprintf(out, "%s\n", linha);
        free(g); free(roi);
    }
    fclose(out); fclose(lst); free(trab);
    return 0;
}
