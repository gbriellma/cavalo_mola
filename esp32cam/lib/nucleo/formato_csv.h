/* Formato único da linha CSV (firmware e teste no PC). Todos os valores cabem em 32 bits
   para 512x320; usa %ld para funcionar igual no newlib da ESP32 e no glibc. */
#pragma once
#include <stdio.h>
#include "nucleo.h"
#define PAV_CSV_CABECALHO "nome,status,n_trincas,comprimento_trincas_px,area_trincas_px,largura_media_trincas_16,densidade_trincas_1000,longitudinal,transversal,diagonal,couro_de_jacare,n_buracos,area_buracos_px,area_buracos_pct_100,indice_dano_1000,severidade,limiar_textura,hash_trincas,hash_buracos,tempo_us,t_estatisticas_us,t_buracos_us,t_trincas_us,t_esqueleto_us,t_medidas_us,hash_estrada,t_estrada_us"
static inline int pav_csv_linha(char *b, size_t n, const char *nome, int rc, const pav_quant_t *q, long us) {
    return snprintf(b, n, "%s,%d,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%s,%ld,%08lx,%08lx,%ld,%lu,%lu,%lu,%lu,%lu,%08lx,%lu",
        nome, rc, (long)q->n_trincas, (long)q->comprimento_trincas_px, (long)q->area_trincas_px,
        (long)q->largura_media_trincas_16, (long)q->densidade_trincas_1000,
        (long)q->tipos[0], (long)q->tipos[1], (long)q->tipos[2], (long)q->tipos[3],
        (long)q->n_buracos, (long)q->area_buracos_px, (long)q->area_buracos_pct_100,
        (long)q->indice_dano_1000, PAV_SEVERIDADE[q->severidade], (long)q->limiar_textura,
        (unsigned long)q->hash_trincas, (unsigned long)q->hash_buracos, us,
        (unsigned long)q->t_etapa_us[0], (unsigned long)q->t_etapa_us[1], (unsigned long)q->t_etapa_us[2],
        (unsigned long)q->t_etapa_us[3], (unsigned long)q->t_etapa_us[4],
        (unsigned long)q->hash_estrada, (unsigned long)q->t_estrada_us);
}
