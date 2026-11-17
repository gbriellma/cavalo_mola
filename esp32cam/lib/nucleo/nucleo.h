/*
 * nucleo.h — Núcleo determinístico de detecção e quantificação de danos em pavimentos.
 * Porta linha a linha de src/pavimento/nucleo.py (mesmos inteiros, mesmas bordas,
 * mesma ordem de componentes). Compila na ESP32 (PlatformIO) e no PC (gcc).
 * Referências às aulas: ver comentários em nucleo.c e docs/REFERENCIAS.md.
 */
#ifndef PAV_NUCLEO_H
#define PAV_NUCLEO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t raio;
    int32_t trinca_a16, trinca_b16, trinca_fecho, trinca_area_min, trinca_comp_min, trinca_larg_max16;
    int32_t jacare_bif_100px, jacare_comp_min;
    int32_t buraco_k4, buraco_abertura, buraco_fecho, buraco_area_min, buraco_circ_min_1000, buraco_escuro_min;
    int32_t sev_min_1000, sev_media_1000, sev_alta_1000;
} pav_params_t;

typedef struct {
    int32_t n_trincas;
    int64_t comprimento_trincas_px, area_trincas_px;
    int64_t largura_media_trincas_16, densidade_trincas_1000;
    int32_t tipos[4];            /* longitudinal, transversal, diagonal, couro_de_jacare */
    int32_t n_buracos;
    int64_t area_buracos_px, area_buracos_pct_100;
    int64_t indice_dano_1000;
    int32_t severidade;          /* 0 nenhum, 1 baixa, 2 media, 3 alta */
    int32_t limiar_textura;
    uint32_t hash_trincas, hash_buracos;   /* FNV-1a das máscaras (verificação bit a bit) */
    uint32_t t_etapa_us[5];
    uint32_t hash_estrada;   /* FNV-1a da máscara da estrada (preenchido por quem chama pav_estrada) */
    uint32_t t_estrada_us;  /* estatisticas, buracos, trincas, esqueleto, medidas (0 se sem relógio) */
} pav_quant_t;

/* Memória de trabalho: alocar com pav_bytes_trabalho(w,h) (na ESP32: PSRAM). */
size_t pav_bytes_trabalho(int w, int h);

/* g: intensidade (w*h), roi: 0/1 (w*h). saida (opcional, w*h): 0 fundo, 128 trinca,
   200 esqueleto, 255 buraco. Retorna 0 se ok. */
int pav_analisar(const uint8_t *g, const uint8_t *roi, int w, int h, const pav_params_t *p,
                 void *trabalho, pav_quant_t *q, uint8_t *saida);

uint32_t pav_fnv1a(const uint8_t *m, size_t n);

/* relógio opcional em microssegundos para medir o tempo de cada etapa */
void pav_definir_relogio(uint32_t (*f)(void));

extern const char *PAV_TIPOS[4];
extern const char *PAV_SEVERIDADE[4];

#ifdef __cplusplus
}
#endif
#endif
