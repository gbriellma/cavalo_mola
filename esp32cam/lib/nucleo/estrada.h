/* estrada.h — detecção da área da estrada por crescimento de regiões (porta de estrada.py).
   Referências: 7_2 Segmentação sl.4-14 (crescimento de regiões), 7_1 Cores sl.70-71 (HSI). */
#ifndef PAV_ESTRADA_H
#define PAV_ESTRADA_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int32_t bloco, tol_i, tol_s, tol_d, semente_altura, semente_larg_pct, fecho;
} pav_estrada_params_t;

/* i_img: intensidade HSI; s_img: saturação HSI (0..255); roi: saída 0/1 (w*h).
   trabalho: pav_bytes_trabalho(w,h) bytes (o mesmo buffer do núcleo). Retorna 0 se ok. */
int pav_estrada(const uint8_t *i_img, const uint8_t *s_img, int w, int h,
                const pav_estrada_params_t *p, void *trabalho, uint8_t *roi);
#ifdef __cplusplus
}
#endif
#endif
