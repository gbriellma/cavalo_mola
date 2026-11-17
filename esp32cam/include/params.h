// Gerado por scripts/gerar_params_h.py — NÃO EDITAR À MÃO
#pragma once

#define P_RAIO 9
#define P_TRINCA_A16 16
#define P_TRINCA_B16 32
#define P_TRINCA_FECHO 3
#define P_TRINCA_AREA_MIN 25
#define P_TRINCA_COMP_MIN 14
#define P_TRINCA_LARG_MAX16 48
#define P_JACARE_BIF_100PX 6
#define P_JACARE_COMP_MIN 60
#define P_BURACO_K4 28
#define P_BURACO_ABERTURA 11
#define P_BURACO_FECHO 1
#define P_BURACO_AREA_MIN 250
#define P_BURACO_CIRC_MIN_1000 300
#define P_BURACO_ESCURO_MIN 2
#define P_SEV_MIN_1000 10
#define P_SEV_MEDIA_1000 214
#define P_SEV_ALTA_1000 522

#define PAV_PARAMS_PADRAO { .raio = P_RAIO, .trinca_a16 = P_TRINCA_A16, .trinca_b16 = P_TRINCA_B16, .trinca_fecho = P_TRINCA_FECHO, .trinca_area_min = P_TRINCA_AREA_MIN, .trinca_comp_min = P_TRINCA_COMP_MIN, .trinca_larg_max16 = P_TRINCA_LARG_MAX16, .jacare_bif_100px = P_JACARE_BIF_100PX, .jacare_comp_min = P_JACARE_COMP_MIN, .buraco_k4 = P_BURACO_K4, .buraco_abertura = P_BURACO_ABERTURA, .buraco_fecho = P_BURACO_FECHO, .buraco_area_min = P_BURACO_AREA_MIN, .buraco_circ_min_1000 = P_BURACO_CIRC_MIN_1000, .buraco_escuro_min = P_BURACO_ESCURO_MIN, .sev_min_1000 = P_SEV_MIN_1000, .sev_media_1000 = P_SEV_MEDIA_1000, .sev_alta_1000 = P_SEV_ALTA_1000 }

// Detecção da estrada (params_estrada.json)
#define PAV_ESTRADA_PADRAO { .bloco = 8, .tol_i = 56, .tol_s = 30, .tol_d = 10, .semente_altura = 8, .semente_larg_pct = 20, .fecho = 1 }
