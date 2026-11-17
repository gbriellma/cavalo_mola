"""Parâmetros do núcleo determinístico — todos INTEIROS.

A etapa de buracos em Python não exige paridade com o firmware C.

Ponto fixo: grandezas com sufixo 16 estão multiplicadas por 16; k4 por 4; *_1000 por 1000.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import NamedTuple


class _ReferenciaSeveridade(NamedTuple):
    """Contrato imutável anterior à busca: params.json e índice originais."""
    sev_min_1000: int = 10
    sev_media_1000: int = 214
    sev_alta_1000: int = 522
    dano_peso_trinca: int = 40
    dano_peso_buraco: int = 25


REFERENCIA_SEVERIDADE = _ReferenciaSeveridade()


@dataclass(frozen=True)
class Parametros:
    raio: int = 7                 # janela local (2r+1)x(2r+1) para média e desvio padrão
    # Trincas: limiarização variável  I(x,y) < m_xy - (a*s_xy + b)
    trinca_a16: int = 24          # a = 1.5
    trinca_b16: int = 64          # b = 4 níveis de cinza
    trinca_fraca_a16: int = 8     # candidato permissivo; precisa de semente forte
    trinca_fraca_b16: int = 16
    trinca_abertura_linha: int = 3  # EE H/V/diagonais ou comprimento conectado (1 desliga)
    trinca_fecho: int = 3         # comprimento do EE linear (1 = desligado)
    trinca_area_min: int = 15     # px (na resolução de processamento)
    trinca_comp_min: int = 12     # comprimento mínimo do esqueleto (px)
    trinca_larg_max16: int = 64   # largura média máxima (área/comprimento) = 4 px
    trinca_along_min16: int = 48  # comprimento / largura >= 3
    jacare_bif_100px: int = 6     # bifurcações por 100 px de esqueleto para "couro de jacaré"
    jacare_comp_min: int = 60
    # Buracos: textura = desvio padrão local; limiar global = mediana + k*MAD
    raio_buraco: int = 9          # raio da janela da textura de buracos (independe do raio das trincas)
    buraco_modo: int = 0          # 0: textura+escurecimento com histerese; 1: textura (aulas 7_1/6_1)
    buraco_k4: int = 12           # k = 3
    buraco_abertura: int = 3      # abertura pequena; área elimina resíduos
    buraco_fecho: int = 3         # lado do EE do fechamento
    buraco_area_min: int = 150    # px
    buraco_circ_min_1000: int = 300  # circularidade 4πA/P² mínima (x1000)
    buraco_escuro_min: int = 0    # média da região <= mediana da pista - escuro_min
    buraco_margem_fraca: int = 8  # média local <= mediana da pista - margem
    buraco_margem_forte: int = 16 # sementes: escurecimento maior E textura
    buraco_usar_otsu: int = 0     # 1: limita também pelo Otsu inteiro da pista
    buraco_extensao_min_1000: int = 350  # área / área da bbox (não casco convexo)
    buraco_aspecto_max_1000: int = 3000  # maior / menor lado da bbox
    buraco_fracao_escura_min_1000: int = 600  # pixels escuros / área
    buraco_borda_margem: int = 2  # rejeita região nesta faixa da borda da ROI; 0 desliga
    # Previsão calibrável; a verdade usa REFERENCIA_SEVERIDADE, separadamente.
    dano_peso_trinca: int = 40
    dano_peso_buraco: int = 25
    sev_min_1000: int = 10
    sev_media_1000: int = 150
    sev_alta_1000: int = 400

    def salvar(self, caminho: str):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def carregar(cls, caminho: str) -> "Parametros":
        with open(caminho, encoding="utf-8") as f:
            d = json.load(f)
        nomes = {f.name for f in fields(cls)}
        return cls(**{k: int(v) for k, v in d.items() if k in nomes})

    def cabecalho_c(self) -> str:
        """Gera o params.h do firmware a partir destes valores (fonte única de verdade)."""
        linhas = ["// Gerado por scripts/gerar_params_h.py — NÃO EDITAR À MÃO", "#pragma once", ""]
        for k, v in asdict(self).items():
            linhas.append(f"#define P_{k.upper()} {int(v)}")
        return "\n".join(linhas) + "\n"


# Valores candidatos inteiros para ajuste SOMENTE no treino (T4).
GRADE_BUSCA = {
    "raio_buraco": (5, 7, 9, 11, 13),
    "buraco_modo": (0, 1),
    "trinca_a16": (16, 24, 32),
    "trinca_b16": (32, 64, 96),
    "trinca_fraca_a16": (4, 8, 12),
    "trinca_fraca_b16": (16, 24, 32),
    "trinca_abertura_linha": (1, 3, 5),
    "trinca_fecho": (1, 3, 5),
    "trinca_area_min": (15, 25, 40),
    "trinca_comp_min": (12, 14, 20),
    "trinca_larg_max16": (48, 64, 96),
    "trinca_along_min16": (32, 48, 64),
    "sev_min_1000": (1, 10, 25),
    "sev_media_1000": (100, 150, 214, 300),
    "sev_alta_1000": (400, 522, 700),
    "dano_peso_trinca": (20, 40, 60),
    "dano_peso_buraco": (15, 25, 40),
    "buraco_k4": (8, 12, 20, 28),
    "buraco_abertura": (1, 3, 5),
    "buraco_fecho": (1, 3, 5),
    "buraco_area_min": (80, 150, 250),
    "buraco_circ_min_1000": (200, 300, 400),
    "buraco_escuro_min": (0, 2, 8),
    "buraco_margem_fraca": (4, 8, 12),
    "buraco_margem_forte": (12, 16, 24),
    "buraco_usar_otsu": (0, 1),
    "buraco_extensao_min_1000": (250, 350, 500),
    "buraco_aspecto_max_1000": (2000, 3000, 4000),
    "buraco_fracao_escura_min_1000": (500, 600, 750),
    "buraco_borda_margem": (0, 2, 4),
}
