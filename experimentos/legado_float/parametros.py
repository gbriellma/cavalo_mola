"""Parâmetros do pipeline (ajustados no conjunto de treino com scripts/ajustar.py)."""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class Parametros:
    # Pré-processamento
    homo_d0: float = 20.0
    homo_gl: float = 0.5
    homo_gh: float = 1.4
    bilat_d: int = 7
    bilat_sc: float = 30.0
    bilat_ss: float = 7.0
    clahe_clip: float = 2.0
    # Trincas (black-hat fino + limiar adaptativo por estatística da pista)
    trinca_bh_k: int = 15
    trinca_k_std: float = 2.5
    trinca_area_min: int = 60
    trinca_alongamento_min: float = 3.0
    trinca_comprimento_min: int = 25
    # Buracos (mapa de textura = desvio padrão local)
    buraco_tex_k: int = 21
    buraco_abertura: int = 7
    buraco_bh_k: int = 61
    buraco_k_std: float = 1.5
    buraco_area_min: int = 600
    buraco_solidez_min: float = 0.45
    buraco_alongamento_max: float = 4.0
    buraco_textura_min: float = 0.0
    saturacao_max: int = 90

    def salvar(self, caminho: str):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def carregar(cls, caminho: str) -> "Parametros":
        with open(caminho, encoding="utf-8") as f:
            return cls(**json.load(f))
