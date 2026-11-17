"""Classificação de padrões: tipo de trinca e severidade do dano."""
from __future__ import annotations

TIPOS = ("longitudinal", "transversal", "diagonal", "couro_de_jacare")


def tipo_trinca(d: dict) -> str:
    """Regras sobre descritores de forma.

    - Couro de jacaré (malha): muitas bifurcações por comprimento ou baixa
      solidez com área grande -> padrão de blocos interligados.
    - Caso contrário, pela orientação do eixo principal (câmera olhando ao longo
      da via): ~vertical = longitudinal, ~horizontal = transversal.
    """
    bif_por_100px = 100 * d.get("bifurcacoes", 0) / max(d.get("comprimento", 1), 1)
    if (bif_por_100px >= 3 and d.get("comprimento", 0) >= 80) or (d["solidez"] < 0.25 and d["area"] > 1500):
        return "couro_de_jacare"
    a = d["angulo"]
    if a >= 60:
        return "longitudinal"
    if a <= 30:
        return "transversal"
    return "diagonal"


def severidade(indice: float) -> str:
    if indice < 1:
        return "nenhum"
    if indice < 15:
        return "baixa"
    if indice < 40:
        return "media"
    return "alta"
