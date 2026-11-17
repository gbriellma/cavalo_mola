"""Critérios fixados antes da busca; relatórios no formato de avaliar.py."""
import math

FATOR_BURACO = 2.0
QUEDA_F1_TRINCA_IMAGEM = 0.02
FATOR_TEMPO = 2.0
# Histórico do TESTE: somente referência para a verificação final única.
# Tempo não foi fornecido: não pode ser inventado nem produzir aprovação.
BASELINE_TESTE = {
    "trinca_pixel": {"f1": 0.20},
    "trinca_tolerancia_2px": {"f1": 0.32},
    "buraco_pixel": {"f1": 0.08, "iou": 0.04},
    "trinca_deteccao_imagem": {"f1": 0.93},
    "buraco_deteccao_imagem": {"f1": 0.33},
    "quantificacao": {"severidade": {"acuracia": 0.485}},
}


def avaliar_criterios(base, novo):
    """Dados ausentes/não finitos reprovam; nenhum limite é aprendido no teste."""
    regras = [
        ("buraco_f1_pixel", ("buraco_pixel", "f1"), FATOR_BURACO, 0, False),
        ("buraco_iou", ("buraco_pixel", "iou"), FATOR_BURACO, 0, False),
        ("trinca_f1_tol_2px", ("trinca_tolerancia_2px", "f1"), 1, 0, False),
        ("severidade_exata", ("quantificacao", "severidade", "acuracia"), 1, 0, False),
        ("trinca_f1_imagem", ("trinca_deteccao_imagem", "f1"), 1, -QUEDA_F1_TRINCA_IMAGEM, False),
        ("buraco_f1_imagem", ("buraco_deteccao_imagem", "f1"), 1, 0, False),
        ("tempo_por_imagem", ("tempo_ms_mediana",), FATOR_TEMPO, 0, True),
    ]
    criterios = {}
    for nome, caminho, fator, delta, maximo in regras:
        try:
            b, n = base, novo
            for chave in caminho:
                b, n = b[chave], n[chave]
            b, n = float(b), float(n)
            limite = b * fator + delta
            if not all(math.isfinite(x) for x in (b, n, limite)):
                raise ValueError("métrica não finita")
            ok = n <= limite + 1e-12 if maximo else n >= limite - 1e-12
            criterios[nome] = {"aprovado": ok, "baseline": b, "novo": n, "limite": limite}
        except (KeyError, TypeError, ValueError):
            criterios[nome] = {"aprovado": False, "motivo": "métrica ausente ou inválida"}
    return {"aprovado": all(c["aprovado"] for c in criterios.values()), "criterios": criterios}
