"""Busca inteira determinística no TREINO contra referência congelada.

Prioridades: acertos exatos, acertos dentro de um nível, menor erro ordinal.
Empates escolhem os menores limiares lexicograficamente. Nunca recalcula a
verdade com os parâmetros candidatos. Não há subamostragem nem percentis.
"""
import argparse
import json
import os
from dataclasses import asdict, replace
from multiprocessing import Pool  # compatibilidade com gates de validação

import _caminho
from _comum import preparar, referencia_rotulo, somente_treino, validar_amostras
from pavimento.dataset import listar_amostras, particao
from pavimento.nucleo import analisar, severidade
from pavimento.parametros import Parametros, REFERENCIA_SEVERIDADE

NIVEIS = ("nenhum", "baixa", "media", "alta")


def indice_rotulo(pasta):
    """Índice dos rótulos com pesos exclusivamente congelados."""
    return referencia_rotulo(preparar(pasta))["indice_dano_1000"]


def _buscar_limiares(indices, referencias):
    """Ótimo global em 1 <= min < media < alta <= 1000, O(4000 + N).

    Programação dinâmica separa o histograma em quatro intervalos ordenados.
    A utilidade inteira codifica exatamente a prioridade das três métricas.
    Prefixos e máximo acumulado evitam enumerar um bilhão de trios.
    """
    n = len(indices)
    if not n or len(referencias) != n:
        raise ValueError("Calibração exige previsões e referências não vazias de mesmo tamanho.")
    hist = [[0] * 1001 for _ in NIVEIS]
    for ind, ref in zip(indices, referencias):
        if not 0 <= ind <= 1000:
            raise ValueError("Índice previsto fora de 0..1000.")
        hist[ref][ind] += 1
    prefixos = []
    for classe in range(4):
        prefixo = [0]
        for i in range(1001):
            valor = 0
            for ref in range(4):
                distancia = abs(classe - ref)
                utilidade = ((int(distancia == 0) * (n + 1) + int(distancia <= 1))
                             * (3 * n + 1) - distancia)
                valor += hist[ref][i] * utilidade
            prefixo.append(prefixo[-1] + valor)
        prefixos.append(prefixo)
    # dp[t] contém os intervalos já encerrados antes da fronteira t.
    dp = {t: (prefixos[0][t], (t,)) for t in range(1, 1001)}
    for classe in (1, 2):
        seguinte = {}
        melhor = None
        pref = prefixos[classe]
        for t in range(1, 1001):
            anterior = dp.get(t - 1)
            if anterior is not None:
                candidato = (anterior[0] - pref[t - 1], anterior[1])
                if (melhor is None or candidato[0] > melhor[0]
                        or (candidato[0] == melhor[0] and candidato[1] < melhor[1])):
                    melhor = candidato
            if melhor is not None:
                seguinte[t] = (melhor[0] + pref[t], melhor[1] + (t,))
        dp = seguinte
    melhor = None
    for t, (valor, limites) in dp.items():
        candidato = (valor + prefixos[3][-1] - prefixos[3][t], limites)
        if (melhor is None or candidato[0] > melhor[0]
                or (candidato[0] == melhor[0] and candidato[1] < melhor[1])):
            melhor = candidato
    return melhor[1]


def calibrar_severidade(dados, p=Parametros()):
    """Usa previsões do detector e referência fixa, em treino já preparado.

    O chamador que lê arquivos deve usar somente_treino e validar_amostras.
    Também recusamos nomes de teste nesta API para impedir uso acidental.
    """
    dados = list(dados)
    if not dados:
        raise ValueError("Nenhuma amostra de treino para calibrar.")
    somente_treino([d[0] for d in dados])
    referencias = [NIVEIS.index(referencia_rotulo(d)["severidade"]) for d in dados]
    indices = [analisar(d[1], d[2], p)["quant"]["indice_dano_1000"] for d in dados]
    minimo, media, alta = _buscar_limiares(indices, referencias)
    novo = replace(p, sev_min_1000=minimo, sev_media_1000=media, sev_alta_1000=alta)
    previstos = [NIVEIS.index(severidade(i, novo)) for i in indices]
    return novo, {"amostras": len(dados),
                  "acertos_exatos": sum(a == b for a, b in zip(previstos, referencias)),
                  "acertos_ate_1_nivel": sum(abs(a - b) <= 1 for a, b in zip(previstos, referencias)),
                  "erro_ordinal_total": sum(abs(a - b) for a, b in zip(previstos, referencias)),
                  "referencia_severidade": REFERENCIA_SEVERIDADE._asdict(),
                  "nomes_treino": [d[0] for d in dados]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--particao", choices=("treino", "teste"), default="treino")
    parser.add_argument("--params", default=_caminho.PARAMS)
    parser.add_argument("--saida", default=None, help="JSON de parâmetros e auditoria")
    # main() também é chamada sem argumentos pelos gates das ferramentas.
    args = parser.parse_args([] if argv is None else argv)
    if args.particao != "treino":
        raise ValueError("Vazamento recusado: calibração permite somente treino, nunca teste.")
    pastas = [d for d in listar_amostras(args.dataset or _caminho.dataset_atual())
              if particao(os.path.basename(d)) == "treino"]
    somente_treino(pastas)
    validar_amostras(pastas)  # lote inteiro, antes de parâmetros, cache ou detector
    p = Parametros.carregar(args.params)
    novo, auditoria = calibrar_severidade([preparar(d) for d in pastas], p)
    saida = asdict(novo)
    saida["calibracao_severidade"] = auditoria
    with open(args.saida or args.params, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{auditoria['amostras']} imagens de treino | "
          f"acertos exatos: {auditoria['acertos_exatos']} | "
          f"limiares: {novo.sev_min_1000} < {novo.sev_media_1000} < {novo.sev_alta_1000}")


if __name__ == "__main__":
    import sys
    try:
        main(sys.argv[1:])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}")
