"""Avaliação no dataset contra os rótulos (CRACK/POTHOLE), na resolução de processamento.

Uso: python scripts/avaliar.py --particao teste
Saídas: outputs/avaliacao_<particao>.json, outputs/por_imagem_<particao>.csv
"""
import argparse, csv, json, os, time
from functools import partial
from dataclasses import asdict
from multiprocessing import Pool

import _caminho
import numpy as np

from _comum import avaliar_amostra, preparar, referencia_rotulo, validar_amostras
from _nota import pearson
from _criterios import BASELINE_TESTE, avaliar_criterios
from pavimento.dataset import listar_amostras, particao
from pavimento.metricas import deteccao_imagem, resumir
from pavimento.parametros import Parametros


def _uma(pasta, p):
    d = preparar(pasta)
    t = time.perf_counter()
    r = avaliar_amostra(d, p)
    r["tempo_ms"] = round(1000 * (time.perf_counter() - t), 2)
    r["ref"] = referencia_rotulo(d, p)
    return r


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra, rb) if np.std(a) and np.std(b) else 0.0


def metricas_quantificacao(res):
    """Diferencial do projeto: quão bem a QUANTIFICAÇÃO concorda com os rótulos."""
    out = {}
    for k in ("comprimento_trincas_px", "area_buracos_px", "indice_dano_1000"):
        pr = np.array([r["quant"][k] for r in res], float)
        rf = np.array([r["ref"][k] for r in res], float)
        out[k] = {"pearson": round(pearson(pr, rf), 4), "spearman": round(_spearman(pr, rf), 4),
                  "erro_absoluto_mediano": float(np.median(np.abs(pr - rf))),
                  "razao_total_prev_real": round(float(pr.sum() / max(rf.sum(), 1)), 4)}
    niv = {"nenhum": 0, "baixa": 1, "media": 2, "alta": 3}
    sp = np.array([niv[r["quant"]["severidade"]] for r in res]); sr = np.array([niv[r["ref"]["severidade"]] for r in res])
    conf = [[int(((sr == i) & (sp == j)).sum()) for j in range(4)] for i in range(4)]
    out["severidade"] = {"acuracia": round(float((sp == sr).mean()), 4),
                         "acuracia_ate_1_nivel": round(float((np.abs(sp - sr) <= 1).mean()), 4),
                         "matriz_confusao_linhas_rotulo_colunas_previsto(nenhum,baixa,media,alta)": conf}
    return out


def avaliar_pastas(pastas, p, processos=1):
    pastas = validar_amostras(pastas)
    if processos == 1:
        res = [_uma(pasta, p) for pasta in pastas]
    else:
        with Pool(processos) as pool:
            res = pool.map(partial(_uma, p=p), pastas, chunksize=8)
    return agregar(res, p)


def agregar(res, p):
    soma = lambda k: [sum(r[k][i] for r in res) for i in range(4)]
    tempos = sorted(r["tempo_ms"] for r in res)
    ag = {"n_imagens": len(res), "resolucao": "512x320 (metade de 1024x640)", "params": asdict(p),
          "trinca_pixel": resumir(*soma("trinca_px")),
          "trinca_tolerancia_2px": resumir(*soma("trinca_tol")),
          "buraco_pixel": resumir(*soma("buraco_px")),
          "trinca_deteccao_imagem": deteccao_imagem([r["tem_trinca_pred"] for r in res], [r["tem_trinca_gt"] for r in res]),
          "buraco_deteccao_imagem": deteccao_imagem([r["tem_buraco_pred"] for r in res], [r["tem_buraco_gt"] for r in res]),
          "quantificacao": metricas_quantificacao(res),
          "tempo_ms_mediana": tempos[len(tempos) // 2], "tempo_ms_p95": tempos[int(0.95 * len(tempos))]}
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particao", default="teste", choices=["treino", "teste", "todas"])
    ap.add_argument("--processos", type=int, default=os.cpu_count())
    ap.add_argument("--baseline", action="store_true", help="salva baseline_treino.json (somente treino)")
    ap.add_argument("--comparar", metavar="BASELINE_JSON", help="compara com relatório anterior da mesma partição")
    a = ap.parse_args()
    if a.processos < 1:
        ap.error("--processos deve ser positivo")
    if a.baseline and a.particao != "treino":
        ap.error("baseline só pode ser gerado no treino")
    if a.comparar and (a.particao == "todas" or a.baseline):
        ap.error("comparação exige uma partição e não pode sobrescrever o baseline")
    base = None
    if a.comparar:
        with open(a.comparar, encoding="utf-8-sig") as f:
            base = json.load(f)
        if base.get("particao", a.particao) != a.particao:
            ap.error("baseline pertence a outra partição")
    elif a.particao == "teste":
        # Verificação final; nunca consultada por ajustar.py. O histórico dado
        # não contém tempo: esse critério reprova até haver baseline completo.
        base = BASELINE_TESTE
    raiz = _caminho.dataset_atual()
    p = Parametros.carregar(_caminho.PARAMS) if os.path.exists(_caminho.PARAMS) else Parametros()
    pastas = [d for d in listar_amostras(raiz)
              if a.particao == "todas" or particao(os.path.basename(d)) == a.particao]
    validar_amostras(pastas)
    with Pool(a.processos) as pool:
        res = pool.map(partial(_uma, p=p), pastas, chunksize=8)
    ag = agregar(res, p)
    ag.update({"particao": a.particao, "dataset": raiz,
               "referencia_severidade": "rotulos com parametros do relatorio; nao congelada T3"})
    if base is not None:
        ag["criterios"] = avaliar_criterios(base, ag)
    os.makedirs(_caminho.SAIDAS, exist_ok=True)
    with open(os.path.join(_caminho.SAIDAS, f"avaliacao_{a.particao}.json"), "w", encoding="utf-8") as f:
        json.dump(ag, f, indent=2, allow_nan=False)
    if a.baseline:
        with open(os.path.join(_caminho.SAIDAS, "baseline_treino.json"), "w", encoding="utf-8") as f:
            json.dump(ag, f, indent=2, ensure_ascii=False, allow_nan=False)
    campos = ["nome", "indice_dano_1000", "severidade", "n_trincas", "comprimento_trincas_px",
              "largura_media_trincas_16", "n_buracos", "area_buracos_pct_100"]
    with open(os.path.join(_caminho.SAIDAS, f"por_imagem_{a.particao}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(campos + ["tempo_ms"])
        for r in res:
            w.writerow([r["nome"]] + [r["quant"][k] for k in campos[1:]] + [r["tempo_ms"]])
    print(json.dumps(ag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}")
