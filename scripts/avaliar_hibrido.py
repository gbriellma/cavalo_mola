"""Compara o pipeline CLÁSSICO e o HÍBRIDO no conjunto de treino do DNIT.
Uso: python scripts/avaliar_hibrido.py [--limite N]   (usa params_pixel.json)
Saída: outputs/avaliacao_hibrido.json
"""
import argparse, json, os, time
from functools import partial
from multiprocessing import Pool

import numpy as np

import _caminho
from _comum import preparar, referencia_rotulo, validar_amostras
from pavimento.boosting import carregar
from pavimento.dataset import listar_amostras, particao
from pavimento.hibrido import carregar_classificador, ParamPixel, analisar_hibrido
from pavimento.metricas import confusao, resumir
from pavimento.nucleo import analisar
from pavimento.parametros import Parametros

M = None


def _init():
    global M
    M = carregar_classificador(_caminho.RAIZ, os.environ.get("CLASSIFICADOR", "bayes"))


def _uma(pasta, p, pp):
    d = preparar(pasta)
    _, g, roi, gt, gb = d
    out = {"ref": referencia_rotulo(d, p)}
    for tipo in ("classico", "hibrido"):
        t = time.perf_counter()
        r = analisar(g, roi, p) if tipo == "classico" else analisar_hibrido(g, roi, p, M, pp)
        out[tipo] = {"t": confusao(r["trincas"], gt, roi), "tt": confusao(r["trincas"], gt, roi, tolerancia=2),
                     "b": confusao(r["buracos"], gb, roi), "q": r["quant"], "ms": 1000 * (time.perf_counter() - t)}
    return out


def _sp(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()
    try:
        raiz = _caminho.dataset_atual()
        pastas = [d for d in listar_amostras(raiz) if particao(os.path.basename(d)) == "treino"]
        validar_amostras(pastas)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    if a.limite:
        pastas = pastas[:: max(1, len(pastas) // a.limite)][: a.limite]
    if not pastas:
        ap.error("Nenhuma amostra selecionada; verifique --limite.")
    arq = os.path.join(_caminho.RAIZ, os.environ.get("PARAMS_PIXEL", "params_pixel.json"))
    pp = ParamPixel(**json.load(open(arq))) if os.path.exists(arq) else ParamPixel()
    p = Parametros.carregar(_caminho.PARAMS)
    with Pool(os.cpu_count(), initializer=_init) as pool:
        res = pool.map(partial(_uma, p=p, pp=pp), pastas, chunksize=4)
    niv = {"nenhum": 0, "baixa": 1, "media": 2, "alta": 3}
    from dataclasses import asdict
    out = {"n_imagens": len(res), "params_pixel": asdict(pp)}
    for tipo in ("classico", "hibrido"):
        soma = lambda k: [sum(r[tipo][k][i] for r in res) for i in range(4)]
        lp = np.array([r[tipo]["q"]["comprimento_trincas_px"] for r in res], float)
        lr = np.array([r["ref"]["comprimento_trincas_px"] for r in res], float)
        bp = np.array([r[tipo]["q"]["area_buracos_px"] for r in res], float)
        br = np.array([r["ref"]["area_buracos_px"] for r in res], float)
        sp = np.array([niv[r[tipo]["q"]["severidade"]] for r in res]); sr = np.array([niv[r["ref"]["severidade"]] for r in res])
        out[tipo] = {"trinca_pixel": resumir(*soma("t")), "trinca_tol2": resumir(*soma("tt")), "buraco_pixel": resumir(*soma("b")),
                     "comp_trinca_spearman": round(_sp(lp, lr), 4), "comp_trinca_pearson": round(float(np.corrcoef(lp, lr)[0, 1]), 4),
                     "comp_razao_total": round(float(lp.sum() / max(lr.sum(), 1)), 3),
                     "area_buraco_spearman": round(_sp(bp, br), 4), "area_buraco_pearson": round(float(np.corrcoef(bp, br)[0, 1]), 4),
                     "severidade_acerto": round(float((sp == sr).mean()), 4),
                     "severidade_ate_1": round(float((np.abs(sp - sr) <= 1).mean()), 4),
                     "tempo_ms_mediana": round(float(np.median([r[tipo]["ms"] for r in res])), 1)}
    json.dump(out, open(os.path.join(_caminho.SAIDAS, "avaliacao_hibrido.json"), "w"), indent=2, ensure_ascii=False)
    for tipo in ("classico", "hibrido"):
        print(tipo, json.dumps(out[tipo], ensure_ascii=False))


if __name__ == "__main__":
    main()
