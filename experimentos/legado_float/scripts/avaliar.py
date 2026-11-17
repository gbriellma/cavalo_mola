"""Avalia o pipeline no dataset (treino/teste) contra as máscaras rotuladas.

Uso: python scripts/avaliar.py --particao teste [--limite 200] [--exemplos 12]
Gera outputs/avaliacao_<particao>.json e outputs/por_imagem_<particao>.csv
"""
import argparse, csv, json, os
from multiprocessing import Pool

import _caminho  # noqa: F401
from pavimento.dataset import carregar, listar_amostras, particao
from pavimento.metricas import confusao, deteccao_imagem, resumir
from pavimento.classificador import carregar_modelos
from pavimento.parametros import Parametros
from pavimento.pipeline import processar

P = None
M = None


def _init(caminho, caminho_modelos):
    global P, M
    P = Parametros.carregar(caminho) if caminho and os.path.exists(caminho) else Parametros()
    M = carregar_modelos(caminho_modelos) if caminho_modelos and os.path.exists(caminho_modelos) else None


def avaliar_uma(pasta):
    s = carregar(pasta)
    roi = s.mascaras["LANE"]
    r = processar(s.imagem, roi, P, M)
    pt, pb = r["trincas"]["mascara"], r["buracos"]["mascara"]
    gt, gb = s.mascaras["CRACK"], s.mascaras["POTHOLE"]
    q = r["quant"]
    return {"nome": s.nome,
            "trinca_px": confusao(pt, gt, roi), "trinca_tol": confusao(pt, gt, roi, tolerancia=5),
            "buraco_px": confusao(pb, gb, roi),
            "tem_trinca_gt": bool(gt.sum() > 50), "tem_trinca_pred": q["comprimento_trincas_px"] > 60,
            "tem_buraco_gt": bool(gb.sum() > 300), "tem_buraco_pred": q["n_buracos"] > 0,
            "quant": q}


def agregar(res):
    soma = lambda k: [sum(r[k][i] for r in res) for i in range(4)]
    return {
        "n_imagens": len(res),
        "trinca_pixel": resumir(*soma("trinca_px")),
        "trinca_tolerancia_5px": resumir(*soma("trinca_tol")),
        "buraco_pixel": resumir(*soma("buraco_px")),
        "trinca_deteccao_imagem": deteccao_imagem([r["tem_trinca_pred"] for r in res], [r["tem_trinca_gt"] for r in res]),
        "buraco_deteccao_imagem": deteccao_imagem([r["tem_buraco_pred"] for r in res], [r["tem_buraco_gt"] for r in res]),
        "tempo_medio_s": round(sum(r["quant"]["tempo_s"] for r in res) / max(len(res), 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particao", default="teste", choices=["treino", "teste", "todas"])
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--params", default=os.path.join(_caminho.RAIZ, "params.json"))
    ap.add_argument("--modelos", default=os.path.join(_caminho.RAIZ, "modelos.json"),
                    help="use '' para desativar o classificador (só regras)")
    ap.add_argument("--processos", type=int, default=os.cpu_count())
    a = ap.parse_args()
    pastas = [d for d in listar_amostras(_caminho.DATASET)
              if a.particao == "todas" or particao(os.path.basename(d)) == a.particao]
    if a.limite:
        pastas = pastas[:: max(1, len(pastas) // a.limite)][: a.limite]
    with Pool(a.processos, initializer=_init, initargs=(a.params, a.modelos)) as pool:
        res = pool.map(avaliar_uma, pastas, chunksize=4)
    ag = agregar(res)
    os.makedirs(_caminho.SAIDAS, exist_ok=True)
    with open(os.path.join(_caminho.SAIDAS, f"avaliacao_{a.particao}{'' if a.modelos else '_regras'}.json"), "w", encoding="utf-8") as f:
        json.dump(ag, f, indent=2, ensure_ascii=False)
    with open(os.path.join(_caminho.SAIDAS, f"por_imagem_{a.particao}.csv"), "w", newline="", encoding="utf-8") as f:
        campos = ["nome", "indice_dano", "severidade", "n_trincas", "comprimento_trincas_px",
                  "largura_media_trincas_px", "n_buracos", "area_buracos_pct_pista", "tempo_s"]
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in res:
            w.writerow({"nome": r["nome"], **{k: r["quant"][k] for k in campos[1:]}})
    print(json.dumps(ag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
