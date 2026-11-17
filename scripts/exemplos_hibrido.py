"""Painéis do pipeline híbrido em imagens de TESTE: blocos classificados, máscaras e rótulo.
Uso: python scripts/exemplos_hibrido.py [--n 8]  -> outputs/hibrido_exemplos/*.png + index.html"""
import argparse, json, os

import cv2
import numpy as np

import _caminho
from _comum import preparar
from pavimento.blocos import B
from pavimento.boosting import carregar
from pavimento.dataset import particao
from pavimento.hibrido import carregar_classificador, ParamPixel, analisar_hibrido
from pavimento.parametros import Parametros
from pavimento.visualizacao import comparar, legenda, painel, sobrepor


def blocos_vis(g, cls):
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for nome, cor in (("trinca", (0, 0, 255)), ("buraco", (255, 0, 255))):
        for r, c in np.argwhere(cls[nome]):
            cv2.rectangle(out, (c * B, r * B), (c * B + B - 1, r * B + B - 1), cor, 1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
    p = Parametros.carregar(_caminho.PARAMS)
    pp = ParamPixel(**json.load(open(os.path.join(_caminho.RAIZ, os.environ.get("PARAMS_PIXEL", "params_pixel.json")))))
    m = carregar_classificador(_caminho.RAIZ, os.environ.get("CLASSIFICADOR", "bayes"))
    nomes = sorted(f[:-4] for f in os.listdir(_caminho.CACHE) if f.endswith(".npz") and particao(f[:-4]) == "teste")
    # escolhe imagens variadas: com buraco rotulado e com muitas trincas
    escolhidos = []
    for n in nomes:
        _, g, roi, gt, gb = preparar(os.path.join(_caminho.DATASET, n))
        if (gb.sum() > 400 and sum(1 for e in escolhidos if e[1] == "b") < a.n // 2) or \
           (gb.sum() == 0 and gt.sum() > 1500 and sum(1 for e in escolhidos if e[1] == "t") < a.n - a.n // 2):
            escolhidos.append((n, "b" if gb.sum() > 400 else "t"))
        if len(escolhidos) >= a.n:
            break
    dest = os.path.join(_caminho.SAIDAS, "hibrido_exemplos"); os.makedirs(dest, exist_ok=True)
    html = ["<!doctype html><meta charset='utf-8'><title>Híbrido</title><style>body{background:#111;color:#eee;font-family:sans-serif}"
            "img{max-width:100%}</style><h1>Pipeline híbrido — imagens de teste</h1>"
            "<p>Painel: blocos classificados (vermelho=trinca, magenta=buraco) | resultado quantificado | comparação "
            "com o rótulo (amarelo=acerto, vermelho=falso positivo, verde=não detectado)</p>"]
    for n, _ in escolhidos:
        _, g, roi, gt, gb = preparar(os.path.join(_caminho.DATASET, n))
        r = analisar_hibrido(g, roi, p, m, pp)
        img = painel([("Blocos classificados", blocos_vis(g, r["blocos"])),
                      ("Resultado", legenda(sobrepor(g, roi, r["trincas"], r["esqueleto"], r["buracos"]), r["quant"])),
                      ("vs rotulo", comparar(g, r["trincas"], r["buracos"], gt, gb))], colunas=3)
        cv2.imwrite(os.path.join(dest, n + ".png"), img)
        html.append(f"<h3>{n}</h3><img src='{n}.png'>")
    open(os.path.join(dest, "index.html"), "w", encoding="utf-8").write("\n".join(html))
    print(len(escolhidos), "exemplos em", dest)


if __name__ == "__main__":
    main()
