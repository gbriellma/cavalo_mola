"""Sistema completo no RTK (validação): estrada AUTOMÁTICA + detecção/quantificação de danos.
Avalia trincas (classe 12) e buracos (classe 11) só nas imagens de ASFALTO.
Uso: python scripts/avaliar_rtk.py   -> outputs/avaliacao_rtk.json
"""
import json, os

import numpy as np

import _caminho
from _rtk import carregar, listar, superficie
from pavimento.estrada import ParamEstrada, crescer_estrada, saturacao_hsi
from pavimento.metricas import confusao, deteccao_imagem, resumir
from pavimento.nucleo import analisar, intensidade_hsi
from pavimento.parametros import Parametros


def main():
    p = Parametros.carregar(_caminho.PARAMS)
    pe = ParamEstrada(**json.load(open(os.path.join(_caminho.RAIZ, "params_estrada.json"))))
    ct, ctt, cb = [0] * 4, [0] * 4, [0] * 4
    dt, db, n = ([], []), ([], []), 0
    for nome in listar("val"):
        bgr, lab = carregar(nome)
        if superficie(lab) != "asfalto":
            continue
        rgb = bgr[..., ::-1]
        g = intensidade_hsi(rgb)
        roi, _ = crescer_estrada(g, saturacao_hsi(rgb), pe)
        r = analisar(g, roi, p)
        gt, gb = lab == 12, lab == 11
        ct = [a + b for a, b in zip(ct, confusao(r["trincas"], gt))]
        ctt = [a + b for a, b in zip(ctt, confusao(r["trincas"], gt, tolerancia=2))]
        cb = [a + b for a, b in zip(cb, confusao(r["buracos"], gb))]
        dt[0].append(r["quant"]["n_trincas"] > 0); dt[1].append(gt.sum() >= 15)
        db[0].append(r["quant"]["n_buracos"] > 0); db[1].append(gb.sum() >= 30)
        n += 1
    out = {"dataset": "RTK validação, só asfalto, estrada automática", "n_imagens": n,
           "trinca_pixel": resumir(*ct), "trinca_tolerancia_2px": resumir(*ctt), "buraco_pixel": resumir(*cb),
           "trinca_deteccao_imagem": deteccao_imagem(*dt), "buraco_deteccao_imagem": deteccao_imagem(*db)}
    json.dump(out, open(os.path.join(_caminho.SAIDAS, "avaliacao_rtk.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
