"""Avalia a detecção automática da estrada na VALIDAÇÃO do RTK e gera exemplos visuais.

Saídas: outputs/avaliacao_estrada_rtk.json e outputs/estrada_exemplos/*.png
Uso: python scripts/avaliar_estrada.py
"""
import json, os, time

from dataclasses import asdict

import cv2
import numpy as np

import _caminho
from _rtk import carregar, listar, superficie
from pavimento.estrada import ParamEstrada, crescer_estrada, saturacao_hsi
from pavimento.nucleo import intensidade_hsi

ARQ = os.path.join(_caminho.RAIZ, "params_estrada.json")


def main():
    p = ParamEstrada(**json.load(open(ARQ))) if os.path.exists(ARQ) else ParamEstrada()
    destino = os.path.join(_caminho.SAIDAS, "estrada_exemplos")
    os.makedirs(destino, exist_ok=True)
    por_tipo, tempos, todos = {}, [], []
    dano_dentro = [0, 0]   # pixels de buraco/trinca do rótulo que caem dentro da estrada detectada
    for k, nome in enumerate(listar("val")):
        bgr, lab = carregar(nome)
        rgb = bgr[..., ::-1]
        t = time.perf_counter()
        m, info = crescer_estrada(intensidade_hsi(rgb), saturacao_hsi(rgb), p)
        tempos.append(1000 * (time.perf_counter() - t))
        gt = lab > 0
        tp, fp, fn = int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum())
        iou = tp / max(tp + fp + fn, 1)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        dano = (lab == 11) | (lab == 12)
        dano_dentro[0] += int((dano & m).sum()); dano_dentro[1] += int(dano.sum())
        tipo = superficie(lab)
        por_tipo.setdefault(tipo, []).append((iou, prec, rec))
        todos.append((iou, prec, rec))
        if k % 14 == 0:
            vis = bgr.copy()
            cor = vis.copy(); cor[m] = (0, 200, 0)
            vis = cv2.addWeighted(cor, 0.4, vis, 0.6, 0)
            cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
            cv2.putText(vis, f"{tipo} IoU {iou:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(destino, nome), np.hstack([bgr, vis]))
    resumo = lambda L: {"n": len(L), "iou": round(float(np.mean([x[0] for x in L])), 4),
                        "precisao": round(float(np.mean([x[1] for x in L])), 4),
                        "revocacao": round(float(np.mean([x[2] for x in L])), 4)}
    out = {"dataset": "RTK validação (Rateke et al.)", "params": asdict(p),
           "geral": resumo(todos), "por_superficie": {k: resumo(v) for k, v in por_tipo.items()},
           "pixels_de_dano_dentro_da_estrada_detectada_pct": round(100 * dano_dentro[0] / max(dano_dentro[1], 1), 2),
           "tempo_ms_mediana": round(float(np.median(tempos)), 2)}
    json.dump(out, open(os.path.join(_caminho.SAIDAS, "avaliacao_estrada_rtk.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
