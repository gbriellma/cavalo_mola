"""Processa uma amostra do dataset (ou uma foto qualquer) e gera painel de etapas + JSON.

Uso:
  python scripts/processar_imagem.py dataset/Dataset/v1/1014586_RS_386_386RS124739_29855
  python scripts/processar_imagem.py foto.jpg --roi mascara_pista.png
"""
import argparse, json, os

import cv2
import numpy as np

import _caminho
from pavimento.dataset import carregar
from pavimento.nucleo import analisar, intensidade_hsi, reduzir_2x, reduzir_mascara_2x
from pavimento.parametros import Parametros
from pavimento.visualizacao import comparar, legenda, painel, sobrepor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--roi", help="máscara da pista (se a entrada for uma foto)")
    ap.add_argument("--saida", default=os.path.join(_caminho.SAIDAS, "exemplos"))
    a = ap.parse_args()
    p = Parametros.carregar(_caminho.PARAMS) if os.path.exists(_caminho.PARAMS) else Parametros()
    gt = None
    if os.path.isdir(a.entrada):
        s = carregar(a.entrada)
        img, roi_full, nome = s.imagem, s.mascaras["LANE"], s.nome
        gt = (reduzir_mascara_2x(s.mascaras["CRACK"], "qualquer"), reduzir_mascara_2x(s.mascaras["POTHOLE"], "qualquer"))
    else:
        img = cv2.imread(a.entrada)
        nome = os.path.splitext(os.path.basename(a.entrada))[0]
        roi_full = cv2.imread(a.roi, 0) > 127 if a.roi else np.ones(img.shape[:2], bool)
    g = reduzir_2x(intensidade_hsi(img))
    roi = reduzir_mascara_2x(roi_full, "maioria")
    r = analisar(g, roi, p)
    sd = np.clip(r["desvio"].astype(np.int32) * 6, 0, 255).astype(np.uint8)
    etapas = [("1. Original (reduzida 2x)", cv2.resize(img, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)),
              ("2. Intensidade HSI + ROI", np.where(roi, g, g // 3).astype(np.uint8)),
              ("3. Media local m_xy", r["media"].astype(np.uint8)),
              ("4. Desvio local s_xy (textura)", sd),
              ("5. Limiar variavel (trincas)", r["cand_trinca"]),
              ("6. Limiar textura (buracos)", r["cand_buraco"]),
              ("7. Trincas filtradas + esqueleto", sobrepor(np.zeros_like(g), roi, r["trincas"], r["esqueleto"], np.zeros_like(roi))),
              ("8. Buracos (abertura+preench.)", r["buracos"]),
              ("9. Resultado quantificado", legenda(sobrepor(g, roi, r["trincas"], r["esqueleto"], r["buracos"]), r["quant"]))]
    if gt is not None:
        etapas.append(("10. vs rotulo (verde=nao detect.)", comparar(g, r["trincas"], r["buracos"], *gt)))
    os.makedirs(a.saida, exist_ok=True)
    cv2.imwrite(os.path.join(a.saida, f"{nome}_painel.png"), painel(etapas))
    json.dump({"quant": r["quant"], "regioes_trinca": r["regioes_trinca"], "regioes_buraco": r["regioes_buraco"]},
              open(os.path.join(a.saida, f"{nome}.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(r["quant"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
