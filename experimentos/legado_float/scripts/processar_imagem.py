"""Processa uma imagem (ou amostra do dataset) e salva painel, sobreposição e JSON.

Uso:
  python scripts/processar_imagem.py dataset/Dataset/v1/1007599_RS_386_386RS289112_28920
  python scripts/processar_imagem.py minha_foto.jpg            # ROI estimada automaticamente
"""
import argparse, json, os

import cv2

import _caminho  # noqa: F401
from pavimento.dataset import carregar
from pavimento.classificador import carregar_modelos
from pavimento.parametros import Parametros
from pavimento.pipeline import processar
from pavimento.realce import espectro
from pavimento.visualizacao import comparar_gt, legenda_quantificacao, painel, sobrepor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", help="pasta de amostra do dataset ou arquivo de imagem")
    ap.add_argument("--params", default=os.path.join(_caminho.RAIZ, "params.json"))
    ap.add_argument("--modelos", default=os.path.join(_caminho.RAIZ, "modelos.json"))
    ap.add_argument("--saida", default=os.path.join(_caminho.SAIDAS, "exemplos"))
    a = ap.parse_args()
    p = Parametros.carregar(a.params) if os.path.exists(a.params) else Parametros()
    m = carregar_modelos(a.modelos) if a.modelos and os.path.exists(a.modelos) else None
    gt = None
    if os.path.isdir(a.entrada):
        s = carregar(a.entrada)
        img, roi, nome, gt = s.imagem, s.mascaras["LANE"], s.nome, s.mascaras
    else:
        img, roi, nome = cv2.imread(a.entrada), None, os.path.splitext(os.path.basename(a.entrada))[0]
    r = processar(img, roi, p, m)
    os.makedirs(a.saida, exist_ok=True)
    etapas = [("1. Original", img), ("2. Canal L (Lab)", r["pre"]["L"]),
              ("3. Espectro FFT", espectro(r["pre"]["L"])), ("4. Filtro homomorfico", r["pre"]["homomorfico"]),
              ("5. Bilateral + CLAHE", r["pre"]["realcada"]), ("6. Black-hat (trincas)", r["trincas"]["blackhat"] * 4),
              ("7. Limiar trincas", r["trincas"]["bruta"]), ("8. Trincas filtradas", r["trincas"]["mascara"]),
              ("9. Black-hat (buracos)", r["buracos"]["blackhat"] * 3), ("10. Buracos", r["buracos"]["mascara"]),
              ("11. Resultado", legenda_quantificacao(sobrepor(img, r["trincas"]["mascara"], r["buracos"]["mascara"], r["roi"]), r["quant"]))]
    if gt is not None:
        etapas.append(("12. vs rotulo (verde=GT)", comparar_gt(img, r["trincas"]["mascara"], r["buracos"]["mascara"], gt["CRACK"], gt["POTHOLE"])))
    cv2.imwrite(os.path.join(a.saida, f"{nome}_painel.jpg"), painel(etapas))
    cv2.imwrite(os.path.join(a.saida, f"{nome}_resultado.jpg"), etapas[10][1])
    q = dict(r["quant"])
    with open(os.path.join(a.saida, f"{nome}.json"), "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2, ensure_ascii=False)
    print(json.dumps(q, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
