"""Gera imagens visíveis a partir da saída da ESP32-CAM (máscaras PGM do cartão SD).

Para cada amostra: imagem de entrada | sobreposição do resultado da placa, com a
quantificação lida do resultados.csv (quando existir). Gera também um mosaico e um
index.html para abrir no navegador.

Uso:
  python scripts/visualizar_esp32.py --entrada para_o_cartao_sd/pav/entrada \
                                     --saida-esp para_o_cartao_sd/pav/saida \
                                     --destino outputs/esp32_visual
Cores: vermelho = trinca, amarelo = esqueleto (comprimento), magenta = buraco, ciano = pista/estrada
(nas amostras sem _roi.pgm o contorno ciano é a estrada detectada automaticamente).
"""
import argparse, csv, html, os

import cv2
import numpy as np

import _caminho
from comparar_esp32 import carregar_entrada, ler_pgm
from pavimento.estrada import ParamEstrada


def sobrepor(g, roi, res):
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 200, 0), 1)
    out[res == 128] = (0, 0, 230)
    out[res == 200] = (0, 255, 255)
    bur = (res == 255).astype(np.uint8)
    cor = out.copy(); cor[bur > 0] = (255, 0, 255)
    out = cv2.addWeighted(cor, 0.45, out, 0.55, 0)
    cnts, _ = cv2.findContours(bur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 0, 255), 2)
    return out


def faixa(img, linhas):
    h = 20 * len(linhas) + 8
    barra = np.zeros((h, img.shape[1], 3), np.uint8)
    for i, t in enumerate(linhas):
        cv2.putText(barra, t, (6, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, barra])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=os.path.join(_caminho.RAIZ, "para_o_cartao_sd", "pav", "entrada"))
    ap.add_argument("--saida-esp", default=os.path.join(_caminho.RAIZ, "para_o_cartao_sd", "pav", "saida"))
    ap.add_argument("--destino", default=os.path.join(_caminho.SAIDAS, "esp32_visual"))
    a = ap.parse_args()
    os.makedirs(a.destino, exist_ok=True)
    import json
    arq_e = os.path.join(_caminho.RAIZ, "params_estrada.json")
    pe = ParamEstrada(**json.load(open(arq_e))) if os.path.exists(arq_e) else ParamEstrada()
    csv_f = os.path.join(a.saida_esp, "resultados.csv")
    quant = {r["nome"]: r for r in csv.DictReader(open(csv_f, encoding="utf-8"))} if os.path.exists(csv_f) else {}
    nomes = open(os.path.join(a.entrada, "lista.txt")).read().split()
    miniaturas, cartoes = [], []
    for nome in nomes:
        f_res = os.path.join(a.saida_esp, nome + "_res.pgm")
        if not os.path.exists(f_res):
            continue
        g, roi, _ = carregar_entrada(a.entrada, nome, pe)   # pista rotulada ou estrada automática
        roi = roi.astype(np.uint8)
        res = ler_pgm(f_res)
        q = quant.get(nome)
        if q:
            linhas = [f"{nome}  |  ESP32: {int(q['tempo_us'])/1000:.0f} ms  |  indice {int(q['indice_dano_1000'])/10:.1f}/100 ({q['severidade']})",
                      f"trincas: {q['n_trincas']}  comprimento {q['comprimento_trincas_px']} px  largura {int(q['largura_media_trincas_16'])/16:.1f} px",
                      f"buracos: {q['n_buracos']}  area {int(q['area_buracos_pct_100'])/100:.2f}% da pista"]
        else:
            nt = int((res == 128).sum() + (res == 200).sum()); nb = int((res == 255).sum())
            linhas = [f"{nome}  |  (sem linha no CSV)", f"pixels de trinca: {nt}   comprimento (esqueleto): {int((res == 200).sum())} px",
                      f"pixels de buraco: {nb}"]
        lado = np.hstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), sobrepor(g, roi, res)])
        img = faixa(lado, linhas)
        cv2.imwrite(os.path.join(a.destino, nome + ".png"), img)
        miniaturas.append(cv2.resize(sobrepor(g, roi, res), (256, int(256 * g.shape[0] / g.shape[1]))))
        cartoes.append((nome, linhas))
    if miniaturas:
        h = max(m.shape[0] for m in miniaturas)
        miniaturas = [cv2.copyMakeBorder(m, 0, h - m.shape[0], 0, 0, cv2.BORDER_CONSTANT) for m in miniaturas]
        while len(miniaturas) % 5:
            miniaturas.append(np.zeros_like(miniaturas[0]))
        mosaico = np.vstack([np.hstack(miniaturas[i:i + 5]) for i in range(0, len(miniaturas), 5)])
        cv2.imwrite(os.path.join(a.destino, "mosaico.jpg"), mosaico, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(os.path.join(a.destino, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Cavalo Mola - saida ESP32</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}img{max-width:100%;border:1px solid #444}"
                "figure{margin:0 0 24px}</style><h1>Cavalo Mola — resultados da ESP32-CAM</h1>"
                "<p>Vermelho = trinca, amarelo = esqueleto (comprimento), magenta = buraco, ciano = pista.</p>"
                "<p><img src='mosaico.jpg'></p>")
        for nome, linhas in cartoes:
            f.write(f"<figure><img src='{nome}.png'><figcaption>{html.escape(' | '.join(linhas))}</figcaption></figure>")
    print(f"{len(cartoes)} imagens geradas em {a.destino} (abra index.html)")


if __name__ == "__main__":
    main()
