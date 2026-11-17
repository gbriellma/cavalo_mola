"""Prepara amostras para o cartão SD da ESP32-CAM (ou para o teste do núcleo C no PC).

Para cada amostra grava, na resolução de processamento (512x320):
  <nome>.pgm      intensidade HSI reduzida 2x (P5, 8 bits)
  <nome>_roi.pgm  região da pista (0/255)
e um lista.txt com os nomes. O firmware lê exatamente estes arquivos.

Uso:
  python scripts/exportar_sd.py --destino E:/pav/entrada --n 40          # cartão SD
  python scripts/exportar_sd.py --destino outputs/sd_teste --n 40        # teste local
  python scripts/exportar_sd.py --destino E:/pav/entrada --fonte rtk --n 40  # estrada automática na placa
"""
import argparse, os

import numpy as np

import _caminho
from _comum import preparar, validar_amostras
from pavimento.dataset import listar_amostras, particao


def gravar_pgm(caminho, img):
    h, w = img.shape
    with open(caminho, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def amostras_dnit(particao_alvo):
    """Cache só pode ser usado com as entradas originais presentes e validadas."""
    raiz = _caminho.dataset_atual()
    pastas = [d for d in listar_amostras(raiz) if particao(os.path.basename(d)) == particao_alvo]
    return validar_amostras(pastas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", required=True)
    ap.add_argument("--n", type=int, default=40, help="quantidade de amostras")
    ap.add_argument("--particao", default="teste", choices=["treino", "teste"])
    ap.add_argument("--fonte", default="dnit", choices=["dnit", "rtk"],
                    help="dnit: grava a máscara da pista (_roi.pgm); rtk: grava a saturação HSI "
                         "(_sat.pgm) e a ESP32 detecta a estrada sozinha")
    a = ap.parse_args()
    if a.n <= 0:
        ap.error("--n deve ser positivo")
    # Falhar antes de criar saídas, inclusive com cache vazio ou antigo.
    pastas = amostras_dnit(a.particao) if a.fonte == "dnit" else None
    os.makedirs(a.destino, exist_ok=True)
    nomes = []
    if a.fonte == "rtk":
        from _rtk import carregar, listar
        from pavimento.estrada import saturacao_hsi
        from pavimento.nucleo import intensidade_hsi
        lista = listar("val" if a.particao == "teste" else "train")
        for arq in lista[:: max(1, len(lista) // a.n)][: a.n]:
            bgr, _ = carregar(arq)
            rgb = bgr[..., ::-1]
            curto = "r" + os.path.splitext(arq)[0][-6:]
            gravar_pgm(os.path.join(a.destino, curto + ".pgm"), intensidade_hsi(rgb))
            gravar_pgm(os.path.join(a.destino, curto + "_sat.pgm"), saturacao_hsi(rgb))
            nomes.append((curto, "RTK/" + arq))
    else:
        pastas = pastas[:: max(1, len(pastas) // a.n)][: a.n]
        for d in pastas:
            nome, g, roi, _, _ = preparar(d)
            curto = nome.split("_")[0]            # nome curto (FAT/8.3 friendly o bastante)
            gravar_pgm(os.path.join(a.destino, curto + ".pgm"), g)
            gravar_pgm(os.path.join(a.destino, curto + "_roi.pgm"), roi.astype(np.uint8) * 255)
            nomes.append((curto, nome))
    with open(os.path.join(a.destino, "lista.txt"), "w", newline="\n") as f:
        f.write("\n".join(c for c, _ in nomes) + "\n")
    with open(os.path.join(a.destino, "mapa_nomes.csv"), "w", newline="\n") as f:
        f.write("curto,nome\n" + "\n".join(f"{c},{n}" for c, n in nomes) + "\n")
    tam = sum(os.path.getsize(os.path.join(a.destino, x)) for x in os.listdir(a.destino))
    print(f"{len(nomes)} amostras gravadas em {a.destino} ({tam/1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}")
