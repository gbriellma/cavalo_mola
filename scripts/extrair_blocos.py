"""Extrai descritores e rótulos por bloco de todas as amostras (cache do DNIT) -> outputs/blocos_dnit.npz
Uso: python scripts/extrair_blocos.py"""
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

import _caminho
from _comum import preparar
from pavimento.blocos import descritores, rotulos_blocos
from pavimento.dataset import particao
from pavimento.parametros import Parametros


def _uma(nome, p):
    _, g, roi, gt, gb = preparar(os.path.join(_caminho.DATASET, nome))
    X, _ = descritores(g, roi, p)
    y = rotulos_blocos(gt, gb, roi)
    m = y >= 0
    return nome, X[m], y[m], np.argwhere(m)


def main():
    p = Parametros.carregar(_caminho.PARAMS)
    nomes = sorted(f[:-4] for f in os.listdir(_caminho.CACHE) if f.endswith(".npz"))
    with Pool(os.cpu_count()) as pool:
        res = pool.map(partial(_uma, p=p), nomes, chunksize=8)
    X = np.concatenate([r[1] for r in res]); y = np.concatenate([r[2] for r in res])
    img = np.concatenate([np.full(len(r[2]), i) for i, r in enumerate(res)])
    pos = np.concatenate([r[3] for r in res])
    teste = np.array([particao(r[0]) == "teste" for r in res])[img]
    np.savez_compressed(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"), X=X, y=y, img=img, pos=pos,
                        teste=teste, nomes=np.array([r[0] for r in res]))
    print("blocos", len(y), "| normal", int((y == 0).sum()), "trinca", int((y == 1).sum()), "buraco", int((y == 2).sum()),
          "| imagens", len(res), "teste", int(teste.sum()))


if __name__ == "__main__":
    main()
