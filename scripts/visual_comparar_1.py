"""Uso: vis1.py <src> <params> <saida.npz> [nomes.json]  — gera máscaras; sem nomes.json seleciona exemplos do teste."""
import json, sys, os
import numpy as np
import bench

def masks(n, nu, p):
    g, roi, gt, gb = bench.carregar(n)
    r = nu.analisar(g, roi, p)
    return r["trincas"], r["buracos"], r["quant"]

if __name__ == "__main__":
    src, pj, out = sys.argv[1:4]
    bench.init(src); nu, pa = bench.M["nu"], bench.M["pa"]
    pd = json.load(open(pj)); p = pa.Parametros(**{k: v for k, v in pd.items() if k in pa.Parametros.__dataclass_fields__})
    if len(sys.argv) > 4:
        nomes = json.load(open(sys.argv[4]))
    else:
        info = []
        for n in bench.nomes("teste"):
            g, roi, gt, gb = bench.carregar(n)
            t, b, q = masks(n, nu, p)
            ab = int((gb & roi).sum()); ap = int(b.sum())
            iou = int((b & gb).sum()) / max(int((b | gb).sum()), 1)
            ct = int((t & gt).sum()) / max(int((t | gt).sum()), 1)
            info.append((n, ab, ap, iou, ct, q["n_trincas"]))
        ach = sorted([i for i in info if i[1] >= 400], key=lambda i: -i[3])
        acertos = [i[0] for i in ach[:6]]
        falhas = [i[0] for i in ach if i[3] == 0][:3]
        fps = [i[0] for i in sorted([i for i in info if i[1] < 75], key=lambda i: -i[2])[:3]]
        trincas = [i[0] for i in sorted([i for i in info if i[1] < 75 and i[5] > 0], key=lambda i: -i[4])[:4]]
        nomes = {"Buracos: melhores acertos": acertos, "Buracos: falhas (rótulo existe, nada coincide)": falhas,
                 "Falsos positivos de buraco (sem buraco no rótulo)": fps, "Trincas: melhores casos": trincas}
        json.dump(nomes, open("nomes_vis.json", "w"), indent=1)
    flat = [n for v in nomes.values() for n in v]
    T = {}; B = {}
    for n in flat:
        t, b, _ = masks(n, nu, p); T[n] = np.packbits(t); B[n] = np.packbits(b)
    np.savez_compressed(out, **{"t_" + n: T[n] for n in flat}, **{"b_" + n: B[n] for n in flat})
    print("ok", len(flat))
