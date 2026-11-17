"""Busca coordenada determinística, SOMENTE no treino (cache). Nota = a de scripts/_nota.py.

Uso: [OBJ=buraco] [SO=prefixo1,prefixo2] python scripts/ajustar_cache.py --src . --inicio params.json --saida novo.json
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

import avaliar_cache as bench
OBJ = os.environ.get("OBJ", "geral")
SO = os.environ.get("SO", "")


def nota(res, me):
    soma = lambda k: [sum(r[k][i] for r in res) for i in range(4)]
    ft = me.resumir(*soma("trinca_tol"))["f1"]
    bu = me.resumir(*soma("buraco_px"))
    dt = me.deteccao_imagem([r["tc_pr"] for r in res], [r["tc_gt"] for r in res])["f1"]
    db = me.deteccao_imagem([r["bu_pr"] for r in res], [r["bu_gt"] for r in res])["f1"]
    lp = [r["comp_pr"] for r in res]; lg = [r["comp_gt"] for r in res]
    ap = [r["ab_pr"] for r in res]; ag = [r["ab_gt"] for r in res]
    ct, cb = bench.corr(lp, lg), bench.corr(ap, ag)
    vies = abs(float(np.log((sum(lp) + 1) / (sum(lg) + 1))))
    if OBJ == "buraco":
        v = 3 * bu["f1"] + 3 * bu["iou"] + db + 0.5 * cb
    else:
        v = 2 * bu["f1"] + 2 * bu["iou"] + ft + 0.3 * (dt + db) + ct + cb - 0.5 * vies
    return v, dict(f1_bur=bu["f1"], iou_bur=bu["iou"], f1_tol=ft, det_t=dt, det_b=db, ct=round(ct, 3), cb=round(cb, 3), vies=round(vies, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--inicio", required=True)
    ap.add_argument("--saida", required=True); ap.add_argument("--passadas", type=int, default=2)
    ap.add_argument("--passo", type=int, default=4, help="usa 1 de cada N imagens do treino")
    ap.add_argument("--grupos", default="todos")
    a = ap.parse_args()
    bench.init(a.src)
    pa = bench.M["pa"]
    p = {k: v for k, v in json.load(open(a.inicio, encoding="utf-8-sig")).items()}
    grade = dict(getattr(pa, "GRADE_BUSCA", {}))
    # grade ampla = valores da grade + vizinhança do valor atual (x0.5, x0.75, x1.25, x1.5, x2)
    proib = lambda k: k.startswith("sev_") or "dano_peso" in k
    ns = bench.nomes("treino")[::a.passo]
    print("imagens de busca:", len(ns), flush=True)
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=bench.init, initargs=(a.src,)) as pool:
        def ev(pd):
            res = pool.map(bench.uma, [(n, pd) for n in ns], chunksize=4)
            return nota(res, bench.M["me"])
        melhor, info = ev(p)
        print(f"inicial {melhor:.4f} {info}", flush=True)
        for ps in range(a.passadas):
            mudou = False
            for k in list(p):
                if proib(k) or k not in pa.Parametros.__dataclass_fields__:
                    continue
                if SO and not any(k.startswith(x) for x in SO.split(",")):
                    continue
                atual = p[k]
                cands = set(grade.get(k, ()))
                for f in (0.5, 0.75, 1.25, 1.5, 2.0):
                    cands.add(int(round(atual * f)))
                if atual == 0:
                    cands |= {1, 2, 4}
                if atual in (0, 1) or k in ("buraco_usar_otsu",):
                    pass
                cands = sorted(c for c in cands if c != atual and c >= 0)
                if k == "buraco_usar_otsu":
                    cands = [1 - atual]
                if k == "raio":
                    cands = [c for c in cands if 3 <= c <= 15]
                for c in cands:
                    q = dict(p); q[k] = c
                    try:
                        n_, inf = ev(q)
                    except Exception as e:  # valor inválido para o núcleo
                        continue
                    if n_ > melhor + 1e-9:
                        melhor, info, p, mudou = n_, inf, q, True
                print(f"[p{ps}] {k}={p[k]} nota={melhor:.4f} {info} ({time.time()-t0:.0f}s)", flush=True)
                json.dump(p, open(a.saida, "w"), indent=2)
            if not mudou:
                break
    print("FINAL", melhor, info)
    json.dump(p, open(a.saida, "w"), indent=2)


if __name__ == "__main__":
    main()
