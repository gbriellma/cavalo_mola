"""Avaliação sobre o cache .npz de outputs/cache (sem o dataset bruto); mesma divisão treino/teste (hash do nome).

Uso: python scripts/avaliar_cache.py --particao teste [--src <raiz com src/pavimento>] [--params params.json] [--out r.json]
CAVALO_CACHE aponta para outra pasta de .npz. Severidade de referência congelada: 10/214/522, pesos 40/25.
"""
import argparse, glob, hashlib, json, os, sys, time
from dataclasses import asdict
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("CAVALO_CACHE", os.path.join(RAIZ, "outputs", "cache"))
REF = SimpleNamespace(sev_min_1000=10, sev_media_1000=214, sev_alta_1000=522)  # referência congelada
W_T, W_B = 40, 25
DADOS = {}
M = {}


def particao(nome, frac=0.3):
    return "teste" if int(hashlib.md5(nome.encode()).hexdigest(), 16) % 1000 < frac * 1000 else "treino"


def nomes(part):
    ns = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE, "*.npz")))
    return [n for n in ns if part == "todas" or particao(n) == part]


def init(src):
    sys.path.insert(0, os.path.join(src, "src"))
    import pavimento.nucleo as nu, pavimento.metricas as me, pavimento.parametros as pa
    M.update(nu=nu, me=me, pa=pa)


def carregar(n):
    if n not in DADOS:
        z = np.load(os.path.join(CACHE, n + ".npz"))
        DADOS[n] = (z["g"], z["roi"], z["gt"], z["gb"])
    return DADOS[n]


def sev(ind):
    return "nenhum" if ind < REF.sev_min_1000 else "baixa" if ind < REF.sev_media_1000 else "media" if ind < REF.sev_alta_1000 else "alta"


def indice(comp, area_b, n):
    n = max(n, 1)
    return min(1000, (W_T * 1000 * comp) // n + (W_B * ((10000 * area_b) // n)) // 10)


def uma(args):
    n, pd = args
    nu, me, pa = M["nu"], M["me"], M["pa"]
    p = pa.Parametros(**{k: v for k, v in pd.items() if k in pa.Parametros.__dataclass_fields__})
    g, roi, gt, gb = carregar(n)
    t = time.perf_counter()
    r = nu.analisar(g, roi, p)
    dt = 1000 * (time.perf_counter() - t)
    comp = int(nu.afinar(gt & roi).sum()); ab = int((gb & roi).sum()); nr = int(roi.sum())
    ind = indice(comp, ab, nr)
    q = r["quant"]
    return {"trinca_px": me.confusao(r["trincas"], gt, roi), "trinca_tol": me.confusao(r["trincas"], gt, roi, tolerancia=2),
            "buraco_px": me.confusao(r["buracos"], gb, roi),
            "tc_gt": bool(gt.sum() >= 15), "tc_pr": q["n_trincas"] > 0, "bu_gt": bool(gb.sum() >= 75), "bu_pr": q["n_buracos"] > 0,
            "comp_pr": q["comprimento_trincas_px"], "ab_pr": q["area_buracos_px"], "ind_pr": q["indice_dano_1000"], "sev_pr": q["severidade"],
            "comp_gt": comp, "ab_gt": ab, "ind_gt": ind, "sev_gt": sev(ind), "ms": dt}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() and len(a) > 1 else 0.0


def agregar(res, me):
    soma = lambda k: [sum(r[k][i] for r in res) for i in range(4)]
    niv = {"nenhum": 0, "baixa": 1, "media": 2, "alta": 3}
    sp = np.array([niv[r["sev_pr"]] for r in res]); sr = np.array([niv[r["sev_gt"]] for r in res])
    ms = sorted(r["ms"] for r in res)
    rk = lambda a: np.argsort(np.argsort(a))
    lp = [r["comp_pr"] for r in res]; lg = [r["comp_gt"] for r in res]
    ap = [r["ab_pr"] for r in res]; ag = [r["ab_gt"] for r in res]
    return {"n": len(res),
            "trinca_pixel": me.resumir(*soma("trinca_px")), "trinca_tol2": me.resumir(*soma("trinca_tol")),
            "buraco_pixel": me.resumir(*soma("buraco_px")),
            "trinca_img": me.deteccao_imagem([r["tc_pr"] for r in res], [r["tc_gt"] for r in res]),
            "buraco_img": me.deteccao_imagem([r["bu_pr"] for r in res], [r["bu_gt"] for r in res]),
            "spearman_comp": round(corr(rk(lp), rk(lg)), 4), "razao_comp": round(sum(lp) / max(sum(lg), 1), 4),
            "spearman_area_buraco": round(corr(rk(ap), rk(ag)), 4),
            "sev_exata": round(float((sp == sr).mean()), 4), "sev_ate1": round(float((abs(sp - sr) <= 1).mean()), 4),
            "ms_med": round(ms[len(ms) // 2], 2), "ms_p95": round(ms[int(.95 * len(ms))], 2)}


def avaliar(src, pd, part, pool=None, ns=None):
    ns = ns or nomes(part)
    if pool is None:
        with Pool(os.cpu_count(), initializer=init, initargs=(src,)) as pool:
            res = pool.map(uma, [(n, pd) for n in ns], chunksize=8)
    else:
        res = pool.map(uma, [(n, pd) for n in ns], chunksize=8)
    init(src) if not M else None
    return agregar(res, M["me"]), res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=RAIZ); ap.add_argument("--params", default=os.path.join(RAIZ, "params.json"))
    ap.add_argument("--particao", default="teste"); ap.add_argument("--out")
    a = ap.parse_args()
    init(a.src)
    pd = json.load(open(a.params, encoding="utf-8-sig"))
    ag, _ = avaliar(a.src, pd, a.particao)
    s = json.dumps(ag, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s)
