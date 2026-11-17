"""Classificador de blocos em DOIS ESTÁGIOS (auto-contexto) com boosting do zero — treino RETOMÁVEL.

Estágio 1: boosting sobre os descritores do bloco (blocos.py).
Estágio 2: descritores + escores do estágio 1 do próprio bloco e dos 8 vizinhos (média, máximo),
por classe. Os escores de estágio 1 usados no treino do estágio 2 vêm de validação cruzada em
2 dobras por imagem (cross-fitting), para não vazar informação. Tudo determinístico.

Cada execução avança o treino por ~150 s e salva o progresso em outputs/treino_blocos/.
Rode repetidamente até aparecer "FIM":   python scripts/treinar_blocos2.py
"""
import json, os, pickle, time

import numpy as np

import _caminho
from pavimento.boosting import Boosting, salvar

ARV, PROF, TAXA = 500, 4, 0.1
DIR = os.path.join(_caminho.SAIDAS, "treino_blocos")
T0 = time.time()
ORCAMENTO = float(os.environ.get("ORCAMENTO", 150))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f1(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precisao": round(p, 4), "revocacao": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "tp": tp, "fp": fp, "fn": fn}


def job(nome, X, alvo, tr, val):
    """Treina (retomando) um modelo; devolve o modelo pronto ou None se ainda não terminou."""
    arq = os.path.join(DIR, nome + ".pkl")
    neg = np.flatnonzero(tr & ~alvo)[::3]
    idx = np.sort(np.concatenate([np.flatnonzero(tr & alvo), neg]))
    if os.path.exists(arq):
        m, estado = pickle.load(open(arq, "rb"))
        if estado["fim"]:
            return m
    else:
        m, estado = Boosting(n_arvores=ARV, profundidade=PROF, taxa=TAXA), None
    resta = ORCAMENTO - (time.time() - T0)
    if resta < 10:
        return None
    log(f"{nome}: continuando de {len(m.arvores)} árvores")
    estado = m.treinar(X[idx], alvo[idx].astype(float), X[val], alvo[val].astype(float),
                       paciencia=40, log=log, tempo_max=resta, estado=estado)
    pickle.dump((m, estado), open(arq, "wb"))
    log(f"{nome}: {len(m.arvores)} árvores, fim={estado['fim']}")
    return m if estado["fim"] else None


def contexto_escores(esc, img, pos):
    out = np.zeros((len(esc), 3), np.int64)
    ordem = np.argsort(img, kind="stable")
    for grupo in np.split(ordem, np.flatnonzero(np.diff(img[ordem])) + 1):
        p = pos[grupo]
        gh, gw = p[:, 0].max() + 3, p[:, 1].max() + 3
        grade = np.zeros((gh, gw), np.int64); tem = np.zeros((gh, gw), bool)
        grade[p[:, 0] + 1, p[:, 1] + 1] = esc[grupo]; tem[p[:, 0] + 1, p[:, 1] + 1] = True
        soma = np.zeros(len(grupo), np.int64); cnt = np.zeros(len(grupo), np.int64)
        mx = esc[grupo].copy()
        primeiro = np.ones(len(grupo), bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = p[:, 0] + 1 + dy, p[:, 1] + 1 + dx
                ok = tem[yy, xx]; v = grade[yy, xx]
                soma += np.where(ok, v, 0); cnt += ok
                mx = np.where(ok & (primeiro | (v > mx)), v, mx); primeiro &= ~ok
        out[grupo, 0] = esc[grupo]; out[grupo, 1] = soma // np.maximum(cnt, 1); out[grupo, 2] = mx
    return out


def main():
    os.makedirs(DIR, exist_ok=True)
    d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
    X, y, img, pos, teste = d["X"], d["y"], d["img"], d["pos"], d["teste"]
    val = (~teste) & (img % 5 == 0); tr = (~teste) & ~val; dobra = img % 2
    modelos, ctx = {}, []
    for classe, nome in ((1, "trinca"), (2, "buraco")):
        alvo = y == classe
        ms = [job(f"{nome}_e1_dobra{k}", X, alvo, tr & (dobra != k), val & (dobra != k)) for k in (0, 1)]
        m1 = job(f"{nome}_e1", X, alvo, tr, val)
        if None in ms or m1 is None:
            log("PARCIAL — rode de novo para continuar"); return
        esc = np.zeros(len(y), np.int64)
        for k in (0, 1):
            esc[dobra == k] = ms[k].pontuar(X[dobra == k])
        esc[~tr] = m1.pontuar(X[~tr])
        modelos[nome + "_e1"] = m1
        ctx.append(contexto_escores(esc, img, pos))
    X2 = np.concatenate([X] + ctx, axis=1)
    rel, pred = {}, {}
    for classe, nome in ((1, "trinca"), (2, "buraco")):
        alvo = y == classe
        m2 = job(f"{nome}_e2", X2, alvo, tr, val)
        if m2 is None:
            log("PARCIAL — rode de novo para continuar"); return
        sv = m2.pontuar(X2[val])
        cands = np.unique(np.percentile(sv, np.linspace(50, 99.9, 300)).astype(np.int64))
        lim = int(max(cands, key=lambda t: f1(sv >= t, alvo[val])["f1"]))
        st = m2.pontuar(X2[teste])
        modelos[nome + "_e2"] = m2; modelos[nome + "_limiar"] = lim
        pred[nome] = st >= lim
        rel[nome] = {"arvores_e1": len(modelos[nome + "_e1"].arvores), "arvores_e2": len(m2.arvores),
                     "limiar": lim, "validacao": f1(sv >= lim, alvo[val]), "teste": f1(st >= lim, alvo[teste])}
    rel["dano_qualquer_teste"] = f1(pred["trinca"] | pred["buraco"], y[teste] > 0)
    rel["config"] = {"arvores_max": ARV, "profundidade": PROF, "taxa": TAXA}
    rel["referencia"] = "Fernandes, Passos et al. (2023), mesmo dataset: F=75,16% (blocos com trinca/buraco)"
    salvar(modelos, os.path.join(_caminho.RAIZ, "modelos_blocos.json"))
    json.dump(rel, open(os.path.join(_caminho.SAIDAS, "avaliacao_blocos.json"), "w"), indent=2, ensure_ascii=False)
    log("FIM", json.dumps({k: rel[k]["teste"] if "teste" in rel[k] else rel[k] for k in ("trinca", "buraco", "dano_qualquer_teste")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
