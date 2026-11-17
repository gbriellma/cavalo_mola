"""Busca coordenada determinística restrita ao treino, com critérios fixados."""
import argparse, json, os, time
from pathlib import Path
from dataclasses import asdict, replace
from multiprocessing import Pool

import _caminho
from _comum import avaliar_amostra, preparar, validar_amostras, somente_treino
from _nota import nota
from _criterios import avaliar_criterios
from avaliar import avaliar_pastas
from pavimento import parametros as modulo_parametros
from pavimento.dataset import listar_amostras, particao
from pavimento.parametros import Parametros

GRADE_PADRAO = {
    "raio": [5, 7, 9, 11],
    "trinca_a16": [8, 12, 16, 20, 24, 32, 40],
    "trinca_b16": [0, 32, 64, 96, 128, 192],
    "trinca_fecho": [1, 3, 5],
    "trinca_area_min": [8, 15, 25, 40],
    "trinca_comp_min": [6, 10, 14, 20, 30],
    "trinca_larg_max16": [24, 32, 40, 48, 64],
    "buraco_k4": [12, 16, 20, 28, 36, 44, 56],
    "buraco_abertura": [3, 5, 7, 9, 11],
    "buraco_fecho": [1, 5, 9, 13],
    "buraco_area_min": [150, 250, 400, 600, 900],
    "buraco_circ_min_1000": [150, 300, 450, 600, 700, 800],
    "buraco_escuro_min": [-10, -5, 0, 2, 4, 8],
}
def montar_grade(modulo=modulo_parametros):
    grade = {**GRADE_PADRAO, **getattr(modulo, "GRADE_BUSCA", {})}
    # Pesos do índice e limiares de severidade pertencem à calibração T3.
    return {k: v for k, v in grade.items()
            if not k.startswith("sev_") and "peso" not in k.lower()
            and "weight" not in k.lower() and "indice_dano" not in k.lower()}


GRADE = montar_grade()
DADOS = []


REF = []   # (comprimento real das trincas, área real dos buracos) por imagem


def _init(pastas):
    global DADOS, REF
    from pavimento.nucleo import afinar
    somente_treino(pastas)
    validar_amostras(pastas)
    DADOS = [preparar(d) for d in pastas]
    REF = [(int(afinar(gt & roi).sum()), int((gb & roi).sum())) for _, _, roi, gt, gb in DADOS]


def _nota(pd):
    res = [avaliar_amostra(d, Parametros(**pd)) for d in DADOS]
    return nota(res, REF)


def carregar_baseline(raiz):
    """Lê a referência original; nunca a redefine com os parâmetros da busca."""
    caminho = Path(_caminho.SAIDAS) / "baseline_treino.json"
    instrucao = "Gere a referência com o código original: python scripts/avaliar.py --particao treino --baseline"
    try:
        with caminho.open(encoding="utf-8-sig") as arquivo:
            base = json.load(arquivo)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Baseline de treino ausente ou ilegível: {caminho}. {instrucao}") from exc
    if not isinstance(base, dict) or base.get("status") == "pendente":
        raise ValueError(f"Baseline de treino pendente ou inválido: {caminho}. {instrucao}")
    if base.get("particao") != "treino":
        raise ValueError("Baseline deve pertencer à partição treino.")
    if "dataset" in base and Path(base["dataset"]).resolve() != Path(raiz).resolve():
        raise ValueError("Baseline de treino pertence a outro dataset.")
    atuais = {k: v for k, v in asdict(Parametros()).items() if k.startswith("sev_")}
    params = base.get("params")
    salvos = {k: v for k, v in params.items() if k.startswith("sev_")} if isinstance(params, dict) else {}
    # Ausências e chaves antigas também são incompatíveis: não inferir limiares.
    divergentes = sorted(k for k in atuais.keys() | salvos.keys()
                         if k not in atuais or k not in salvos or atuais[k] != salvos[k])
    if divergentes:
        raise ValueError("Baseline de treino incompatível: parâmetros de severidade "
                         "divergentes ou ausentes: " + ", ".join(divergentes) + f". {instrucao}")
    # Comparar consigo mesmo reprova a duplicação de buracos por definição;
    # aqui interessa apenas que todas as métricas sejam presentes e finitas.
    verificacao = avaliar_criterios(base, base)
    if any("baseline" not in c for c in verificacao["criterios"].values()):
        raise ValueError(f"Baseline de treino incompleto ou inválido: {caminho}. {instrucao}")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imagens", type=int, default=80)
    ap.add_argument("--passadas", type=int, default=1)
    ap.add_argument("--params", nargs="*", default=None, help="restringe a busca a estes parâmetros")
    a = ap.parse_args()
    if a.n_imagens <= 0 or a.passadas < 0:
        ap.error("--n-imagens deve ser positivo e --passadas não negativo")
    if a.params and any(k not in GRADE for k in a.params):
        ap.error("Parâmetro fora da grade permitida (severidade/pesos não são ajustáveis)")
    p = Parametros.carregar(_caminho.PARAMS) if os.path.exists(_caminho.PARAMS) else Parametros()
    raiz = _caminho.dataset_atual()
    pastas = [d for d in listar_amostras(raiz) if particao(os.path.basename(d)) == "treino"]
    somente_treino(pastas)
    validar_amostras(pastas)
    # Critérios usam a referência original congelada e TODO o treino final.
    # Reexecutar a busca não aumenta o alvo nem substitui o baseline salvo.
    todas_treino = pastas
    baseline = carregar_baseline(raiz)
    pastas = pastas[::max(1, len(pastas) // a.n_imagens)][:a.n_imagens]
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=_init, initargs=(pastas,)) as pool:
        melhor, info = pool.apply(_nota, (asdict(p),))
        print(f"inicial {melhor:.4f} {info}", flush=True)
        for _ in range(a.passadas):
            for k in (a.params or GRADE):
                cands = [replace(p, **{k: v}) for v in GRADE[k] if v != getattr(p, k)]
                notas = pool.map(_nota, [asdict(c) for c in cands])
                for c, (n, inf) in zip(cands, notas):
                    if n > melhor + 1e-9:
                        melhor, info, p = n, inf, c
                print(f"{k}={getattr(p, k)}  nota={melhor:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    novo = avaliar_pastas(todas_treino, p, processos=1)
    criterios = avaliar_criterios(baseline, novo)
    print(json.dumps({"criterios_treino": criterios}, indent=2))
    if not criterios["aprovado"]:
        raise ValueError("Parâmetros reprovados nos critérios de treino; params.json não foi gravado.")
    p.salvar(_caminho.PARAMS)
    print(json.dumps({"nota": melhor, **info, "params": asdict(p)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}")
