"""Preparo validado, cache com origem e avaliação compartilhada."""
import json
import os
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np

import _caminho
from pavimento.dataset import Amostra, CLASSES, particao
from pavimento.metricas import confusao
from pavimento.parametros import REFERENCIA_SEVERIDADE
from pavimento.nucleo import afinar, analisar, indice_dano, intensidade_hsi, reduzir_2x, reduzir_mascara_2x, severidade

CACHE_VERSAO = 2


def arquivos_entrada(pasta):
    pasta = Path(pasta).resolve()
    nome = pasta.name
    # README: RAW + CRACK/POTHOLE/LANE. Nenhuma máscara é opcional;
    # ausência de dano deve ser representada por um PNG válido vazio.
    return [pasta / f"{nome}_RAW.jpg"] + [pasta / f"{nome}_{c}.png" for c in CLASSES]


def manifesto(pasta):
    textual = Path(pasta)
    treino = particao(textual.name) == "treino"
    if treino:
        somente_treino([textual])
    pasta = textual.resolve()
    arquivos = []
    for f in arquivos_entrada(pasta):
        origem = f.resolve()
        # Uma pasta real de treino também pode conter links para RAW/rótulos
        # do teste. Validar cada destino antes da leitura e do uso do cache.
        # Amostras de teste continuam permitidas na avaliação final.
        if treino:
            try:
                somente_treino([origem.parent])
            except ValueError as exc:
                raise ValueError(
                    f"Amostra {textual.name}: entrada {f} com origem proibida: {origem}. {exc}"
                ) from exc
        if not origem.is_file():
            raise ValueError(f"Amostra {pasta.name}: arquivo obrigatório ausente: {f}")
        st = origem.stat()
        arquivos.append({"entrada": str(f), "caminho": str(origem),
                         "tamanho": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {"versao": CACHE_VERSAO, "dataset": str(pasta.parent), "amostra": str(pasta),
            "particao": particao(pasta.name), "arquivos": arquivos}


def carregar(pasta):
    manifesto(pasta)
    fs = arquivos_entrada(pasta)
    img = cv2.imread(str(fs[0]), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Amostra {Path(pasta).name}: RAW ilegível: {fs[0]}")
    masks = {}
    for classe, f in zip(CLASSES, fs[1:]):
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != img.shape[:2]:
            raise ValueError(f"Amostra {Path(pasta).name}: máscara ilegível ou dimensões incompatíveis: {f}")
        masks[classe] = m > 127
    return Amostra(Path(pasta).name, img, masks)


def validar_amostras(pastas):
    """Valida o lote inteiro antes do cálculo, inclusive entradas já em cache."""
    pastas = list(pastas)
    if not pastas:
        raise ValueError("Nenhuma amostra encontrada no dataset/partição solicitada.")
    erros = []
    for pasta in pastas:
        try:
            carregar(pasta)
        except (ValueError, OSError) as exc:
            erros.append(str(exc))
    if erros:
        raise ValueError("Amostras inválidas:\n" + "\n".join(erros))
    return pastas


def somente_treino(pastas):
    pastas = list(pastas)
    proibidas = []
    for pasta in pastas:
        # Conferir tanto o nome apresentado quanto a origem efetivamente lida
        # por carregar/manifesto: links e junctions não mudam a partição real.
        textual = Path(pasta)
        resolvida = textual.resolve()
        if any(particao(p.name) != "treino"
               or any(x.lower() == "teste" for x in p.parts)
               for p in (textual, resolvida)):
            proibidas.append(f"{textual} (origem: {resolvida})")
    if proibidas:
        raise ValueError("Vazamento recusado: somente treino é permitido; pastas de teste: " + ", ".join(proibidas))
    return pastas


def preparar(pasta):
    """Cache só é reutilizado se origem, versão e todos os arquivos coincidirem."""
    atual = manifesto(pasta)  # Nunca retorna cache se falta uma entrada.
    nome = Path(pasta).name
    f = os.path.join(_caminho.CACHE, nome + ".npz")
    if os.path.isfile(f):
        try:
            with np.load(f, allow_pickle=False) as z:
                if json.loads(str(z["manifesto"].item())) == atual:
                    dados = tuple(z[k] for k in ("g", "roi", "gt", "gb"))
                    if (dados[0].ndim == 2 and all(x.shape == dados[0].shape for x in dados)
                            and dados[0].dtype == np.uint8
                            and all(x.dtype == np.bool_ for x in dados[1:])):
                        return (nome, *dados)
        except (ValueError, KeyError, OSError, EOFError, BadZipFile):
            pass  # Cache antigo/corrompido não é fonte de verdade.
    s = carregar(pasta)
    g = reduzir_2x(intensidade_hsi(s.imagem))
    roi = reduzir_mascara_2x(s.mascaras["LANE"], "maioria")
    gt = reduzir_mascara_2x(s.mascaras["CRACK"], "qualquer")
    gb = reduzir_mascara_2x(s.mascaras["POTHOLE"], "qualquer")
    if manifesto(pasta) != atual:
        raise ValueError(f"Amostra {nome}: arquivos alterados durante a leitura; repita o preparo.")
    os.makedirs(_caminho.CACHE, exist_ok=True)
    np.savez_compressed(f, g=g, roi=roi, gt=gt, gb=gb, manifesto=json.dumps(atual, sort_keys=True))
    return nome, g, roi, gt, gb


def avaliar_amostra(dados, p, tol=2):
    nome, g, roi, gt, gb = dados
    r = analisar(g, roi, p)
    return {"nome": nome,
            "trinca_px": confusao(r["trincas"], gt, roi),
            "trinca_tol": confusao(r["trincas"], gt, roi, tolerancia=tol),
            "buraco_px": confusao(r["buracos"], gb, roi),
            "tem_trinca_gt": bool(gt.sum() >= 15), "tem_trinca_pred": r["quant"]["n_trincas"] > 0,
            "tem_buraco_gt": bool(gb.sum() >= 75), "tem_buraco_pred": r["quant"]["n_buracos"] > 0,
            "quant": r["quant"]}


def referencia_rotulo(dados, p=None):
    """Verdade congelada; p é aceito por compatibilidade, mas nunca consultado."""
    _, _, roi, gt, gb = dados
    comp, area_b, n = int(afinar(gt & roi).sum()), int((gb & roi).sum()), int(roi.sum())
    ref = REFERENCIA_SEVERIDADE
    ind = indice_dano(comp, area_b, n, ref.dano_peso_trinca, ref.dano_peso_buraco)
    return {"comprimento_trincas_px": comp, "area_buracos_px": area_b, "indice_dano_1000": ind,
            "severidade": severidade(ind, ref)}
