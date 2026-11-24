"""Gerenciamento de entradas (cenários, dataset DNIT, fotos avulsas) e parâmetros."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pavimento.dataset import carregar as carregar_amostra_dnit
from pavimento.dataset import particao as particao_dnit
from pavimento.parametros import Parametros

from .cenarios import carregar_cenario, listar_cenarios, poligono_para_mascara

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_PARAMS = RAIZ_PROJETO / "params.json"

DESTAQUES_DNIT = (
    {
        "nome": "1092481_DF_251_251BDF0052_00028",
        "titulo": "DNIT Real · DF-251 (Couro de Jacaré)",
        "subtitulo": "1024×630 · Gabarito real · Teste",
    },
    {
        "nome": "992851_RS_386_386RS191729_01020",
        "titulo": "DNIT Real · RS-386 (Buraco Grande · IoU 0,71)",
        "subtitulo": "1024×640 · Gabarito real · Teste",
    },
    {
        "nome": "1014586_RS_386_386RS124739_29855",
        "titulo": "DNIT Real · RS-386 (Trincas Moderadas)",
        "subtitulo": "1024×640 · Gabarito real · Teste",
    },
    {
        "nome": "1092981_DF_251_251BDF0052_00911",
        "titulo": "DNIT Real · DF-251 (Rodovia com Sombra)",
        "subtitulo": "1024×630 · Gabarito real · Teste",
    },
)

LIMITES_PARAMS: Dict[str, Tuple[int, int]] = {
    "raio": (1, 20),
    "trinca_a16": (0, 80),
    "trinca_b16": (0, 640),
    "trinca_fraca_a16": (0, 64),
    "trinca_fraca_b16": (0, 320),
    "trinca_abertura_linha": (1, 9),
    "trinca_fecho": (1, 9),
    "trinca_area_min": (1, 300),
    "trinca_comp_min": (1, 150),
    "trinca_larg_max16": (16, 256),
    "trinca_along_min16": (0, 160),
    "jacare_bif_100px": (1, 30),
    "jacare_comp_min": (10, 250),
    "raio_buraco": (1, 20),
    "buraco_modo": (0, 1),
    "buraco_k4": (0, 80),
    "buraco_abertura": (1, 21),
    "buraco_fecho": (1, 21),
    "buraco_area_min": (1, 2000),
    "buraco_circ_min_1000": (0, 950),
    "buraco_escuro_min": (0, 80),
    "buraco_margem_fraca": (0, 80),
    "buraco_margem_forte": (0, 100),
    "buraco_usar_otsu": (0, 1),
    "buraco_extensao_min_1000": (0, 900),
    "buraco_aspecto_max_1000": (1000, 20000),
    "buraco_fracao_escura_min_1000": (0, 1000),
    "buraco_borda_margem": (0, 20),
    "dano_peso_trinca": (0, 200),
    "dano_peso_buraco": (0, 200),
    "sev_min_1000": (0, 900),
    "sev_media_1000": (1, 950),
    "sev_alta_1000": (2, 1000),
}

POLIGONO_PADRAO_FOTO = [(0.22, 0.00), (0.78, 0.00), (0.96, 1.00), (0.04, 1.00)]


@dataclass
class EntradaAtiva:
    tipo: str  # "cenario" | "dnit" | "arquivo"
    identificador: str
    titulo: str
    subtitulo: str
    img_bgr: np.ndarray
    roi_padrao: np.ndarray
    reduzir_padrao: bool
    poligono_inicial: List[Tuple[float, float]]
    gt_trinca: Optional[np.ndarray] = None
    gt_buraco: Optional[np.ndarray] = None
    tem_roi_propria: bool = True


_DATASET_PASTA_CUSTOM: Optional[Path] = None
_DATASET_CACHE: Optional[Tuple[Path, List[dict]]] = None


def definir_pasta_dataset(caminho: Optional[Path]) -> None:
    global _DATASET_PASTA_CUSTOM, _DATASET_CACHE
    _DATASET_PASTA_CUSTOM = caminho.resolve() if caminho is not None else None
    _DATASET_CACHE = None


def pasta_dataset_dnit() -> Optional[Path]:
    if _DATASET_PASTA_CUSTOM is not None and _DATASET_PASTA_CUSTOM.is_dir():
        return _DATASET_PASTA_CUSTOM
    for env_key in ("CAVALO_DATASET", "PAVIMENTO_DATASET"):
        val = os.environ.get(env_key)
        if val:
            cand = Path(val).expanduser()
            if cand.is_dir():
                return cand.resolve()
    padrao = RAIZ_PROJETO / "dataset" / "Dataset" / "v1"
    if padrao.is_dir():
        return padrao.resolve()
    return None


def indice_dataset_dnit() -> Tuple[Optional[Path], List[dict]]:
    global _DATASET_CACHE
    pasta = pasta_dataset_dnit()
    if pasta is None:
        return None, []
    if _DATASET_CACHE is not None and _DATASET_CACHE[0] == pasta:
        return _DATASET_CACHE

    itens: List[dict] = []
    for entry in sorted(os.scandir(pasta), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        nome = entry.name
        if not (Path(entry.path) / f"{nome}_RAW.jpg").is_file():
            continue
        partes = nome.split("_")
        estado = partes[1] if len(partes) > 1 and len(partes[1]) == 2 else "OUTRO"
        rodovia = f"{estado}-{partes[2]}" if len(partes) > 2 else estado
        itens.append({
            "nome": nome,
            "particao": particao_dnit(nome),
            "estado": estado,
            "rodovia": rodovia,
            "caminho": str(Path(entry.path).resolve()),
        })
    _DATASET_CACHE = (pasta, itens)
    return _DATASET_CACHE


def resumo_dataset_dnit() -> dict:
    pasta, itens = indice_dataset_dnit()
    if pasta is None or not itens:
        return {"disponivel": False, "pasta": None, "total": 0, "n_teste": 0, "n_treino": 0, "destaques": [], "estados": []}
    nomes_set = {it["nome"] for it in itens}
    destaques = [d for d in DESTAQUES_DNIT if d["nome"] in nomes_set]
    estados = sorted({it["estado"] for it in itens})
    n_teste = sum(1 for it in itens if it["particao"] == "teste")
    return {
        "disponivel": True,
        "pasta": str(pasta),
        "total": len(itens),
        "n_teste": n_teste,
        "n_treino": len(itens) - n_teste,
        "destaques": destaques,
        "estados": estados,
    }


def filtrar_dataset_dnit(
    busca: str = "",
    particao_f: str = "todas",
    estado_f: str = "todos",
) -> List[dict]:
    _, itens = indice_dataset_dnit()
    q = busca.strip().lower()
    return [
        it for it in itens
        if (particao_f == "todas" or it["particao"] == particao_f)
        and (estado_f == "todos" or it["estado"] == estado_f)
        and (not q or q in it["nome"].lower() or q in it["rodovia"].lower())
    ]


def carregar_entrada_cenario(cenario_id: str) -> EntradaAtiva:
    bgr, roi, info = carregar_cenario(cenario_id)
    return EntradaAtiva(
        tipo="cenario",
        identificador=info.id,
        titulo=info.titulo,
        subtitulo=info.subtitulo,
        img_bgr=bgr,
        roi_padrao=roi,
        reduzir_padrao=info.reduzir_padrao,
        poligono_inicial=[(float(px), float(py)) for px, py in info.roi_poligono],
        gt_trinca=None,
        gt_buraco=None,
        tem_roi_propria=True,
    )


def carregar_entrada_dnit(pasta_ou_nome: str) -> EntradaAtiva:
    cand = Path(pasta_ou_nome)
    if not cand.is_dir():
        pasta_ds = pasta_dataset_dnit()
        if pasta_ds is None:
            raise FileNotFoundError("Dataset DNIT não localizado.")
        cand = (pasta_ds / Path(pasta_ou_nome).name).resolve()
    amostra = carregar_amostra_dnit(str(cand))
    h, w = amostra.imagem.shape[:2]
    part = particao_dnit(amostra.nome)
    return EntradaAtiva(
        tipo="dnit",
        identificador=amostra.nome,
        titulo=f"DNIT · {amostra.nome}",
        subtitulo=f"{w}×{h} RGB · Gabarito real ({part})",
        img_bgr=amostra.imagem,
        roi_padrao=amostra.mascaras["LANE"],
        reduzir_padrao=True,
        poligono_inicial=list(POLIGONO_PADRAO_FOTO),
        gt_trinca=amostra.mascaras["CRACK"],
        gt_buraco=amostra.mascaras["POTHOLE"],
        tem_roi_propria=True,
    )


def carregar_entrada_arquivo(caminho_img: str, caminho_roi: Optional[str] = None) -> EntradaAtiva:
    arq = Path(caminho_img).expanduser().resolve()
    # Se o usuário abriu *_RAW.jpg de uma amostra completa do DNIT, carrega automaticamente o gabarito!
    if arq.name.endswith("_RAW.jpg") and caminho_roi is None:
        prefixo = arq.name[: -len("_RAW.jpg")]
        pasta_pai = arq.parent
        if all((pasta_pai / f"{prefixo}_{suf}.png").is_file() for suf in ("LANE", "CRACK", "POTHOLE")):
            return carregar_entrada_dnit(str(pasta_pai))

    buf = np.fromfile(str(arq), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível abrir a imagem: {arq.name}")

    h, w = img.shape[:2]
    maior = max(h, w)
    if maior > 1280:
        escala = 1280.0 / maior
        nw = max(64, (int(round(w * escala)) // 2) * 2)
        nh = max(64, (int(round(h * escala)) // 2) * 2)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    tem_roi = False
    if caminho_roi:
        arq_roi = Path(caminho_roi).expanduser().resolve()
        buf_r = np.fromfile(str(arq_roi), dtype=np.uint8)
        rm = cv2.imdecode(buf_r, cv2.IMREAD_GRAYSCALE)
        if rm is None:
            raise ValueError(f"Máscara ROI inválida: {arq_roi.name}")
        if rm.shape[:2] != (h, w):
            rm = cv2.resize(rm, (w, h), interpolation=cv2.INTER_NEAREST)
        roi = rm > 127
        tem_roi = True
    else:
        roi = np.ones((h, w), dtype=bool)

    return EntradaAtiva(
        tipo="arquivo",
        identificador=str(arq),
        titulo=f"Foto · {arq.name}",
        subtitulo=f"{w}×{h} RGB" + (" · Com máscara ROI" if tem_roi else " · Imagem inteira"),
        img_bgr=img,
        roi_padrao=roi,
        reduzir_padrao=max(h, w) > 640,
        poligono_inicial=list(POLIGONO_PADRAO_FOTO),
        gt_trinca=None,
        gt_buraco=None,
        tem_roi_propria=tem_roi,
    )


def resolver_roi(
    entrada: EntradaAtiva,
    roi_modo: str,
    poligono: List[Tuple[float, float]],
) -> np.ndarray:
    h, w = entrada.img_bgr.shape[:2]
    if roi_modo == "inteira":
        return np.ones((h, w), dtype=bool)
    if roi_modo == "poligono" and len(poligono) >= 3:
        return poligono_para_mascara((h, w), poligono)
    return entrada.roi_padrao


def carregar_presets_params() -> Dict[str, dict]:
    padrao = Parametros()
    final = Parametros.carregar(str(ARQUIVO_PARAMS)) if ARQUIVO_PARAMS.is_file() else padrao
    slides_anterior = asdict(final)
    slides_anterior.update({
        "raio": 4,
        "trinca_a16": 32,
        "trinca_b16": 150,
        "trinca_area_min": 19,
        "dano_peso_trinca": 60,
        "sev_min_1000": 0,
        "sev_media_1000": 109,
        "sev_alta_1000": 535,
    })
    return {
        "final": asdict(final),
        "slides_anterior": slides_anterior,
        "padrao": asdict(padrao),
    }


def construir_parametros(bruto: Optional[dict]) -> Parametros:
    base = Parametros.carregar(str(ARQUIVO_PARAMS)) if ARQUIVO_PARAMS.is_file() else Parametros()
    if not bruto:
        return base
    valores = asdict(base)
    nomes_validos = {f.name for f in fields(Parametros)}
    for k, v in bruto.items():
        if k not in nomes_validos:
            continue
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            continue
        if k in LIMITES_PARAMS:
            lo, hi = LIMITES_PARAMS[k]
            iv = max(lo, min(hi, iv))
        valores[k] = iv
    return Parametros(**valores)


def montar_relatorio_json(entrada: EntradaAtiva, resultado: dict) -> dict:
    """Extrai um dicionário limpo e serializável em JSON para exportação."""
    return {
        "entrada": {
            "tipo": entrada.tipo,
            "identificador": entrada.identificador,
            "titulo": entrada.titulo,
        },
        "dimensoes": resultado["dimensoes"],
        "tempo_nucleo_ms": resultado["tempo_nucleo_ms"],
        "quantificacao": resultado["quant"],
        "formula_dano": resultado["formula_dano"],
        "estatisticas": resultado["estatisticas"],
        "metricas_gabarito_dnit": resultado["metricas_gt"],
        "parametros": resultado["params"],
        "regioes_trinca": resultado["regioes_trinca"],
        "regioes_buraco": resultado["regioes_buraco"],
    }


__all__ = [
    "ARQUIVO_PARAMS",
    "DESTAQUES_DNIT",
    "EntradaAtiva",
    "LIMITES_PARAMS",
    "RAIZ_PROJETO",
    "carregar_entrada_arquivo",
    "carregar_entrada_cenario",
    "carregar_entrada_dnit",
    "carregar_presets_params",
    "construir_parametros",
    "definir_pasta_dataset",
    "filtrar_dataset_dnit",
    "indice_dataset_dnit",
    "listar_cenarios",
    "montar_relatorio_json",
    "pasta_dataset_dnit",
    "resolver_roi",
    "resumo_dataset_dnit",
]
