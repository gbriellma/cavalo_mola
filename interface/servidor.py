"""Servidor HTTP local em Python puro (stdlib) para a interface web do Sistema de Detecção de Danos em Pavimentos."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import threading
import uuid
import webbrowser
from dataclasses import asdict, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from pavimento.dataset import carregar as carregar_amostra_dnit
from pavimento.dataset import particao as particao_dnit
from pavimento.parametros import Parametros

from .cenarios import CENARIOS_POR_ID, carregar_cenario, listar_cenarios, poligono_para_mascara
from .pipeline_didatico import executar_pipeline_didatico, gerar_painel_oficial_png

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_STATIC = Path(__file__).resolve().parent / "static"
ARQUIVO_PARAMS = RAIZ_PROJETO / "params.json"

DESTAQUES_DNIT = (
    {
        "nome": "1014586_RS_386_386RS124739_29855",
        "titulo": "RS-386 · Trincas Moderadas",
        "subtitulo": "1024×640 · ROI + gabarito reais · Partição teste",
    },
    {
        "nome": "1092481_DF_251_251BDF0052_00028",
        "titulo": "DF-251 · Trincas Severas (Couro de Jacaré)",
        "subtitulo": "1024×630 · ROI + gabarito reais · Partição teste",
    },
    {
        "nome": "992851_RS_386_386RS191729_01020",
        "titulo": "RS-386 · Buraco Grande (IoU 0,71)",
        "subtitulo": "1024×640 · ROI + gabarito reais · Partição teste",
    },
    {
        "nome": "1092981_DF_251_251BDF0052_00911",
        "titulo": "DF-251 · Rodovia Real (Exemplo)",
        "subtitulo": "1024×630 · ROI + gabarito reais · Partição teste",
    },
)

_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: Optional[Tuple[Path, List[dict]]] = None


def _pasta_dataset() -> Optional[Path]:
    cand = Path(os.environ.get("PAVIMENTO_DATASET", str(RAIZ_PROJETO / "dataset" / "Dataset" / "v1"))).expanduser()
    if cand.is_dir():
        return cand.resolve()
    return None


def _indice_dataset() -> Tuple[Optional[Path], List[dict]]:
    global _DATASET_CACHE
    pasta = _pasta_dataset()
    if pasta is None:
        return None, []
    with _DATASET_CACHE_LOCK:
        if _DATASET_CACHE is not None and _DATASET_CACHE[0] == pasta:
            return _DATASET_CACHE
        itens: List[dict] = []
        for entry in sorted(os.scandir(pasta), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            nome = entry.name
            partes = nome.split("_")
            estado = partes[1] if len(partes) > 1 and len(partes[1]) == 2 else "OUTRO"
            rodovia = f"{estado}-{partes[2]}" if len(partes) > 2 else estado
            itens.append({
                "nome": nome,
                "particao": particao_dnit(nome),
                "estado": estado,
                "rodovia": rodovia,
            })
        _DATASET_CACHE = (pasta, itens)
        return _DATASET_CACHE


def info_dataset_init() -> dict:
    pasta, itens = _indice_dataset()
    if pasta is None or not itens:
        return {"disponivel": False, "total": 0, "destaques": [], "estados": []}
    nomes_set = {it["nome"] for it in itens}
    destaques = [d for d in DESTAQUES_DNIT if d["nome"] in nomes_set]
    estados = sorted({it["estado"] for it in itens})
    n_teste = sum(1 for it in itens if it["particao"] == "teste")
    return {
        "disponivel": True,
        "total": len(itens),
        "n_teste": n_teste,
        "n_treino": len(itens) - n_teste,
        "destaques": destaques,
        "estados": estados,
    }


def consultar_dataset(q: str = "", particao_f: str = "todas", estado_f: str = "todos",
                      offset: int = 0, limit: int = 60) -> dict:
    pasta, itens = _indice_dataset()
    if pasta is None:
        return {"disponivel": False, "total": 0, "filtrado": 0, "amostras": []}
    q_low = q.strip().lower()
    filtrados = [
        it for it in itens
        if (particao_f == "todas" or it["particao"] == particao_f)
        and (estado_f == "todos" or it["estado"] == estado_f)
        and (not q_low or q_low in it["nome"].lower() or q_low in it["rodovia"].lower())
    ]
    offset = max(0, offset)
    limit = max(1, min(200, limit))
    return {
        "disponivel": True,
        "total": len(itens),
        "filtrado": len(filtrados),
        "offset": offset,
        "limit": limit,
        "amostras": filtrados[offset:offset + limit],
    }

# Armazenamento em memória das imagens enviadas nesta sessão (evita gravar lixo em disco)
_UPLOADS_LOCK = threading.Lock()
_UPLOADS: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]] = {}
_MAX_UPLOADS = 12


# Limites seguros para cada parâmetro inteiro editável via UI
LIMITES_PARAMS = {
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


def carregar_presets_params() -> dict:
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


def _decodificar_imagem_b64(data_url: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise ValueError("Arquivo de imagem inválido ou formato não suportado.")
    return img


def _resolver_entrada(payload: dict) -> Tuple[np.ndarray, np.ndarray, bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Resolve (img_bgr, roi_full, aplicar_reducao_2x, gt_trinca, gt_buraco)."""
    upload_id = payload.get("upload_id")
    amostra_dnit = payload.get("amostra_dnit")
    cenario_id = payload.get("cenario_id", "rodovia_dnit_01")
    roi_modo = payload.get("roi_modo", "cenario")
    roi_poligono = payload.get("roi_poligono")
    gt_t: Optional[np.ndarray] = None
    gt_b: Optional[np.ndarray] = None

    if amostra_dnit:
        pasta, _ = _indice_dataset()
        if pasta is None:
            raise FileNotFoundError("Dataset DNIT não encontrado localmente.")
        nome_limpo = Path(str(amostra_dnit)).name
        alvo = (pasta / nome_limpo).resolve()
        if alvo.parent != pasta or not alvo.is_dir():
            raise KeyError(f"Amostra DNIT não encontrada: {nome_limpo}")
        amostra = carregar_amostra_dnit(str(alvo))
        img_bgr = amostra.imagem
        roi_padrao = amostra.mascaras["LANE"]
        gt_t = amostra.mascaras["CRACK"]
        gt_b = amostra.mascaras["POTHOLE"]
        h, w = img_bgr.shape[:2]
        reduzir_default = True
    elif upload_id:
        with _UPLOADS_LOCK:
            if upload_id not in _UPLOADS:
                raise KeyError("Imagem enviada expirou; envie novamente.")
            img_bgr, roi_up, _nome = _UPLOADS[upload_id]
        h, w = img_bgr.shape[:2]
        reduzir_default = max(h, w) > 640
        roi_padrao = roi_up if roi_up is not None else np.ones((h, w), dtype=bool)
    else:
        img_bgr, roi_padrao, info = carregar_cenario(cenario_id)
        h, w = img_bgr.shape[:2]
        reduzir_default = info.reduzir_padrao

    if roi_modo == "inteira":
        roi_full = np.ones((h, w), dtype=bool)
    elif roi_modo == "poligono" and isinstance(roi_poligono, list) and len(roi_poligono) >= 3:
        pts = [(float(pt[0]), float(pt[1])) for pt in roi_poligono]
        roi_full = poligono_para_mascara((h, w), pts)
    else:
        roi_full = roi_padrao

    reduzir_2x = payload.get("reduzir_2x")
    if reduzir_2x is None:
        reduzir_2x = reduzir_default
    return img_bgr, roi_full, bool(reduzir_2x), gt_t, gt_b


def registrar_upload(imagem_b64: str, nome: str, roi_b64: Optional[str] = None) -> dict:
    img = _decodificar_imagem_b64(imagem_b64, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # Fotos muito grandes de celular são reescaladas para teto de 1280 px mantendo proporção
    maior = max(h, w)
    if maior > 1280:
        escala = 1280.0 / maior
        nw, nh = max(64, int(round(w * escala)) // 2 * 2), max(64, int(round(h * escala)) // 2 * 2)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    roi_mask: Optional[np.ndarray] = None
    if roi_b64:
        rm = _decodificar_imagem_b64(roi_b64, cv2.IMREAD_GRAYSCALE)
        if rm.shape[:2] != (h, w):
            rm = cv2.resize(rm, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_mask = rm > 127

    uid = uuid.uuid4().hex[:12]
    with _UPLOADS_LOCK:
        if len(_UPLOADS) >= _MAX_UPLOADS:
            primeiro = next(iter(_UPLOADS))
            del _UPLOADS[primeiro]
        _UPLOADS[uid] = (img, roi_mask, nome or "foto_enviada.png")

    return {
        "upload_id": uid,
        "nome": nome or "foto_enviada.png",
        "largura": w,
        "altura": h,
        "reduzir_padrao": max(h, w) > 640,
        "tem_roi_upload": roi_mask is not None,
    }


def atualizar_roi_upload(upload_id: str, roi_b64: str) -> dict:
    with _UPLOADS_LOCK:
        if upload_id not in _UPLOADS:
            raise KeyError("Imagem enviada expirou; envie a foto novamente.")
        img, _old_roi, nome = _UPLOADS[upload_id]
        h, w = img.shape[:2]
        rm = _decodificar_imagem_b64(roi_b64, cv2.IMREAD_GRAYSCALE)
        if rm.shape[:2] != (h, w):
            rm = cv2.resize(rm, (w, h), interpolation=cv2.INTER_NEAREST)
        _UPLOADS[upload_id] = (img, rm > 127, nome)
    return {"upload_id": upload_id, "tem_roi_upload": True}


METRICAS_OFICIAIS_TESTE = [
    {"metrica": "Trincas — Detecção por imagem (F1)", "inicial": "0,927", "anterior": "0,928", "final": "0,928", "destaque": "Revocação de 100,0% (0 falsos negativos)"},
    {"metrica": "Trincas — Pixel tol. 2 px (F1)", "inicial": "0,321", "anterior": "0,430", "final": "0,531", "destaque": "+66% sobre o inicial"},
    {"metrica": "Trincas — Pixel exato (F1 / IoU)", "inicial": "0,200 / 0,111", "anterior": "0,234 / 0,132", "final": "0,326 / 0,194", "destaque": "+63% em F1 exato"},
    {"metrica": "Comprimento de trincas (Spearman)", "inicial": "0,554", "anterior": "0,683", "final": "0,712", "destaque": "Forte ordenação entre vias"},
    {"metrica": "Buracos — Detecção por imagem (F1)", "inicial": "0,330", "anterior": "0,380", "final": "0,380", "destaque": "Revocação 33,8% → 53,8%"},
    {"metrica": "Buracos — Pixel (F1 / IoU)", "inicial": "0,085 / 0,044", "anterior": "0,112 / 0,059", "final": "0,112 / 0,059", "destaque": "Buracos = 0,69% dos pixels"},
    {"metrica": "Severidade — Acerto exato (4 classes)", "inicial": "48,5%", "anterior": "56,3%", "final": "57,4%", "destaque": "Acaso = 25% (4 classes)"},
    {"metrica": "Severidade — Erro de até 1 nível", "inicial": "90,1%", "anterior": "94,1%", "final": "93,2%", "destaque": "Triagem consistente"},
]


class PavimentoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        # Log conciso apenas para erros HTTP >= 400
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)

    def _enviar_bytes(self, corpo: bytes, content_type: str, status: int = 200,
                      extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(corpo)

    def _enviar_json(self, obj: dict, status: int = 200) -> None:
        dados = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._enviar_bytes(dados, "application/json; charset=utf-8", status=status)

    def _ler_json(self) -> dict:
        tam = int(self.headers.get("Content-Length", "0"))
        if tam <= 0:
            return {}
        if tam > 32 * 1024 * 1024:
            raise ValueError("Requisição excede o limite de 32 MB.")
        bruto = self.rfile.read(tam)
        return json.loads(bruto.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        caminho = parsed.path

        if caminho in ("/", "/index.html"):
            arq = PASTA_STATIC / "index.html"
            self._enviar_bytes(arq.read_bytes(), "text/html; charset=utf-8")
            return

        if caminho.startswith("/static/"):
            rel = caminho[len("/static/"):]
            alvo = (PASTA_STATIC / rel).resolve()
            if not str(alvo).startswith(str(PASTA_STATIC.resolve())) or not alvo.is_file():
                self._enviar_json({"erro": "Arquivo não encontrado"}, status=HTTPStatus.NOT_FOUND)
                return
            mime, _ = mimetypes.guess_type(str(alvo))
            if alvo.suffix == ".js":
                mime = "application/javascript; charset=utf-8"
            elif alvo.suffix == ".css":
                mime = "text/css; charset=utf-8"
            self._enviar_bytes(alvo.read_bytes(), mime or "application/octet-stream")
            return

        if caminho == "/api/init":
            self._enviar_json({
                "cenarios": listar_cenarios(),
                "dataset": info_dataset_init(),
                "presets": carregar_presets_params(),
                "metricas_oficiais": METRICAS_OFICIAIS_TESTE,
            })
            return

        if caminho == "/api/dataset":
            qs = parse_qs(parsed.query)
            q = qs.get("q", [""])[0]
            part = qs.get("particao", ["todas"])[0]
            est = qs.get("estado", ["todos"])[0]
            try:
                off = int(qs.get("offset", ["0"])[0])
                lim = int(qs.get("limit", ["60"])[0])
            except ValueError:
                off, lim = 0, 60
            self._enviar_json(consultar_dataset(q, part, est, off, lim))
            return

        self._enviar_json({"erro": "Rota não encontrada"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        caminho = parsed.path

        try:
            payload = self._ler_json()

            if caminho == "/api/analisar":
                img_bgr, roi_full, reduzir, gt_t, gt_b = _resolver_entrada(payload)
                p = construir_parametros(payload.get("params"))
                resultado = executar_pipeline_didatico(
                    img_bgr, roi_full, p, aplicar_reducao_2x=reduzir,
                    gt_trinca_full=gt_t, gt_buraco_full=gt_b,
                )
                self._enviar_json(resultado)
                return

            if caminho == "/api/upload":
                imagem_b64 = payload.get("imagem_b64")
                if not imagem_b64:
                    raise ValueError("Campo imagem_b64 obrigatório.")
                info = registrar_upload(imagem_b64, payload.get("nome", "foto.png"), payload.get("roi_b64"))
                self._enviar_json(info)
                return

            if caminho == "/api/upload_roi":
                uid = payload.get("upload_id")
                roi_b64 = payload.get("roi_b64")
                if not uid or not roi_b64:
                    raise ValueError("Campos upload_id e roi_b64 obrigatórios.")
                self._enviar_json(atualizar_roi_upload(uid, roi_b64))
                return

            if caminho == "/api/exportar_painel":
                img_bgr, roi_full, reduzir, gt_t, gt_b = _resolver_entrada(payload)
                p = construir_parametros(payload.get("params"))
                png_bytes = gerar_painel_oficial_png(
                    img_bgr, roi_full, p, aplicar_reducao_2x=reduzir,
                    gt_trinca_full=gt_t, gt_buraco_full=gt_b,
                )
                self._enviar_bytes(
                    png_bytes,
                    "image/png",
                    extra_headers={"Content-Disposition": 'attachment; filename="pavimento_painel.png"'},
                )
                return

            self._enviar_json({"erro": "Rota não encontrada"}, status=HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._enviar_json({"erro": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._enviar_json({"erro": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self._enviar_json({"erro": f"Erro interno: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def _escolher_porta(host: str, porta_inicial: int) -> int:
    for porta in range(porta_inicial, porta_inicial + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, porta))
                return porta
            except OSError:
                continue
    return 0


def criar_servidor(host: str = "127.0.0.1", porta: int = 8080) -> ThreadingHTTPServer:
    porta_livre = _escolher_porta(host, porta)
    return ThreadingHTTPServer((host, porta_livre), PavimentoHandler)


def iniciar(host: str = "127.0.0.1", porta: int = 8080, abrir_navegador: bool = True) -> None:
    srv = criar_servidor(host, porta)
    porta_real = srv.server_address[1]
    url = f"http://{host}:{porta_real}"
    print("=" * 68)
    print("  Detecção de Danos em Pavimentos — Interface Web (PI 2026.2 · UFRPE)")
    print("=" * 68)
    print(f"  Servidor local ativo em: {url}")
    print("  Pressione Ctrl+C no terminal para encerrar.")
    print("=" * 68, flush=True)
    if abrir_navegador:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
    finally:
        srv.server_close()
