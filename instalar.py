#!/usr/bin/env python3
"""Instalador completo do Sistema de Detecção e Quantificação de Danos em Pavimentos.

Compatível com Linux, macOS e Windows usando apenas a biblioteca padrão do Python 3.
Cada etapa imprime no terminal o que está verificando, baixando ou configurando.

Exemplos de uso:
  python3 instalar.py                # instala .venv, dependências, dataset DNIT e roda testes
  python3 instalar.py --abrir        # instala tudo e abre a interface web ao final
  python3 instalar.py --sem-dataset  # instala apenas código e dependências (sem baixar ~220 MB)
  python3 instalar.py --zip-local ~/Downloads/t576ydh9v8-4.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Fontes oficiais do dataset público "Cracks and Potholes in Road Images" (DNIT)
# Passos et al., Mendeley Data v4, 2020 — DOI: 10.17632/t576ydh9v8.4 (CC BY 4.0)
# ---------------------------------------------------------------------------
URL_DATASET_ZIP = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/t576ydh9v8-4.zip"
URL_MENDELEY_API = "https://data.mendeley.com/public-api/datasets/t576ydh9v8/files?version=4"
URL_PAGINA_DATASET = "https://data.mendeley.com/datasets/t576ydh9v8/4"

VERSAO_PYTHON_MIN = (3, 10)
ARQUIVOS_ESSENCIAIS = (
    "requirements.txt",
    "params.json",
    "src/pavimento/nucleo.py",
    "scripts/interface_visual.py",
)


def _banner(numero: int, total: int, titulo: str, explicacao: str) -> None:
    """Imprime cabeçalho padronizado deixando claro o objetivo da etapa atual."""
    linha = "=" * 72
    print(f"\n{linha}")
    print(f"  ETAPA {numero}/{total} — {titulo}")
    print(linha)
    print(f"  O que faz: {explicacao}", flush=True)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"  [..] {msg}", flush=True)


def _aviso(msg: str) -> None:
    print(f"  [AVISO] {msg}", flush=True)


def _python_do_venv(pasta_venv: Path) -> Path:
    """Resolve o executável Python dentro da virtualenv em Linux/macOS ou Windows."""
    if os.name == "nt":
        return pasta_venv / "Scripts" / "python.exe"
    return pasta_venv / "bin" / "python"


# ---------------------------------------------------------------------------
# ETAPA 1: Verificação do Python e da integridade do repositório
# ---------------------------------------------------------------------------
def etapa_1_verificar_ambiente(raiz: Path) -> None:
    _banner(
        1,
        5,
        "Verificar ambiente Python e arquivos do projeto",
        "Confere se o Python tem versão >= 3.10 e se o código-fonte essencial está íntegro.",
    )
    v = sys.version_info
    if (v.major, v.minor) < VERSAO_PYTHON_MIN:
        raise RuntimeError(
            f"Python {v.major}.{v.minor} encontrado, mas o projeto exige Python "
            f"{VERSAO_PYTHON_MIN[0]}.{VERSAO_PYTHON_MIN[1]} ou superior."
        )
    _ok(f"Python {v.major}.{v.minor}.{v.micro} compatível ({sys.executable}).")

    faltando = [rel for rel in ARQUIVOS_ESSENCIAIS if not (raiz / rel).is_file()]
    if faltando:
        raise FileNotFoundError(
            "Arquivos essenciais não encontrados no diretório do projeto: "
            + ", ".join(faltando)
        )
    _ok(f"Estrutura do repositório validada em: {raiz}")


# ---------------------------------------------------------------------------
# ETAPA 2: Criação do ambiente virtual (.venv)
# ---------------------------------------------------------------------------
def etapa_2_criar_ambiente_virtual(raiz: Path, recriar: bool = False) -> Path:
    _banner(
        2,
        5,
        "Preparar ambiente virtual isolado (.venv)",
        "Cria uma pasta .venv própria para não misturar pacotes do projeto com o Python do sistema.",
    )
    pasta_venv = raiz / ".venv"
    python_venv = _python_do_venv(pasta_venv)

    if recriar and pasta_venv.exists():
        _info("Removendo ambiente virtual anterior (--recriar-venv)...")
        shutil.rmtree(pasta_venv)

    if python_venv.is_file():
        _ok(f"Ambiente virtual já existente reutilizado: {python_venv}")
        return python_venv

    _info(f"Criando ambiente virtual em {pasta_venv}...")
    subprocess.run([sys.executable, "-m", "venv", str(pasta_venv)], check=True)
    if not python_venv.is_file():
        raise RuntimeError(f"Não foi possível localizar o executável em {python_venv}")
    _ok(f"Ambiente virtual criado com sucesso: {python_venv}")
    return python_venv


# ---------------------------------------------------------------------------
# ETAPA 3: Instalação das dependências (NumPy e OpenCV)
# ---------------------------------------------------------------------------
def etapa_3_instalar_dependencias(
    raiz: Path, python_venv: Path, forcar_pip: bool = False
) -> None:
    _banner(
        3,
        5,
        "Instalar dependências Python (NumPy e OpenCV)",
        "Garante que NumPy (cálculo inteiro do núcleo) e OpenCV (leitura/exibição) estejam instalados no .venv.",
    )
    if not forcar_pip:
        checagem = subprocess.run(
            [str(python_venv), "-c", "import numpy, cv2; print(numpy.__version__, cv2.__version__)"],
            capture_output=True,
            text=True,
        )
        if checagem.returncode == 0:
            versoes = checagem.stdout.strip().split()
            np_v = versoes[0] if len(versoes) > 0 else "?"
            cv_v = versoes[1] if len(versoes) > 1 else "?"
            _ok(f"Dependências já instaladas no .venv (NumPy {np_v}, OpenCV {cv_v}) — pip pulado.")
            return

    req_file = raiz / "requirements.txt"
    _info("Atualizando pip dentro do .venv...")
    subprocess.run(
        [str(python_venv), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=str(raiz),
        check=True,
    )
    _info(f"Instalando pacotes listados em {req_file.name}...")
    subprocess.run(
        [str(python_venv), "-m", "pip", "install", "-r", str(req_file)],
        cwd=str(raiz),
        check=True,
    )
    _ok("Dependências instaladas com sucesso no ambiente virtual.")


# ---------------------------------------------------------------------------
# ETAPA 4: Download, extração segura e verificação do Dataset DNIT
# ---------------------------------------------------------------------------
def extrair_zip_seguro(caminho_zip: Path, destino: Path) -> None:
    """Extrai um arquivo .zip validando todos os caminhos contra path traversal (zip slip)."""
    destino_abs = destino.resolve()
    destino_abs.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(caminho_zip, "r") as zf:
        membros = zf.infolist()
        total = max(len(membros), 1)
        for idx, info in enumerate(membros, start=1):
            alvo = (destino_abs / info.filename).resolve()
            if alvo != destino_abs and destino_abs not in alvo.parents:
                raise ValueError(f"Entrada insegura no arquivo zip: {info.filename}")
            if info.is_dir():
                alvo.mkdir(parents=True, exist_ok=True)
            else:
                alvo.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as origem, open(alvo, "wb") as saida:
                    shutil.copyfileobj(origem, saida)
            if total > 50 and (idx % 1000 == 0 or idx == total):
                pct = (100 * idx) // total
                print(f"\r  [..] Extraindo {caminho_zip.name}: {idx}/{total} itens ({pct}%)", end="", flush=True)
        if total > 50:
            print()


def _contar_amostras_validas(pasta_v1: Path) -> int:
    """Conta pastas em v1/ que possuem ao menos a imagem *_RAW.jpg esperada pelo dataset."""
    if not pasta_v1.is_dir():
        return 0
    total = 0
    with os.scandir(pasta_v1) as it:
        for entry in it:
            if entry.is_dir() and (Path(entry.path) / f"{entry.name}_RAW.jpg").is_file():
                total += 1
    return total


def instalar_dataset_de_zip(caminho_zip: Path, pasta_dataset: Path) -> Path:
    """Extrai tanto o pacote completo da Mendeley (zip aninhado) quanto o zip direto de v1/.

    Formatos aceitos:
      1) Pacote Mendeley (`t576ydh9v8-4.zip`): contém `Dataset/Cracks-and-Potholes-in-Road-Images.zip`,
         que por sua vez contém `v1/<amostra>/*_RAW.jpg`.
      2) Pacote interno (`Cracks-and-Potholes-in-Road-Images.zip`): contém `v1/<amostra>/*_RAW.jpg`.
      3) Zip contendo diretamente as pastas `<amostra>/*_RAW.jpg` na raiz.
    """
    if not zipfile.is_zipfile(caminho_zip):
        raise zipfile.BadZipFile(f"Arquivo inválido ou corrompido: {caminho_zip}")

    pasta_dataset.mkdir(parents=True, exist_ok=True)
    pasta_v1 = pasta_dataset / "Dataset" / "v1"

    with zipfile.ZipFile(caminho_zip, "r") as zf:
        nomes = zf.namelist()

    tem_zip_interno = any(n.endswith("Cracks-and-Potholes-in-Road-Images.zip") for n in nomes)
    tem_v1 = any(n.startswith("v1/") or "/v1/" in n for n in nomes)

    if tem_zip_interno:
        _info(f"Pacote Mendeley detectado ({caminho_zip.name}); extraindo nível externo...")
        extrair_zip_seguro(caminho_zip, pasta_dataset)
        zip_interno = next(pasta_dataset.rglob("Cracks-and-Potholes-in-Road-Images.zip"), None)
        if zip_interno is None:
            raise FileNotFoundError("Zip interno Cracks-and-Potholes-in-Road-Images.zip não encontrado após extração.")
        _info(f"Extraindo arquivo interno do dataset ({zip_interno.name})...")
        extrair_zip_seguro(zip_interno, pasta_dataset / "Dataset")
    elif tem_v1:
        _info(f"Pacote direto com pasta v1/ detectado ({caminho_zip.name}); extraindo...")
        with tempfile.TemporaryDirectory(prefix="pav_ds_") as tmp:
            tmp_dir = Path(tmp)
            extrair_zip_seguro(caminho_zip, tmp_dir)
            v1_encontrada = next((p for p in tmp_dir.rglob("v1") if p.is_dir()), None)
            if v1_encontrada is None:
                raise FileNotFoundError("Pasta v1/ não localizada dentro do arquivo zip.")
            pasta_v1.parent.mkdir(parents=True, exist_ok=True)
            if pasta_v1.exists():
                shutil.rmtree(pasta_v1)
            shutil.move(str(v1_encontrada), str(pasta_v1))
    else:
        _info(f"Extraindo amostras diretamente para {pasta_v1}...")
        extrair_zip_seguro(caminho_zip, pasta_v1)

    n_amostras = _contar_amostras_validas(pasta_v1)
    if n_amostras == 0:
        raise RuntimeError(f"Nenhuma amostra *_RAW.jpg encontrada em {pasta_v1} após a extração.")
    return pasta_v1


def _baixar_arquivo_com_progresso(url: str, destino: Path) -> None:
    """Baixa um arquivo via HTTP com barra de progresso em MB/% usando arquivo temporário .part."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Pavimentos-Instalador/1.0 (Python urllib)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        total_hdr = resp.headers.get("Content-Length")
        total_bytes = int(total_hdr) if total_hdr and total_hdr.isdigit() else 0
        baixado = 0
        bloco = 256 * 1024
        with open(parcial, "wb") as out:
            while True:
                chunk = resp.read(bloco)
                if not chunk:
                    break
                out.write(chunk)
                baixado += len(chunk)
                mb = baixado / (1024 * 1024)
                if total_bytes > 0:
                    tot_mb = total_bytes / (1024 * 1024)
                    pct = min(100, (100 * baixado) // total_bytes)
                    print(f"\r  [..] Baixando dataset: {mb:6.1f} / {tot_mb:.1f} MB ({pct:3d}%)", end="", flush=True)
                else:
                    print(f"\r  [..] Baixando dataset: {mb:6.1f} MB", end="", flush=True)
        print()
    parcial.replace(destino)


def _resolver_urls_mendeley() -> List[str]:
    """Consulta a API pública da Mendeley Data v4 como fallback caso a URL direta mude."""
    urls: List[str] = [URL_DATASET_ZIP]
    try:
        req = urllib.request.Request(
            URL_MENDELEY_API,
            headers={"Accept": "application/json", "User-Agent": "Pavimentos-Instalador/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            itens = json.loads(resp.read().decode("utf-8"))
        if isinstance(itens, list):
            for item in itens:
                cd = item.get("content_details") or {}
                dl = cd.get("download_url") or item.get("download_url")
                if dl and dl not in urls:
                    urls.append(dl)
    except Exception:
        pass
    return urls


def etapa_4_configurar_dataset_dnit(
    raiz: Path, sem_dataset: bool = False, zip_local: Optional[str] = None
) -> None:
    _banner(
        4,
        5,
        "Configurar dataset público DNIT (Cracks and Potholes)",
        "Verifica se as 2.235 fotos reais (RAW + CRACK + POTHOLE + LANE) estão em dataset/Dataset/v1/ e baixa/extrai se necessário.",
    )
    pasta_dataset = raiz / "dataset"
    pasta_v1 = pasta_dataset / "Dataset" / "v1"

    if sem_dataset:
        _aviso("Download/extração do dataset ignorado via --sem-dataset.")
        _info("A interface web continuará funcionando normalmente com cenários embutidos e upload de fotos.")
        return

    existentes = _contar_amostras_validas(pasta_v1)
    if existentes > 0:
        _ok(f"Dataset DNIT já instalado em {pasta_v1} ({existentes} amostras válidas encontradas).")
        return

    candidato_zip: Optional[Path] = None
    if zip_local:
        candidato_zip = Path(zip_local).expanduser().resolve()
        if not candidato_zip.is_file():
            raise FileNotFoundError(f"Arquivo --zip-local não encontrado: {candidato_zip}")
        _info(f"Usando arquivo local informado: {candidato_zip}")
    else:
        for caminho in (
            pasta_dataset / "t576ydh9v8-4.zip",
            pasta_dataset / "Dataset" / "Cracks-and-Potholes-in-Road-Images.zip",
        ):
            if caminho.is_file() and zipfile.is_zipfile(caminho):
                candidato_zip = caminho
                _info(f"Arquivo zip já existente encontrado localmente: {candidato_zip}")
                break

    if candidato_zip is None:
        destino_zip = pasta_dataset / "t576ydh9v8-4.zip"
        _info("Dataset não encontrado localmente. Iniciando download oficial (~220 MB)...")
        sucesso = False
        ultimo_erro: Optional[Exception] = None
        for url in _resolver_urls_mendeley():
            try:
                _info(f"Tentando fonte: {url}")
                _baixar_arquivo_com_progresso(url, destino_zip)
                if zipfile.is_zipfile(destino_zip):
                    sucesso = True
                    candidato_zip = destino_zip
                    break
                destino_zip.unlink(missing_ok=True)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                ultimo_erro = exc
                destino_zip.with_suffix(".zip.part").unlink(missing_ok=True)

        if not sucesso or candidato_zip is None:
            _aviso(f"Não foi possível baixar o dataset automaticamente ({ultimo_erro}).")
            _aviso(f"Baixe manualmente em: {URL_PAGINA_DATASET}")
            _aviso("Depois execute: python3 instalar.py --zip-local <caminho_do_arquivo.zip>")
            _info("A interface web ainda funciona normalmente com os cenários embutidos e fotos enviadas.")
            return

    v1_final = instalar_dataset_de_zip(candidato_zip, pasta_dataset)
    total = _contar_amostras_validas(v1_final)
    _ok(f"Dataset DNIT extraído e verificado em {v1_final} ({total} amostras prontas).")


# ---------------------------------------------------------------------------
# ETAPA 5: Validação final (importações + suíte de testes automatizados)
# ---------------------------------------------------------------------------
def etapa_5_validar_instalacao(
    raiz: Path, python_venv: Path, pular_testes: bool = False
) -> None:
    _banner(
        5,
        5,
        "Validar módulos do projeto e executar testes automatizados",
        "Confirma que o núcleo inteiro (NumPy), a interface web e os testes unitários executam sem erros.",
    )
    _info("Testando importações de numpy, cv2, pavimento.nucleo e interface.servidor...")
    subprocess.run(
        [
            str(python_venv),
            "-c",
            (
                "import sys; sys.path[:0]=['.', 'src']; "
                "import numpy, cv2, pavimento.nucleo, interface.servidor; "
                "print('Importações OK | NumPy:', numpy.__version__, '| OpenCV:', cv2.__version__)"
            ),
        ],
        cwd=str(raiz),
        check=True,
    )
    _ok("Todos os módulos principais importados com sucesso.")

    if pular_testes:
        _aviso("Execução dos testes automatizados pulada via --pular-testes.")
    else:
        _info("Executando suíte completa de testes (python -m unittest discover -s tests)...")
        subprocess.run(
            [str(python_venv), "-m", "unittest", "discover", "-s", "tests"],
            cwd=str(raiz),
            check=True,
        )
        _ok("Suíte de testes automatizados aprovada!")

    py_rel = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    print("\n" + "=" * 72)
    print("  INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 72)
    print("  Comandos principais para testar o projeto:")
    print(f"    1. Interface Web Interativa: {py_rel} scripts/interface_visual.py")
    print(f"    2. Processar uma imagem:     {py_rel} scripts/processar_imagem.py <pasta_ou_foto>")
    print(f"    3. Avaliar 666 testes DNIT:  {py_rel} scripts/avaliar.py --particao teste")
    print(f"    4. Rodar testes unitários:   {py_rel} -m unittest discover -s tests -v")
    print("=" * 72, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Instalador completo do Sistema de Detecção e Quantificação de Danos em Pavimentos."
    )
    ap.add_argument(
        "--sem-dataset",
        action="store_true",
        help="Pula download e extração do dataset DNIT (~220 MB).",
    )
    ap.add_argument(
        "--zip-local",
        default=None,
        metavar="CAMINHO",
        help="Usa um arquivo .zip local já baixado do Mendeley Data (t576ydh9v8-4.zip ou interno).",
    )
    ap.add_argument(
        "--recriar-venv",
        action="store_true",
        help="Remove e recria a pasta .venv do zero.",
    )
    ap.add_argument(
        "--forcar-pip",
        action="store_true",
        help="Força reinstalação de requirements.txt mesmo se NumPy e OpenCV já estiverem no .venv.",
    )
    ap.add_argument(
        "--pular-testes",
        action="store_true",
        help="Pula a execução da suíte unittest na etapa final.",
    )
    ap.add_argument(
        "--abrir",
        action="store_true",
        help="Inicia a interface web automaticamente ao terminar a instalação.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    raiz = Path(__file__).resolve().parent

    print("========================================================================")
    print("  Instalador — Detecção e Quantificação de Danos em Pavimentos (PI 2026.2)")
    print("========================================================================")

    etapa_1_verificar_ambiente(raiz)
    python_venv = etapa_2_criar_ambiente_virtual(raiz, recriar=args.recriar_venv)
    etapa_3_instalar_dependencias(raiz, python_venv, forcar_pip=args.forcar_pip)
    etapa_4_configurar_dataset_dnit(
        raiz, sem_dataset=args.sem_dataset, zip_local=args.zip_local
    )
    etapa_5_validar_instalacao(raiz, python_venv, pular_testes=args.pular_testes)

    if args.abrir:
        print("\nIniciando a interface web interativa...")
        subprocess.run([str(python_venv), "scripts/interface_visual.py"], cwd=str(raiz), check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInstalação interrompida pelo usuário.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERRO] Falha na instalação: {exc}", file=sys.stderr)
        raise SystemExit(1)
