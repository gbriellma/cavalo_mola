import os
import sys
from pathlib import Path

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
DATASET = os.path.realpath(os.environ.get("CAVALO_DATASET", os.path.join(RAIZ, "dataset", "Dataset", "v1")))
SAIDAS = os.path.join(RAIZ, "outputs")
CACHE = os.path.join(SAIDAS, "cache")
PARAMS = os.path.join(RAIZ, "params.json")


def dataset_atual():
    caminho = Path(os.environ.get("CAVALO_DATASET", DATASET)).expanduser().resolve()
    if not caminho.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho}. Configure CAVALO_DATASET para a pasta v1 real.")
    return str(caminho)
