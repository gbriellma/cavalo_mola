"""Inicia a interface gráfica desktop em Python para inspeção etapa por etapa do pipeline.

Uso:
  python scripts/interface_visual.py
  python scripts/interface_visual.py dataset/Dataset/v1/1092481_DF_251_251BDF0052_00028
  python scripts/interface_visual.py caminho/para/foto.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SCRIPT_DIR)
for _p in (_RAIZ, os.path.join(_RAIZ, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interface.app import executar_interface  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Interface gráfica desktop em Python para visualizar todas as etapas do pipeline de pavimentos."
    )
    ap.add_argument(
        "entrada",
        nargs="?",
        default=None,
        help="Caminho opcional para uma foto (.jpg/.png) ou pasta de amostra do dataset DNIT.",
    )
    args = ap.parse_args(argv)
    return executar_interface(caminho_inicial=args.entrada)


if __name__ == "__main__":
    raise SystemExit(main())
