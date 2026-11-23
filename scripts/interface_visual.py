"""Inicia a interface gráfica web local do projeto Detecção de Danos em Pavimentos.

Uso:
  python scripts/interface_visual.py
  python scripts/interface_visual.py --porta 8080
  python scripts/interface_visual.py --sem-navegador
"""
from __future__ import annotations

import argparse
import os
import sys

# Garante importação de `pavimento` e `interface` independente do diretório atual
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SCRIPT_DIR)
for _p in (_RAIZ, os.path.join(_RAIZ, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interface.servidor import iniciar


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Interface gráfica interativa para visualizar e demonstrar cada etapa de Detecção de Danos em Pavimentos."
    )
    ap.add_argument("--host", default="127.0.0.1", help="Endereço de escuta (padrão: 127.0.0.1)")
    ap.add_argument("--porta", type=int, default=8080, help="Porta inicial (padrão: 8080)")
    ap.add_argument("--sem-navegador", action="store_true", help="Não abre o navegador automaticamente")
    args = ap.parse_args(argv)
    iniciar(host=args.host, porta=args.porta, abrir_navegador=not args.sem_navegador)


if __name__ == "__main__":
    main()
