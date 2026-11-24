"""Permite iniciar a interface com `python -m interface`."""
from __future__ import annotations

import os
import sys

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_RAIZ, os.path.join(_RAIZ, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interface.app import executar_interface  # noqa: E402

if __name__ == "__main__":
    alvo = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(executar_interface(caminho_inicial=alvo))
