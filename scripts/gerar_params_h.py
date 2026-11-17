"""Gera esp32cam/include/params.h a partir de params.json (fonte única dos parâmetros)."""
import os
from dataclasses import fields

import _caminho
from pavimento.parametros import Parametros


def main():
    p = Parametros.carregar(_caminho.PARAMS) if os.path.exists(_caminho.PARAMS) else Parametros()
    corpo = p.cabecalho_c()
    campos = ", ".join(f".{f.name} = P_{f.name.upper()}" for f in fields(Parametros))
    corpo += f"\n#define PAV_PARAMS_PADRAO {{ {campos} }}\n"
    import json
    from pavimento.estrada import ParamEstrada
    arq_e = os.path.join(_caminho.RAIZ, "params_estrada.json")
    pe = ParamEstrada(**json.load(open(arq_e))) if os.path.exists(arq_e) else ParamEstrada()
    corpo += "\n// Detecção da estrada (params_estrada.json)\n"
    corpo += "#define PAV_ESTRADA_PADRAO { " + ", ".join(f".{f.name} = {getattr(pe, f.name)}" for f in fields(ParamEstrada)) + " }\n"
    destino = os.path.join(_caminho.RAIZ, "esp32cam", "include", "params.h")
    with open(destino, "w", encoding="utf-8") as f:
        f.write(corpo)
    print("gerado", destino)


if __name__ == "__main__":
    main()
