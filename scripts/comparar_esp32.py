"""Verifica bit a bit que o núcleo em C (ESP32 ou PC) produz o MESMO resultado do Python.

Lê os PGMs de entrada (mesmos que foram para o SD), roda nucleo.analisar() em Python,
e compara todos os campos de quantificação + hash FNV-1a das máscaras com o CSV do C.

Uso:
  python scripts/comparar_esp32.py --entrada E:/pav/entrada --csv E:/pav/saida/resultados.csv
  python scripts/comparar_esp32.py --entrada outputs/sd_teste --pc   # compila e roda o C no PC (gcc)
"""
import argparse, csv, os, subprocess, sys, tempfile

import numpy as np

import _caminho
from pavimento.estrada import ParamEstrada, crescer_estrada
from pavimento.nucleo import TIPOS, analisar
from pavimento.parametros import Parametros


def ler_pgm(f):
    with open(f, "rb") as fp:
        dados = fp.read()
    partes, pos = [], 0
    while len(partes) < 4:                      # P5, w, h, maxval
        while dados[pos:pos + 1].isspace():
            pos += 1
        ini = pos
        while not dados[pos:pos + 1].isspace():
            pos += 1
        partes.append(dados[ini:pos])
    w, h = int(partes[1]), int(partes[2])
    return np.frombuffer(dados[pos + 1:pos + 1 + w * h], np.uint8).reshape(h, w)


def fnv1a(m):
    hsh = 2166136261
    for b in m.astype(np.uint8).ravel().tolist():
        hsh = ((hsh ^ b) * 16777619) & 0xFFFFFFFF
    return f"{hsh:08x}"


def linha_python(nome, g, roi, p, hash_estrada="00000000"):
    r = analisar(g, roi > 0, p)
    q = r["quant"]
    return {"nome": nome, "status": "0", "n_trincas": q["n_trincas"],
            "comprimento_trincas_px": q["comprimento_trincas_px"], "area_trincas_px": q["area_trincas_px"],
            "largura_media_trincas_16": q["largura_media_trincas_16"],
            "densidade_trincas_1000": q["densidade_trincas_1000"],
            **{t: q["tipos_trinca"][t] for t in TIPOS},
            "n_buracos": q["n_buracos"], "area_buracos_px": q["area_buracos_px"],
            "area_buracos_pct_100": q["area_buracos_pct_100"], "indice_dano_1000": q["indice_dano_1000"],
            "severidade": q["severidade"], "limiar_textura": r["limiar_textura"],
            "hash_trincas": fnv1a(r["trincas"]), "hash_buracos": fnv1a(r["buracos"]),
            "hash_estrada": hash_estrada}


def carregar_entrada(entrada, nome, pe):
    """Intensidade + (máscara da pista OU estrada automática a partir da saturação)."""
    g = ler_pgm(os.path.join(entrada, nome + ".pgm"))
    f_roi = os.path.join(entrada, nome + "_roi.pgm")
    if os.path.exists(f_roi):
        return g, ler_pgm(f_roi) > 0, "00000000"
    s = ler_pgm(os.path.join(entrada, nome + "_sat.pgm"))
    roi, _ = crescer_estrada(g, s, pe)
    return g, roi, fnv1a(roi)


def rodar_c_no_pc(entrada):
    raiz = os.path.join(_caminho.RAIZ, "esp32cam")
    exe = os.path.join(tempfile.gettempdir(), "teste_pc" + (".exe" if os.name == "nt" else ""))
    cmd = ["gcc", "-O2", "-I" + os.path.join(raiz, "lib", "nucleo"), "-I" + os.path.join(raiz, "include"),
           os.path.join(raiz, "test_pc", "teste_pc.c"), os.path.join(raiz, "lib", "nucleo", "nucleo.c"),
           os.path.join(raiz, "lib", "nucleo", "estrada.c"), "-o", exe]
    subprocess.run(cmd, check=True)
    saida = os.path.join(tempfile.gettempdir(), "resultados_pc.csv")
    subprocess.run([exe, entrada, saida], check=True)
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--csv", help="resultados.csv gerado pela ESP32")
    ap.add_argument("--pc", action="store_true", help="compila e roda o núcleo C no PC")
    a = ap.parse_args()
    p = Parametros.carregar(_caminho.PARAMS) if os.path.exists(_caminho.PARAMS) else Parametros()
    import json
    arq_e = os.path.join(_caminho.RAIZ, "params_estrada.json")
    pe = ParamEstrada(**json.load(open(arq_e))) if os.path.exists(arq_e) else ParamEstrada()
    arq = rodar_c_no_pc(a.entrada) if a.pc else a.csv
    linhas_c = {r["nome"]: r for r in csv.DictReader(open(arq, encoding="utf-8"))}
    iguais, difs, tempos = 0, [], []
    for nome in open(os.path.join(a.entrada, "lista.txt")).read().split():
        g, roi, h_e = carregar_entrada(a.entrada, nome, pe)
        py = linha_python(nome, g, roi, p, h_e)
        c = linhas_c.get(nome)
        if c is None:
            difs.append((nome, "ausente no CSV do C", "", "")); continue
        tempos.append(int(c["tempo_us"]))
        dif = [(k, str(v), c.get(k)) for k, v in py.items() if str(v) != c.get(k)]
        if dif:
            difs.extend((nome, *d) for d in dif)
        else:
            iguais += 1
    total = iguais + len({d[0] for d in difs})
    print(f"Amostras idênticas (todos os campos + hash das máscaras): {iguais}/{total}")
    if tempos:
        tempos.sort()
        print(f"Tempo do núcleo C: mediana {tempos[len(tempos)//2]/1000:.1f} ms, máx {tempos[-1]/1000:.1f} ms")
    for d in difs[:20]:
        print("DIFERENÇA", d)
    sys.exit(0 if not difs else 1)


if __name__ == "__main__":
    main()
