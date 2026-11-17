"""Regressões sintéticas; melhora real depende do treino/teste de T4.

Fixtures aritméticas sem gerador aleatório. Gates: unittest discover -s tests.
"""
import sys
import unittest
from dataclasses import fields, replace
from math import isqrt
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from pavimento.nucleo import (analisar, histerese_componentes, otsu_inteiro,
                              raiz_inteira)
from pavimento.parametros import GRADE_BUSCA, Parametros


def asfalto():
    y, x = np.indices((320, 512), dtype=np.int32)
    ruido = ((x * 37 + y * 61 + x * y * 7) % 17) - 8
    return (150 + ruido).astype(np.uint8), np.ones(x.shape, bool), y, x


def disco(centro_liso=False, raio=40):
    g, roi, y, x = asfalto()
    d2 = (x - 256) ** 2 + (y - 160) ** 2
    alvo = d2 <= raio ** 2
    textura = ((x * 19 + y * 31 + x * y * 11) % 61) - 30
    g[alvo] = (65 + textura[alvo]).astype(np.uint8)
    centro = d2 <= (raio // 2) ** 2
    if centro_liso:
        g[centro] = 65
    return g, roi, alvo, centro


class TestBuracos(unittest.TestCase):
    def verificar_disco(self, p, centro_liso=False, raio=40):
        g, roi, alvo, centro = disco(centro_liso, raio)
        r = analisar(g, roi, p)
        self.assertEqual(r["quant"]["n_buracos"], 1)
        inter = int((r["buracos"] & alvo).sum())
        uniao = int((r["buracos"] | alvo).sum())
        self.assertGreaterEqual(10 * inter, 6 * uniao)
        if centro_liso:
            self.assertFalse(r["cand_buraco"][160, 256])
            self.assertTrue(r["buracos"][centro].all())
        return r

    def test_disco_defaults_e_json(self):
        for p in (Parametros(), Parametros.carregar(str(RAIZ / "params.json"))):
            with self.subTest(p=p):
                self.verificar_disco(p)

    def test_histerese_recupera_centro_liso(self):
        self.verificar_disco(Parametros(), centro_liso=True)

    def test_buraco_pequeno(self):
        self.verificar_disco(Parametros(), raio=12)

    def test_trinca_fina_e_asfalto(self):
        g, roi, _, _ = asfalto()
        for trinca in (False, True):
            if trinca:
                g[45:275, 255:257] = 35
            r = analisar(g, roi)
            self.assertEqual(r["quant"]["n_buracos"], 0)
            self.assertFalse(r["buracos"].any())

    def test_faixa_alongada_rejeitada_por_aspecto(self):
        g, roi, y, x = asfalto()
        faixa = (x >= 32) & (x < 58) & (y >= 40) & (y < 280)
        g[faixa] = (55 + (x[faixa] * 19 + y[faixa] * 31) % 51).astype(np.uint8)
        # Desliga circularidade e borda para isolar o novo descritor de aspecto.
        p = replace(Parametros(), buraco_circ_min_1000=0, buraco_borda_margem=0)
        self.assertEqual(analisar(g, roi, p)["quant"]["n_buracos"], 0)
        controle = analisar(g, roi, replace(p, buraco_aspecto_max_1000=20000))
        self.assertEqual(controle["quant"]["n_buracos"], 1)

    def test_borda_roi_rejeitada(self):
        g, roi, alvo, _ = disco()
        roi[:, :245] = False  # disco recortado pela borda da pista
        p = replace(Parametros(), buraco_circ_min_1000=0)
        self.assertEqual(analisar(g, roi, p)["quant"]["n_buracos"], 0)
        self.assertEqual(analisar(g, roi, replace(p, buraco_borda_margem=0))
                         ["quant"]["n_buracos"], 1)

    def test_determinismo_descritores_e_api(self):
        g, roi, _, _ = disco()
        a, b = analisar(g, roi), analisar(g, roi)
        antigas = {"media", "desvio", "cand_trinca", "cand_buraco", "trincas",
                   "esqueleto", "buracos", "regioes_trinca", "regioes_buraco",
                   "limiar_textura", "quant"}
        self.assertTrue(antigas <= a.keys())
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(v, b[k])
                self.assertIn(v.dtype.kind, "biu")
            else:
                self.assertEqual(v, b[k])
        for k in ("limiar_textura", "limiar_escuro", "limiar_escuro_forte", "limiar_otsu"):
            self.assertIs(type(a[k]), int)
        for r in a["regioes_buraco"]:
            for v in r.values():
                for n in (v if isinstance(v, tuple) else (v,)):
                    self.assertIs(type(n), int)

    def test_histerese_so_componentes_com_semente(self):
        fraca = np.zeros((9, 12), bool)
        fraca[1:4, 1:4] = True
        fraca[5:8, 8:11] = True
        forte = np.zeros_like(fraca)
        forte[2, 2] = True
        forte[0, 11] = True  # semente fora da região fraca não cresce
        esperado = fraca.copy()
        esperado[5:8, 8:11] = False
        np.testing.assert_array_equal(histerese_componentes(forte, fraca), esperado)

    def test_otsu_inteiro_e_modo_alternativo(self):
        for valores, esperado in (([], 0), ([73] * 20, 73), ([10] * 9 + [200] * 17, 10)):
            self.assertEqual(otsu_inteiro(np.array(valores, dtype=np.uint8)), esperado)
        self.verificar_disco(replace(Parametros(), buraco_usar_otsu=1))

    def test_raiz_inteira(self):
        v = np.arange(65536, dtype=np.int64).reshape(256, 256)
        esperado = np.array([isqrt(int(x)) for x in v.flat], dtype=np.int32).reshape(v.shape)
        np.testing.assert_array_equal(raiz_inteira(v), esperado)

    def test_parametros_grade_e_roi_vazia(self):
        p = Parametros.carregar(str(RAIZ / "params.json"))
        for f in fields(p):
            self.assertIs(type(getattr(p, f.name)), int)
            if f.name.startswith("buraco_"):
                self.assertIn(f.name, GRADE_BUSCA)
                self.assertTrue(all(type(v) is int for v in GRADE_BUSCA[f.name]))
        g, roi, _, _ = asfalto()
        r = analisar(g, ~roi, p)
        self.assertFalse(r["buracos"].any())


if __name__ == "__main__":
    unittest.main()
