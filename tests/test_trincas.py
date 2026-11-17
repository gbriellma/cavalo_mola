"""Regressões T3: morfologia, referência congelada e calibração sem vazamento.

Fixtures inteiras e determinísticas; nenhuma dependência do dataset real.
"""
import itertools
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "scripts")]
import _caminho
import calibrar_severidade as calibracao
from _comum import referencia_rotulo
from pavimento.dataset import particao
from pavimento.metricas import confusao
from pavimento.nucleo import (analisar, dilatar, histerese_componentes,
                              indice_dano, morfologia_direcional, severidade)
from pavimento.parametros import GRADE_BUSCA, Parametros, REFERENCIA_SEVERIDADE


def nome_treino(i=0):
    return [f"img{k}" for k in range(100) if particao(f"img{k}") == "treino"][i]


def asfalto():
    y, x = np.indices((128, 192), dtype=np.int32)
    return (150 + ((x * 37 + y * 61 + x * y * 7) % 9) - 4).astype(np.uint8)


def dados_vazios(i=0):
    g = np.full((20, 20), 150, np.uint8)
    roi = np.ones(g.shape, bool)
    return nome_treino(i), g, roi, np.zeros_like(roi), np.zeros_like(roi)


class TestTrincas(unittest.TestCase):
    def parametros(self):
        # Geometria em imagens sintéticas ideais é o contrato do ALGORITMO (parâmetros
        # padrão). O params.json é calibrado para imagens reais e é coberto por
        # test_params_json_roda_deterministico, não por estas medidas exatas.
        return (Parametros(),)

    def test_params_json_roda_deterministico(self):
        p = Parametros.carregar(str(RAIZ / "params.json"))
        g = asfalto()
        roi = np.ones(g.shape, bool)
        g[60:62, 20:110] = 40  # trinca horizontal fina e escura (2 px, como nas imagens reais)
        a, b = analisar(g, roi, p), analisar(g, roi, p)
        for k in ("trincas", "buracos"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertGreater(a["quant"]["n_trincas"], 0)
        self.assertIs(type(a["quant"]["indice_dano_1000"]), int)

    def test_linha_variavel_f1_pixel_e_pontas(self):
        for p in self.parametros():
            for orientacao in ("horizontal", "vertical", "diagonal", "antidiagonal"):
                with self.subTest(p=p, orientacao=orientacao):
                    g = asfalto()
                    roi = np.ones(g.shape, bool)
                    linha = np.zeros_like(roi)
                    t = np.arange(88)
                    if orientacao == "horizontal":
                        ys, xs = np.full_like(t, 64), t + 24
                    elif orientacao == "vertical":
                        ys, xs = t + 20, np.full_like(t, 64)
                    elif orientacao == "diagonal":
                        ys, xs = t + 20, t + 24
                    else:
                        ys, xs = t + 20, 150 - t
                    linha[ys, xs] = True
                    g[ys, xs] = (35 + (t * 3 % 100)).astype(np.uint8)
                    # Rótulo mais espesso que a fissura visível: tolerância 2 px.
                    gt = dilatar(linha, 5)
                    r = analisar(g, roi, p)
                    tp, fp, recuperados, fn = confusao(r["trincas"], gt, roi, 2)
                    # F1 >= 7/10 por produtos inteiros de precisão/revocação.
                    den = tp * (recuperados + fn) + recuperados * (tp + fp)
                    self.assertGreater(den, 0)
                    self.assertGreaterEqual(20 * tp * recuperados, 7 * den)
                    self.assertTrue(r["trincas"][ys, xs].all())

    def test_angulos_intermediarios_preservam_sementes_e_f1(self):
        # Inclinações 1:2 e 1:3, suas transpostas e reflexões. Em 1:2 nenhum
        # dos quatro EEs retos de 3 px cabe na linha digital de 1 px.
        extremos = (((30, 30), (150, 90)), ((30, 15), (90, 135)),
                    ((30, 120), (150, 60)), ((30, 135), (90, 15)),
                    ((30, 30), (150, 70)), ((30, 15), (70, 135)),
                    ((30, 120), (150, 80)), ((30, 135), (70, 15)))
        for p in self.parametros():
            for inicio, fim in extremos:
                for variavel in (False, True):
                    with self.subTest(p=p, inicio=inicio, fim=fim, variavel=variavel):
                        g = np.full((150, 200), 150, np.uint8)
                        m = np.zeros_like(g)
                        cv2.line(m, inicio, fim, 1, 1)
                        linha = m > 0
                        self.assertEqual(int(linha.sum()), 121)
                        dx, dy = abs(fim[0] - inicio[0]), abs(fim[1] - inicio[1])
                        if max(dx, dy) == 2 * min(dx, dy):
                            self.assertFalse(morfologia_direcional(linha, 3).any())
                        g[linha] = ((35 + np.arange(121) * 3 % 100).astype(np.uint8)
                                    if variavel else 35)
                        roi = np.ones_like(linha)
                        r = analisar(g, roi, p)
                        if not variavel:
                            self.assertEqual(int(r["trinca_forte"].sum()), 121)
                        self.assertTrue(r["trincas"][linha].all())
                        self.assertTrue(r["trincas"][inicio[1], inicio[0]])
                        self.assertTrue(r["trincas"][fim[1], fim[0]])
                        tp, fp, recuperados, fn = confusao(r["trincas"], dilatar(linha, 5), roi, 2)
                        den = tp * (recuperados + fn) + recuperados * (tp + fp)
                        self.assertGreater(den, 0)
                        self.assertGreaterEqual(20 * tp * recuperados, 7 * den)

    def test_fragmento_curto_e_textura_rejeitados(self):
        for p in self.parametros():
            for fragmento in (False, True):
                g = asfalto()
                if fragmento:
                    g[64, 70:75] = 30
                self.assertFalse(analisar(g, np.ones(g.shape, bool), p)["trincas"].any())

    def test_histerese_ramo_conectado_e_isolado_no_detector(self):
        for base in self.parametros():
            p = replace(base, trinca_b16=640)
            g = np.full((128, 192), 150, np.uint8)
            g[40, 20:40] = 35  # semente forte
            g[40, 40:120] = 120  # ramo fraco conectado
            g[90, 60:140] = 120  # mesmo contraste, isolado
            r = analisar(g, np.ones(g.shape, bool), p)
            self.assertFalse(r["trinca_forte"][40, 40:120].any())
            self.assertTrue(r["trinca_fraca"][90, 60:140].all())
            self.assertTrue(r["trincas"][40, 20:120].all())
            self.assertFalse(r["trincas"][90, 60:140].any())

    def test_reconstrucao_geodesica_equivale_histerese_e_recupera_largura(self):
        fraca = np.zeros((30, 70), bool)
        fraca[8:11, 8:62] = True
        fraca[20:23, 20:60] = True
        forte = np.zeros_like(fraca)
        forte[9, 10:50] = True
        x = forte.copy()
        while True:
            seguinte = dilatar(x, 3) & fraca
            if np.array_equal(x, seguinte):
                break
            x = seguinte
        np.testing.assert_array_equal(histerese_componentes(forte, fraca), x)
        self.assertTrue(x[8:11, 8:62].all())
        self.assertFalse(x[20:23].any())

    def test_detector_recupera_largura_fraca_sem_engrossar_fundo(self):
        for base in self.parametros():
            g = np.full((100, 160), 150, np.uint8)
            g[49:52, 20:140] = 120
            g[50, 20:140] = 35
            r = analisar(g, np.ones(g.shape, bool), replace(base, trinca_b16=640))
            self.assertEqual(int(r["trinca_forte"].sum()), 120)
            self.assertTrue(r["trincas"][49:52, 20:140].all())
            self.assertEqual(int(r["trincas"].sum()), 360)

    def test_direcoes_ligam_fragmentos_e_abertura_preserva_linhas(self):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            m = np.zeros((50, 70), bool)
            for d in range(-15, 16):
                if d != 0:
                    m[25 + d * dy, 35 + d * dx] = True
            fechado = morfologia_direcional(m, 3, fechar=True)
            self.assertTrue(fechado[25, 35])
            aberto = morfologia_direcional(fechado, 3)
            self.assertTrue(aberto[25, 35])
            self.assertGreaterEqual(int(aberto.sum()), 31)

    def test_determinismo_e_inteiros(self):
        g = asfalto()
        g[64, 20:172] = 45
        roi = np.ones_like(g, bool)
        a, b = analisar(g, roi), analisar(g, roi)
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(v, b[k])
                self.assertIn(v.dtype.kind, "biu")
            else:
                self.assertEqual(v, b[k])
        for f in fields(Parametros):
            self.assertIs(type(getattr(Parametros(), f.name)), int)
            if f.name.startswith(("trinca_", "sev_", "dano_peso_")):
                self.assertIn(f.name, GRADE_BUSCA)
                self.assertTrue(all(type(v) is int for v in GRADE_BUSCA[f.name]))


class TestSeveridade(unittest.TestCase):
    def test_referencia_imutavel_fora_de_parametros_e_grade(self):
        self.assertEqual(REFERENCIA_SEVERIDADE[:3], (10, 214, 522))
        self.assertEqual(REFERENCIA_SEVERIDADE[3:], (40, 25))
        with self.assertRaises(AttributeError):
            REFERENCIA_SEVERIDADE.sev_min_1000 = 999
        self.assertNotIn("REFERENCIA_SEVERIDADE", asdict(Parametros()))
        self.assertNotIn("REFERENCIA_SEVERIDADE", GRADE_BUSCA)

    def test_invariancia_referencia_e_previsao_vazia_continua_erro(self):
        dados = dados_vazios()
        dados[4][0, :10] = True  # área 10 / ROI 400: índice histórico 625
        ref = REFERENCIA_SEVERIDADE
        self.assertEqual(indice_dano(0, 10, 400, ref.dano_peso_trinca, ref.dano_peso_buraco), 625)
        esperado = referencia_rotulo(dados)
        self.assertEqual(esperado["indice_dano_1000"], 625)
        self.assertEqual(esperado["severidade"], "alta")
        for pesos in ((0, 0), (1, 999), (999, 1)):
            p = replace(Parametros(), sev_min_1000=998, sev_media_1000=999,
                        sev_alta_1000=1000, dano_peso_trinca=pesos[0], dano_peso_buraco=pesos[1])
            self.assertEqual(referencia_rotulo(dados, p), esperado)
            previsto = analisar(dados[1], dados[2], p)["quant"]
            self.assertEqual(previsto["severidade"], "nenhum")
            self.assertNotEqual(previsto["severidade"], esperado["severidade"])
            _, auditoria = calibracao.calibrar_severidade([dados], p)
            self.assertEqual(auditoria["acertos_exatos"], 0)
            self.assertEqual(auditoria["acertos_ate_1_nivel"], 0)

    def test_severidade_e_indice_monotonicos(self):
        for p in (Parametros(), replace(Parametros(), sev_min_1000=998,
                                       sev_media_1000=999, sev_alta_1000=1000)):
            niveis = [calibracao.NIVEIS.index(severidade(i, p)) for i in range(1001)]
            self.assertEqual(niveis, sorted(niveis))
            for comp in (0, 10, 100):
                indices = [indice_dano(comp, a, 10000) for a in range(1001)]
                self.assertEqual(indices, sorted(indices))

    def test_busca_otima_contra_enumeracao_independente(self):
        indices, refs = [0, 50, 100, 200, 800, 1000], [0, 1, 2, 1, 3, 3]
        # Entre valores observados, só a primeira fronteira e as duas seguintes
        # podem mudar o desempate (limiares estritamente crescentes).
        candidatos = sorted({v + d for v in [0] + indices for d in (1, 2, 3) if v + d <= 1000})
        def nota(limites):
            pred = [sum(i >= t for t in limites) for i in indices]
            distancias = [abs(a - b) for a, b in zip(pred, refs)]
            return (sum(d == 0 for d in distancias), sum(d <= 1 for d in distancias),
                    -sum(distancias), tuple(-t for t in limites))
        esperado = max(itertools.combinations(candidatos, 3), key=nota)
        self.assertEqual(calibracao._buscar_limiares(indices, refs), esperado)

    def test_calibracao_real_crescente_deterministica(self):
        vazio = dados_vazios()
        g = asfalto()
        gt = np.zeros_like(g, bool)
        gt[64, 20:172] = True
        g[gt] = 35
        trinca = nome_treino(1), g, np.ones_like(gt), gt, np.zeros_like(gt)
        dados = [vazio, trinca]
        a = calibracao.calibrar_severidade(dados)
        b = calibracao.calibrar_severidade(dados)
        self.assertEqual(a, b)
        p, auditoria = a
        self.assertLess(p.sev_min_1000, p.sev_media_1000)
        self.assertLess(p.sev_media_1000, p.sev_alta_1000)
        self.assertEqual(auditoria["acertos_exatos"], 2)
        self.assertEqual(auditoria["referencia_severidade"], REFERENCIA_SEVERIDADE._asdict())

    def test_recusa_teste_antes_de_qualquer_leitura(self):
        with patch.object(calibracao, "listar_amostras") as listar:
            with self.assertRaisesRegex(ValueError, "somente treino"):
                calibracao.main(["--particao", "teste"])
            listar.assert_not_called()
        d = dados_vazios()
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        with self.assertRaisesRegex(ValueError, "somente treino"):
            calibracao.calibrar_severidade([(nome, *d[1:])])

    def test_cli_valida_grava_auditoria_e_json_carregavel(self):
        with tempfile.TemporaryDirectory(dir=RAIZ) as tmp:
            raiz = Path(tmp)
            dataset = raiz / "dataset"
            pasta = dataset / nome_treino()
            pasta.mkdir(parents=True)
            nome = pasta.name
            cv2.imwrite(str(pasta / f"{nome}_RAW.jpg"), np.full((40, 40, 3), 150, np.uint8))
            for classe in ("CRACK", "POTHOLE", "LANE"):
                m = np.full((40, 40), 255 if classe == "LANE" else 0, np.uint8)
                cv2.imwrite(str(pasta / f"{nome}_{classe}.png"), m)
            saida = raiz / "params.json"
            with patch.object(_caminho, "CACHE", str(raiz / "cache")), \
                    patch.object(calibracao, "validar_amostras", wraps=calibracao.validar_amostras) as validar:
                calibracao.main(["--dataset", str(dataset), "--saida", str(saida)])
                validar.assert_called_once_with([str(pasta)])
            conteudo = json.loads(saida.read_text(encoding="utf-8"))
            self.assertEqual(conteudo["calibracao_severidade"]["referencia_severidade"],
                             REFERENCIA_SEVERIDADE._asdict())
            p = Parametros.carregar(str(saida))
            self.assertLess(p.sev_min_1000, p.sev_media_1000)
            self.assertLess(p.sev_media_1000, p.sev_alta_1000)


if __name__ == "__main__":
    unittest.main()
