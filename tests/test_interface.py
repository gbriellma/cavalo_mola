"""Testes da interface gráfica desktop em Python (PyQt5) e paridade com o núcleo."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RAIZ = Path(__file__).resolve().parents[1]
for p in (str(RAIZ), str(RAIZ / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from interface.app import criar_aplicacao  # noqa: E402
from interface.cenarios import CENARIOS, carregar_cenario  # noqa: E402
from interface.conteudo import ORDEM_ETAPAS  # noqa: E402
from interface.dados import carregar_entrada_arquivo, filtrar_dataset_dnit  # noqa: E402
from interface.janela_principal import JanelaPrincipal  # noqa: E402
from interface.pipeline_didatico import (  # noqa: E402
    _motivo_mais_frequente,
    executar_pipeline_didatico,
    inspecionar_pixel,
)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pavimento.nucleo import analisar, intensidade_hsi, reduzir_2x, reduzir_mascara_2x  # noqa: E402
from pavimento.parametros import Parametros  # noqa: E402


class TestInterfaceDesktop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = criar_aplicacao([])

    def test_paridade_todos_cenarios_e_inspecao_pixel(self):
        p_final = Parametros.carregar(str(RAIZ / "params.json"))
        for c in CENARIOS:
            with self.subTest(cenario=c.id):
                bgr, roi_full, info = carregar_cenario(c.id)
                res = executar_pipeline_didatico(
                    bgr, roi_full, p_final, aplicar_reducao_2x=info.reduzir_padrao
                )

                g = intensidade_hsi(bgr)
                roi = roi_full
                if info.reduzir_padrao:
                    g = reduzir_2x(g)
                    roi = reduzir_mascara_2x(roi, "maioria")
                oficial = analisar(g, roi, p_final)

                self.assertEqual(res["quant"], oficial["quant"])
                self.assertEqual(res["estatisticas"]["limiar_textura"], oficial["limiar_textura"])
                self.assertEqual(res["estatisticas"]["limiar_otsu"], oficial["limiar_otsu"])
                np.testing.assert_array_equal(res["mapas"]["g"], g)
                np.testing.assert_array_equal(res["mapas"]["trincas"], oficial["trincas"])
                np.testing.assert_array_equal(res["mapas"]["buracos"], oficial["buracos"])

                h, w = g.shape
                px_centro = inspecionar_pixel(res, w // 2, h // 2)
                self.assertIsNotNone(px_centro)
                self.assertEqual(px_centro["intensidade"], int(g[h // 2, w // 2]))
                self.assertIsNone(inspecionar_pixel(res, -1, 0))
                self.assertIsNone(inspecionar_pixel(res, w, h))

    def test_resumo_didatico_esquema(self):
        self.assertEqual(_motivo_mais_frequente([]), "Nenhum rejeitado")
        self.assertEqual(
            _motivo_mais_frequente([{"motivos": ["A", "B"]}, {"motivos": ["A"]}]),
            "A",
        )
        p_final = Parametros.carregar(str(RAIZ / "params.json"))
        for cid in ("didatico_buraco_faixa", "didatico_sombra_trinca"):
            bgr, roi_full, info = carregar_cenario(cid)
            res = executar_pipeline_didatico(bgr, roi_full, p_final, aplicar_reducao_2x=info.reduzir_padrao)
            resumo = res["resumo_didatico"]
            self.assertEqual(
                set(resumo.keys()),
                {"etapa1", "etapa2", "etapa3a", "etapa3b", "etapa4", "etapa5"},
            )
            self.assertIn(resumo["etapa5"]["severidade_label"], {"Nenhum", "Baixa", "Média", "Alta"})

    def test_janela_principal_fluxo_completo(self):
        janela = JanelaPrincipal()
        try:
            self.assertIsNotNone(janela._resultado)

            # Percorre todas as etapas do pipeline e modos de comparação
            for eid in ORDEM_ETAPAS:
                janela.selecionar_etapa(eid)
                self.assertEqual(janela._etapa_atual, eid)
                for sub in janela._subetapas_disponiveis(eid):
                    janela._escolher_subetapa(sub[0])
                    self.assertEqual(janela._subetapa_atual, sub[0])

            for modo in ("unico", "lado", "cortina"):
                janela.definir_modo_comparacao(modo)
                self.assertEqual(janela._canvas.modo_comparacao(), modo)

            # Alterna abas principais e clique em tile do mosaico
            janela.alternar_aba_principal(1)
            self.assertEqual(janela._stack_views.currentIndex(), 1)
            janela._tiles_mosaico["4_abertura"].clicado.emit("4", "4_abertura")
            self.assertEqual(janela._stack_views.currentIndex(), 0)
            self.assertEqual(janela._etapa_atual, "4")
            self.assertEqual(janela._subetapa_atual, "4_abertura")

            janela.alternar_aba_principal(2)
            self.assertEqual(janela._stack_views.currentIndex(), 2)
            janela.alternar_aba_principal(0)

            # Presets de parâmetros
            for preset in ("slides_anterior", "padrao", "final"):
                janela.aplicar_preset_parametros(preset)

            # Carregamento de foto externa + máscara ROI + exportação PNG/JSON
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                img = np.full((120, 160, 3), 150, dtype=np.uint8)
                img[58:60, 20:140] = 30
                roi = np.full((120, 160), 255, dtype=np.uint8)
                arq_img = tmp_path / "pista_teste.png"
                arq_roi = tmp_path / "pista_roi.png"
                cv2.imwrite(str(arq_img), img)
                cv2.imwrite(str(arq_roi), roi)

                ent = carregar_entrada_arquivo(str(arq_img), str(arq_roi))
                janela._aplicar_entrada(ent)
                self.assertGreaterEqual(janela._resultado["quant"]["n_trincas"], 1)

                out_png = tmp_path / "painel.png"
                out_json = tmp_path / "relatorio.json"
                janela.exportar_painel_png(str(out_png))
                janela.exportar_relatorio_json(str(out_json))
                self.assertTrue(out_png.read_bytes().startswith(b"\x89PNG"))
                rel = json.loads(out_json.read_text(encoding="utf-8"))
                self.assertIn("quantificacao", rel)
                self.assertIn("estatisticas", rel)

            # Amostra real do dataset DNIT (quando disponível localmente)
            amostras_teste = filtrar_dataset_dnit(particao_f="teste")
            if amostras_teste:
                janela.carregar_caminho_externo(amostras_teste[0]["caminho"])
                self.assertIsNotNone(janela._resultado["metricas_gt"])
                self.assertIn("5_gt_comparar", janela._resultado["imagens_bgr"])
        finally:
            janela.close()


if __name__ == "__main__":
    unittest.main()
