"""Testes da interface gráfica web: paridade com o núcleo, data packs e API HTTP."""
from __future__ import annotations

import base64
import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for p in (str(RAIZ), str(RAIZ / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from interface.cenarios import CENARIOS, carregar_cenario
from interface.pipeline_didatico import _motivo_mais_frequente, executar_pipeline_didatico
from interface.servidor import criar_servidor
from pavimento.nucleo import analisar, intensidade_hsi, reduzir_2x, reduzir_mascara_2x
from pavimento.parametros import Parametros


class TestInterfaceDidatica(unittest.TestCase):
    def test_paridade_todos_cenarios_e_data_packs(self):
        p_final = Parametros.carregar(str(RAIZ / "params.json"))
        for c in CENARIOS:
            with self.subTest(cenario=c.id):
                bgr, roi_full, info = carregar_cenario(c.id)
                res = executar_pipeline_didatico(bgr, roi_full, p_final, aplicar_reducao_2x=info.reduzir_padrao)

                g = intensidade_hsi(bgr)
                roi = roi_full
                if info.reduzir_padrao:
                    g = reduzir_2x(g)
                    roi = reduzir_mascara_2x(roi, "maioria")
                oficial = analisar(g, roi, p_final)

                self.assertEqual(res["quant"], oficial["quant"])
                self.assertEqual(res["estatisticas"]["limiar_textura"], oficial["limiar_textura"])
                self.assertEqual(res["estatisticas"]["limiar_otsu"], oficial["limiar_otsu"])

                # Confere decodificação exata dos data_packs (R=ch2, G=ch1, B=ch0 no OpenCV)
                p1_bytes = base64.b64decode(res["data_packs"]["pack1"].split(",", 1)[1])
                p1 = cv2.imdecode(np.frombuffer(p1_bytes, np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(p1[..., 2], g)
                np.testing.assert_array_equal(p1[..., 1], np.clip(oficial["media"], 0, 255).astype(np.uint8))
                np.testing.assert_array_equal(p1[..., 0], np.clip(oficial["desvio"], 0, 255).astype(np.uint8))

    def test_servidor_http_rotas_completas(self):
        srv = criar_servidor("127.0.0.1", 0)
        porta = srv.server_address[1]
        base = f"http://127.0.0.1:{porta}"
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            # GET / e estáticos
            with urlopen(f"{base}/") as r:
                self.assertEqual(r.status, 200)
                self.assertIn("Detecção de Danos em Pavimentos", r.read().decode("utf-8"))
            with urlopen(f"{base}/static/app.js") as r:
                self.assertEqual(r.status, 200)
            with urlopen(f"{base}/static/styles.css") as r:
                self.assertEqual(r.status, 200)
            with urlopen(f"{base}/static/didatica.js") as r:
                self.assertEqual(r.status, 200)
                self.assertIn("PAVIMENTO_INFO", r.read().decode("utf-8"))

            # GET /api/init
            with urlopen(f"{base}/api/init") as r:
                init = json.loads(r.read().decode("utf-8"))
                self.assertEqual(len(init["cenarios"]), len(CENARIOS))
                self.assertIn("final", init["presets"])
                self.assertIn("slides_anterior", init["presets"])

            # POST /api/analisar
            req = Request(
                f"{base}/api/analisar",
                data=json.dumps({"cenario_id": "didatico_buraco_faixa"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req) as r:
                out = json.loads(r.read().decode("utf-8"))
                self.assertEqual(out["quant"]["n_buracos"], 1)
                self.assertIn("5_final", out["imagens"])

            # POST /api/upload + /api/upload_roi + /api/analisar + /api/exportar_painel
            img = np.full((120, 160, 3), 150, dtype=np.uint8)
            img[58:60, 20:140] = 30
            _, buf_img = cv2.imencode(".png", img)
            roi = np.full((120, 160), 255, dtype=np.uint8)
            _, buf_roi = cv2.imencode(".png", roi)

            req_up = Request(
                f"{base}/api/upload",
                data=json.dumps({
                    "imagem_b64": "data:image/png;base64," + base64.b64encode(buf_img.tobytes()).decode("ascii"),
                    "nome": "teste.png",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req_up) as r:
                up = json.loads(r.read().decode("utf-8"))
                uid = up["upload_id"]

            req_roi = Request(
                f"{base}/api/upload_roi",
                data=json.dumps({
                    "upload_id": uid,
                    "roi_b64": "data:image/png;base64," + base64.b64encode(buf_roi.tobytes()).decode("ascii"),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req_roi) as r:
                self.assertTrue(json.loads(r.read().decode("utf-8"))["tem_roi_upload"])

            req_exp = Request(
                f"{base}/api/exportar_painel",
                data=json.dumps({"upload_id": uid, "reduzir_2x": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req_exp) as r:
                png = r.read()
                self.assertTrue(png.startswith(b"\x89PNG"))

            # GET /api/dataset + análise de amostra real DNIT (quando extraída localmente)
            with urlopen(f"{base}/api/dataset?particao=teste&limit=5") as r:
                ds = json.loads(r.read().decode("utf-8"))
                if ds["disponivel"] and ds["amostras"]:
                    nome = ds["amostras"][0]["nome"]
                    req_dnit = Request(
                        f"{base}/api/analisar",
                        data=json.dumps({"amostra_dnit": nome}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(req_dnit) as rd:
                        out_dnit = json.loads(rd.read().decode("utf-8"))
                        self.assertIsNotNone(out_dnit["metricas_gt"])
                        self.assertIn("5_gt_comparar", out_dnit["imagens"])
                        self.assertIn("5_gt_rotulo", out_dnit["imagens"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_resumo_didatico_completo(self):
        esquema = {
            "etapa1": {
                "pct_fora_roi": (int, float),
                "fator_pixels": (int,),
                "pixels_fora_roi": (int,),
            },
            "etapa2": {
                "s_xy_mediana_roi": (int,),
                "s_xy_p95_roi": (int,),
                "tempo_nucleo_ms": (int, float),
            },
            "etapa3a": {
                "px_otsu": (int,),
                "px_histerese": (int,),
                "pct_ruido_removido": (int, float),
                "otsu_vs_histerese_x": (int, float),
            },
            "etapa3b": {
                "pct_pista_cand": (int, float),
                "mediana_textura": (int,),
                "mad_textura": (int,),
                "limiar_textura": (int,),
            },
            "etapa4": {
                "pct_removido_abertura": (int, float),
                "px_removido_abertura": (int,),
                "px_preenchido_miolo": (int,),
                "px_religado_trincas": (int,),
            },
            "etapa5": {
                "trincas_rejeitadas_pct": (int, float),
                "buracos_rejeitados_pct": (int, float),
                "motivo_trinca_top": (str,),
                "motivo_buraco_top": (str,),
                "severidade_label": (str,),
            },
        }

        def validar_resumo(resumo: dict) -> None:
            self.assertIsInstance(resumo, dict)
            self.assertEqual(set(resumo.keys()), set(esquema.keys()))
            for etapa, campos in esquema.items():
                bloco = resumo[etapa]
                self.assertIsInstance(bloco, dict)
                self.assertEqual(set(bloco.keys()), set(campos.keys()), msg=f"Chaves divergentes em {etapa}")
                for chave, tipos in campos.items():
                    val = bloco[chave]
                    self.assertNotIsInstance(val, bool, msg=f"{etapa}.{chave} não pode ser bool")
                    self.assertIsInstance(val, tipos, msg=f"Tipo inválido em {etapa}.{chave}: {type(val)}")
            self.assertIn(resumo["etapa1"]["fator_pixels"], (1, 4))
            self.assertIn(resumo["etapa5"]["severidade_label"], {"Nenhum", "Baixa", "Média", "Alta"})
            self.assertTrue(resumo["etapa5"]["motivo_trinca_top"])
            self.assertTrue(resumo["etapa5"]["motivo_buraco_top"])

        self.assertEqual(_motivo_mais_frequente([]), "Nenhum componente rejeitado")
        self.assertEqual(
            _motivo_mais_frequente([{"motivos": ["A", "B"]}, {"motivos": ["A"]}]),
            "A",
        )

        p_final = Parametros.carregar(str(RAIZ / "params.json"))
        cenarios_alvo = ("didatico_buraco_faixa", "didatico_sombra_trinca")
        for cid in cenarios_alvo:
            with self.subTest(pipeline_cenario=cid):
                bgr, roi_full, info = carregar_cenario(cid)
                res = executar_pipeline_didatico(bgr, roi_full, p_final, aplicar_reducao_2x=info.reduzir_padrao)
                self.assertIn("resumo_didatico", res)
                validar_resumo(res["resumo_didatico"])

        srv = criar_servidor("127.0.0.1", 0)
        porta = srv.server_address[1]
        base = f"http://127.0.0.1:{porta}"
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            for cid in cenarios_alvo:
                with self.subTest(http_cenario=cid):
                    req = Request(
                        f"{base}/api/analisar",
                        data=json.dumps({"cenario_id": cid}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(req) as r:
                        self.assertEqual(r.status, 200)
                        out = json.loads(r.read().decode("utf-8"))
                        self.assertIn("resumo_didatico", out)
                        validar_resumo(out["resumo_didatico"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
