"""Regressões sintéticas das ferramentas, sem depender do dataset real."""
import copy
from dataclasses import asdict
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import _caminho
from _criterios import avaliar_criterios


class CriteriosTest(unittest.TestCase):
    def relatorios(self):
        base = {"buraco_pixel": {"f1": .08, "iou": .04},
                "trinca_tolerancia_2px": {"f1": .32},
                "quantificacao": {"severidade": {"acuracia": .485}},
                "trinca_deteccao_imagem": {"f1": .93},
                "buraco_deteccao_imagem": {"f1": .33}, "tempo_ms_mediana": 10}
        novo = copy.deepcopy(base)
        novo["buraco_pixel"] = {"f1": .16, "iou": .08}
        novo["trinca_deteccao_imagem"]["f1"] = .91
        novo["tempo_ms_mediana"] = 20
        return base, novo

    def test_limites_inclusivos(self):
        base, novo = self.relatorios()
        self.assertTrue(avaliar_criterios(base, novo)["aprovado"])
        self.assertFalse(avaliar_criterios(base, base)["aprovado"])

    def test_cada_falha_reprova(self):
        base, novo = self.relatorios()
        caminhos = [("buraco_pixel", "f1"), ("buraco_pixel", "iou"),
                    ("trinca_tolerancia_2px", "f1"),
                    ("quantificacao", "severidade", "acuracia"),
                    ("trinca_deteccao_imagem", "f1"), ("buraco_deteccao_imagem", "f1"),
                    ("tempo_ms_mediana",)]
        for caminho in caminhos:
            with self.subTest(caminho=caminho):
                r = copy.deepcopy(novo)
                d = r
                for k in caminho[:-1]:
                    d = d[k]
                d[caminho[-1]] += .001 if caminho == ("tempo_ms_mediana",) else -.001
                self.assertFalse(avaliar_criterios(base, r)["aprovado"])
        self.assertFalse(avaliar_criterios({}, novo)["aprovado"])
        novo["tempo_ms_mediana"] = float("nan")
        self.assertFalse(avaliar_criterios(base, novo)["aprovado"])


# Os gates têm as dependências do projeto; não mascarar sua ausência com skips.
import cv2
import numpy as np
import _comum
import ajustar
import ajustar_pixel
import avaliar_hibrido
import calibrar_severidade
import exportar_sd
from _nota import nota
from pavimento.dataset import particao


class FerramentasTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=RAIZ)
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.cache = self.raiz / "cache"
        self.cache_patch = patch.object(_caminho, "CACHE", str(self.cache))
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def amostra(self, dataset="a", nome="amostra", buraco=False):
        pasta = self.raiz / dataset / nome
        pasta.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(pasta / f"{nome}_RAW.jpg"), np.full((32, 32, 3), 100, np.uint8)))
        for classe in ("CRACK", "POTHOLE", "LANE"):
            m = np.zeros((32, 32), np.uint8)
            if classe == "LANE" or (classe == "POTHOLE" and buraco):
                m[:] = 255
            self.assertTrue(cv2.imwrite(str(pasta / f"{nome}_{classe}.png"), m))
        return pasta

    def test_hibrido_dataset_ausente_aborta_sem_relatorio_ou_pool(self):
        with patch.dict(os.environ, {"CAVALO_DATASET": str(self.raiz / "ausente")}), \
                patch.object(_caminho, "SAIDAS", str(self.raiz)), \
                patch.object(sys, "argv", ["avaliar_hibrido.py"]), \
                patch.object(sys, "stderr", new_callable=io.StringIO) as stderr, \
                patch.object(avaliar_hibrido, "Pool") as pool, \
                patch.object(avaliar_hibrido.json, "dump") as gravar:
            with self.assertRaises(SystemExit) as exc:
                avaliar_hibrido.main()
            self.assertNotEqual(exc.exception.code, 0)
            self.assertRegex(stderr.getvalue(), "Dataset.*CAVALO_DATASET")
            pool.assert_not_called()
            pool.return_value.__enter__.return_value.map.assert_not_called()
            gravar.assert_not_called()
            self.assertFalse((self.raiz / "avaliacao_hibrido.json").exists())

    def test_hibrido_dataset_vazio_aborta_sem_relatorio_ou_pool(self):
        dataset = self.raiz / "vazio"
        dataset.mkdir()
        with patch.dict(os.environ, {"CAVALO_DATASET": str(dataset)}), \
                patch.object(_caminho, "SAIDAS", str(self.raiz)), \
                patch.object(sys, "argv", ["avaliar_hibrido.py"]), \
                patch.object(sys, "stderr", new_callable=io.StringIO) as stderr, \
                patch.object(avaliar_hibrido, "validar_amostras", wraps=_comum.validar_amostras) as validar, \
                patch.object(avaliar_hibrido, "Pool") as pool, \
                patch.object(avaliar_hibrido.json, "dump") as gravar:
            with self.assertRaises(SystemExit) as exc:
                avaliar_hibrido.main()
            self.assertNotEqual(exc.exception.code, 0)
            self.assertIn("Nenhuma amostra", stderr.getvalue())
            validar.assert_called_once_with([])
            pool.assert_not_called()
            pool.return_value.__enter__.return_value.map.assert_not_called()
            gravar.assert_not_called()
            self.assertFalse((self.raiz / "avaliacao_hibrido.json").exists())

    def test_guarda_treino(self):
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        treino = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        with self.assertRaisesRegex(ValueError, "teste"):
            ajustar._init([self.raiz / teste])
        with self.assertRaisesRegex(ValueError, "teste"):
            _comum.somente_treino([self.raiz / "teste" / treino])
        self.assertEqual(_comum.somente_treino([treino]), [treino])

    def test_grade_opcional_e_exclusoes(self):
        self.assertEqual(ajustar.montar_grade(SimpleNamespace()), ajustar.GRADE_PADRAO)
        modulo = SimpleNamespace(GRADE_BUSCA={"novo_parametro": [1, 2], "raio": [3],
            "sev_alta_1000": [999], "peso_trinca": [3], "indice_dano_peso": [4]})
        grade = ajustar.montar_grade(modulo)
        self.assertEqual(grade["novo_parametro"], [1, 2])
        self.assertEqual(grade["raio"], [3])
        for k in ("sev_alta_1000", "peso_trinca", "indice_dano_peso"):
            self.assertNotIn(k, grade)
        with patch.object(ajustar.modulo_parametros, "GRADE_BUSCA", {"extra": [5]}, create=True):
            self.assertIn("extra", ajustar.montar_grade())

    def test_ajuste_reutiliza_baseline_original_entre_execucoes(self):
        base, atual = CriteriosTest().relatorios()
        base["particao"] = "treino"
        base["params"] = asdict(ajustar.Parametros())
        atual["buraco_pixel"] = {"f1": .18, "iou": .10}
        arquivo = self.raiz / "baseline_treino.json"
        arquivo.write_text(json.dumps(base), encoding="utf-8")
        original = arquivo.read_bytes()
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        pasta = self.amostra(nome=nome)
        with patch.object(_caminho, "SAIDAS", str(self.raiz)), \
                patch.object(_caminho, "dataset_atual", return_value=str(pasta.parent)), \
                patch.object(sys, "argv", ["ajustar.py", "--passadas", "0"]), \
                patch.object(ajustar.Parametros, "salvar") as salvar, \
                patch.object(ajustar, "avaliar_pastas", return_value=atual) as avaliar, \
                patch.object(ajustar, "Pool") as pool:
            pool.return_value.__enter__.return_value.apply.return_value = (1.0, {})
            for _ in range(2):
                ajustar.main()
            self.assertEqual(salvar.call_count, 2)
            self.assertEqual(avaliar.call_count, 2)  # Só avaliação final por execução.
            avaliar.assert_called_with([str(pasta)], unittest.mock.ANY, processos=1)
        self.assertEqual(arquivo.read_bytes(), original)

    def test_baseline_severidade_atual_aceito(self):
        base, _ = CriteriosTest().relatorios()
        base.update(particao="treino", params=asdict(ajustar.Parametros()))
        arquivo = self.raiz / "baseline_treino.json"
        arquivo.write_text(json.dumps(base), encoding="utf-8")
        with patch.object(_caminho, "SAIDAS", str(self.raiz)):
            self.assertEqual(ajustar.carregar_baseline(self.raiz), base)

    def test_baseline_severidade_divergente_recusado_no_script(self):
        base, _ = CriteriosTest().relatorios()
        divergentes = {"sev_min_1000": 400, "sev_media_1000": 500, "sev_alta_1000": 800}
        base.update(particao="treino", params={**asdict(ajustar.Parametros()), **divergentes})
        (self.raiz / "baseline_treino.json").write_text(json.dumps(base), encoding="utf-8")
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        pasta = self.amostra(nome=nome)
        with patch.object(_caminho, "SAIDAS", str(self.raiz)), \
                patch.object(_caminho, "dataset_atual", return_value=str(pasta.parent)), \
                patch.object(sys, "argv", ["ajustar.py", "--passadas", "0"]), \
                patch.object(ajustar, "Pool") as pool:
            with self.assertRaisesRegex(ValueError, "Baseline.*incompatível") as exc:
                ajustar.main()
            for chave in divergentes:
                self.assertIn(chave, str(exc.exception))
            pool.assert_not_called()
        codigo = ("import runpy, sys; import _caminho; "
                  "_caminho.SAIDAS = sys.argv[1]; "
                  "script = sys.argv[2]; sys.argv = [script, '--passadas', '0']; "
                  "runpy.run_path(script, run_name='__main__')")
        resultado = subprocess.run(
            [sys.executable, "-c", codigo, str(self.raiz), str(RAIZ / "scripts" / "ajustar.py")],
            cwd=RAIZ, env={**os.environ, "CAVALO_DATASET": str(pasta.parent),
                           "PYTHONPATH": str(RAIZ / "scripts"), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(resultado.returncode, 0)
        self.assertIn("Erro: Baseline", resultado.stderr)
        self.assertNotIn("Traceback", resultado.stderr)
        for chave in divergentes:
            self.assertIn(chave, resultado.stderr)

    def test_baseline_severidade_ausente_ou_chave_incompativel(self):
        base, _ = CriteriosTest().relatorios()
        base["particao"] = "treino"
        atuais = asdict(ajustar.Parametros())
        sem_sev = {k: v for k, v in atuais.items() if not k.startswith("sev_")}
        casos = [(None, [k for k in atuais if k.startswith("sev_")]),
                 (sem_sev, [k for k in atuais if k.startswith("sev_")]),
                 ({k: v for k, v in atuais.items() if k != "sev_min_1000"}, ["sev_min_1000"]),
                 ({**atuais, "sev_antiga_1000": 1}, ["sev_antiga_1000"])]
        for params, chaves in casos:
            with self.subTest(params=params):
                documento = {**base, "params": params} if params is not None else base
                (self.raiz / "baseline_treino.json").write_text(json.dumps(documento), encoding="utf-8")
                with patch.object(_caminho, "SAIDAS", str(self.raiz)):
                    with self.assertRaisesRegex(ValueError, "incompatível.*divergentes ou ausentes") as exc:
                        ajustar.carregar_baseline(self.raiz)
                for chave in chaves:
                    self.assertIn(chave, str(exc.exception))

    def test_baseline_ausente_pendente_incompleto_ou_teste_aborta_antes_do_pool(self):
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        pasta = self.amostra(nome=nome)
        arquivo = self.raiz / "baseline_treino.json"
        base, _ = CriteriosTest().relatorios()
        with patch.object(_caminho, "SAIDAS", str(self.raiz)), \
                patch.object(_caminho, "dataset_atual", return_value=str(pasta.parent)), \
                patch.object(sys, "argv", ["ajustar.py", "--passadas", "0"]), \
                patch.object(ajustar, "avaliar_pastas") as avaliar, \
                patch.object(ajustar, "Pool") as pool, \
                patch.object(ajustar.Parametros, "salvar") as salvar:
            for documento in (None, {"status": "pendente", "particao": "treino"},
                              {"particao": "treino"}, {**base, "particao": "teste"}):
                with self.subTest(documento=documento):
                    if documento is not None:
                        arquivo.write_text(json.dumps(documento), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "[Bb]aseline"):
                        ajustar.main()
            pool.assert_not_called()
            avaliar.assert_not_called()
            salvar.assert_not_called()

    def test_guarda_treino_confere_destino_resolvido(self):
        treino = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        alias = self.raiz / "treino" / treino
        for destino in (self.raiz / teste, self.raiz / "teste" / treino):
            with self.subTest(destino=destino), patch.object(Path, "resolve", return_value=destino):
                with self.assertRaisesRegex(ValueError, "Vazamento recusado"):
                    ajustar._init([alias])

    def test_guarda_treino_link_real(self):
        treino = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        destino = self.amostra(nome=teste)
        alias = self.raiz / treino
        try:
            alias.symlink_to(destino, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Sistema sem permissão para criar link simbólico: {exc}")
        with self.assertRaisesRegex(ValueError, "Vazamento recusado"):
            ajustar._init([alias])

    def test_manifesto_confere_destino_de_cada_arquivo(self):
        treino = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        pasta = self.amostra(nome=treino)
        _comum.preparar(pasta)  # A guarda deve valer mesmo com cache válido.
        resolver = Path.resolve
        # Cobertura independente da permissão de symlink no Windows.
        for entrada in _comum.arquivos_entrada(pasta):
            for pai_destino in (self.raiz / teste, self.raiz / "teste" / treino):
                destino = pai_destino / entrada.name

                def resolver_entrada(caminho, *args, **kwargs):
                    if caminho == entrada:
                        return destino
                    return resolver(caminho, *args, **kwargs)

                with self.subTest(entrada=entrada.name, destino=destino), \
                        patch.object(Path, "resolve", resolver_entrada), \
                        patch.object(cv2, "imread", side_effect=AssertionError("leu antes da guarda")):
                    for func in (_comum.manifesto, _comum.carregar, _comum.preparar):
                        with self.assertRaisesRegex(ValueError, "Vazamento recusado") as exc:
                            func(pasta)
                        self.assertIn(entrada.name, str(exc.exception))
                        self.assertIn(str(destino), str(exc.exception))
                    with self.assertRaisesRegex(ValueError, "Amostras inválidas"):
                        _comum.validar_amostras([pasta])
                    with self.assertRaisesRegex(ValueError, "Vazamento recusado"):
                        ajustar._init([pasta])

    def test_guarda_links_reais_nos_arquivos_com_cache(self):
        treino = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        pasta = self.amostra(nome=treino)
        destino = self.amostra("b", nome=teste, buraco=True)
        for entrada, origem in zip(_comum.arquivos_entrada(pasta), _comum.arquivos_entrada(destino)):
            _comum.preparar(pasta)
            conteudo = entrada.read_bytes()
            entrada.unlink()
            try:
                try:
                    entrada.symlink_to(origem)
                except OSError as exc:
                    self.skipTest(f"Sistema sem permissão para criar link simbólico: {exc}")
                with self.subTest(entrada=entrada.name):
                    for func in (_comum.carregar, _comum.preparar):
                        with self.assertRaisesRegex(ValueError, "Vazamento recusado"):
                            func(pasta)
                    with self.assertRaisesRegex(ValueError, "Vazamento recusado"):
                        ajustar._init([pasta])
            finally:
                if entrada.is_symlink():
                    entrada.unlink()
                entrada.write_bytes(conteudo)

    def test_manifesto_registra_destinos_e_permite_avaliar_teste(self):
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        pasta = self.amostra(nome=teste)
        manifesto = _comum.manifesto(pasta)
        self.assertEqual(manifesto["particao"], "teste")
        for arquivo, entrada in zip(manifesto["arquivos"], _comum.arquivos_entrada(pasta)):
            self.assertEqual(arquivo["caminho"], str(entrada.resolve()))
        self.assertEqual(_comum.carregar(pasta).nome, teste)

    def test_calibracao_preflight_antes_de_subamostrar(self):
        nomes = [f"img{i}" for i in range(100) if particao(f"img{i}") == "treino"][:2]
        pastas = [self.amostra(nome=nome) for nome in nomes]
        # A segunda amostra seria excluída por [::2], mas deve ser validada.
        ausente = pastas[1] / f"{nomes[1]}_LANE.png"
        ausente.unlink()
        with patch.object(_caminho, "dataset_atual", return_value=str(pastas[0].parent)), \
                patch.object(calibrar_severidade, "Pool") as pool, \
                patch.object(calibrar_severidade.Parametros, "carregar") as parametros:
            with self.assertRaisesRegex(ValueError, ausente.name):
                calibrar_severidade.main()
            pool.assert_not_called()
            parametros.assert_not_called()

    def test_pixel_dataset_ausente_com_cache_nao_cria_pool(self):
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        _comum.preparar(self.amostra(nome=nome))
        with patch.dict(os.environ, {"CAVALO_DATASET": str(self.raiz / "ausente")}), \
                patch.object(sys, "argv", ["ajustar_pixel.py"]), \
                patch.object(ajustar_pixel, "Pool") as pool, \
                patch.object(ajustar_pixel, "carregar_classificador") as classificador:
            with self.assertRaisesRegex(FileNotFoundError, "CAVALO_DATASET"):
                ajustar_pixel.main()
            pool.assert_not_called()
            classificador.assert_not_called()
        r = self.executar("ajustar_pixel.py", self.raiz / "ausente", cache=self.cache)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("CAVALO_DATASET", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_pixel_preflight_com_cache_usa_dataset_atual(self):
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        pasta = self.amostra(nome=nome)
        _comum.preparar(pasta)
        ausente = pasta / f"{nome}_LANE.png"
        ausente.unlink()
        with patch.dict(os.environ, {"CAVALO_DATASET": str(pasta.parent)}), \
                patch.object(_caminho, "DATASET", str(self.raiz / "antigo")), \
                patch.object(sys, "argv", ["ajustar_pixel.py"]), \
                patch.object(ajustar_pixel, "Pool") as pool, \
                patch.object(ajustar_pixel.Parametros, "carregar") as parametros:
            with self.assertRaisesRegex(ValueError, ausente.name):
                ajustar_pixel.main()
            pool.assert_not_called()
            parametros.assert_not_called()

    def test_pixel_initializer_recebe_pastas_validadas(self):
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        pasta = self.amostra(nome=nome)
        with patch.object(_caminho, "DATASET", str(self.raiz / "antigo")), \
                patch.object(ajustar_pixel, "carregar_classificador"), \
                patch.object(ajustar_pixel.Parametros, "carregar"), \
                patch.object(ajustar_pixel, "preparar", wraps=_comum.preparar) as preparar:
            ajustar_pixel._init([pasta])
            preparar.assert_called_once_with(pasta)
        teste = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        with self.assertRaisesRegex(ValueError, "Vazamento recusado"):
            ajustar_pixel._init([self.raiz / teste])

    def test_exportacao_sem_dataset_com_cache_vazio_ou_existente(self):
        self.cache.mkdir()
        destino = self.raiz / "sd"
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        for com_cache in (False, True):
            if com_cache:
                _comum.preparar(self.amostra(nome=nome))
            with self.subTest(com_cache=com_cache), \
                    patch.dict(os.environ, {"CAVALO_DATASET": str(self.raiz / "ausente")}), \
                    patch.object(sys, "argv", ["exportar_sd.py", "--destino", str(destino)]), \
                    patch.object(exportar_sd, "preparar") as preparar:
                with self.assertRaisesRegex(FileNotFoundError, "CAVALO_DATASET"):
                    exportar_sd.main()
                preparar.assert_not_called()
                self.assertFalse(destino.exists())
            r = self.executar("exportar_sd.py", self.raiz / "ausente", cache=self.cache,
                              argumentos=["--destino", str(destino)])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("CAVALO_DATASET", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(destino.exists())

    def test_exportacao_preflight_e_env_atual(self):
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "teste")
        pasta = self.amostra(nome=nome)
        _comum.preparar(pasta)
        with patch.dict(os.environ, {"CAVALO_DATASET": str(pasta.parent)}), \
                patch.object(_caminho, "DATASET", str(self.raiz / "antigo")):
            self.assertEqual(exportar_sd.amostras_dnit("teste"), [str(pasta)])
            ausente = pasta / f"{nome}_LANE.png"
            ausente.unlink()
            destino = self.raiz / "sd"
            with patch.object(sys, "argv", ["exportar_sd.py", "--destino", str(destino)]), \
                    patch.object(exportar_sd, "preparar") as preparar:
                with self.assertRaisesRegex(ValueError, ausente.name):
                    exportar_sd.main()
                preparar.assert_not_called()
                self.assertFalse(destino.exists())
        vazio = self.raiz / "vazio"
        vazio.mkdir()
        with patch.dict(os.environ, {"CAVALO_DATASET": str(vazio)}):
            with self.assertRaisesRegex(ValueError, "Nenhuma amostra"):
                exportar_sd.amostras_dnit("teste")

    def test_nota_monotonica(self):
        r = {"trinca_tol": (5, 5, 5, 5), "buraco_px": (2, 8, 2, 8),
             "tem_trinca_pred": True, "tem_trinca_gt": True,
             "tem_buraco_pred": True, "tem_buraco_gt": True,
             "quant": {"comprimento_trincas_px": 10, "area_buracos_px": 10}}
        melhor = copy.deepcopy(r)
        melhor["buraco_px"] = (8, 2, 8, 2)
        self.assertGreater(nota([melhor], [(10, 10)])[0], nota([r], [(10, 10)])[0])
        trinca = copy.deepcopy(r)
        trinca["trinca_tol"] = (8, 2, 8, 2)
        self.assertGreater(nota([trinca], [(10, 10)])[0], nota([r], [(10, 10)])[0])
        self.assertGreater(nota([melhor], [(10, 10)])[0] - nota([r], [(10, 10)])[0],
                           nota([trinca], [(10, 10)])[0] - nota([r], [(10, 10)])[0])

    def test_env_e_cache_origem(self):
        a = self.amostra("a")
        b = self.amostra("b", buraco=True)
        with patch.dict(os.environ, {"CAVALO_DATASET": str(a.parent)}):
            self.assertEqual(Path(_caminho.dataset_atual()), a.parent)
            da = _comum.preparar(Path(_caminho.dataset_atual()) / a.name)
        with patch.dict(os.environ, {"CAVALO_DATASET": str(b.parent)}):
            db = _comum.preparar(Path(_caminho.dataset_atual()) / b.name)
        self.assertFalse(da[4].any())
        self.assertTrue(db[4].all())
        with np.load(self.cache / "amostra.npz") as z:
            m = json.loads(z["manifesto"].item())
        self.assertEqual(m["dataset"], str(b.parent.resolve()))
        self.assertEqual(len(m["arquivos"]), 4)

    def test_cache_reutiliza_e_invalida_arquivo(self):
        pasta = self.amostra()
        _comum.preparar(pasta)
        with patch.object(_comum, "carregar", side_effect=AssertionError("cache deveria ser usado")):
            _comum.preparar(pasta)
        f = pasta / "amostra_POTHOLE.png"
        antes = f.stat()
        cv2.imwrite(str(f), np.full((32, 32), 255, np.uint8))
        os.utime(f, ns=(antes.st_atime_ns, antes.st_mtime_ns + 1_000_000_000))
        self.assertTrue(_comum.preparar(pasta)[4].all())
        raw = pasta / "amostra_RAW.jpg"
        st = raw.stat()
        os.utime(raw, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        with patch.object(_comum, "carregar", wraps=_comum.carregar) as carregar:
            _comum.preparar(pasta)
            carregar.assert_called_once()

    def test_cache_antigo_descartado(self):
        pasta = self.amostra(buraco=True)
        self.cache.mkdir()
        zeros = np.zeros((16, 16), np.uint8)
        np.savez_compressed(self.cache / "amostra.npz", g=zeros, roi=zeros, gt=zeros, gb=zeros)
        self.assertTrue(_comum.preparar(pasta)[4].all())
        with np.load(self.cache / "amostra.npz") as z:
            self.assertIn("manifesto", z.files)

    def test_mascara_obrigatoria_mesmo_com_cache(self):
        pasta = self.amostra()
        _comum.preparar(pasta)
        (pasta / "amostra_CRACK.png").unlink()
        for func in (_comum.carregar, _comum.preparar):
            with self.subTest(func=func.__name__), self.assertRaisesRegex(ValueError, "amostra_CRACK.png"):
                func(pasta)
        outra = self.amostra("b", "segunda")
        (outra / "segunda_LANE.png").unlink()
        with self.assertRaises(ValueError) as exc:
            _comum.validar_amostras([pasta, outra])
        self.assertIn("amostra_CRACK.png", str(exc.exception))
        self.assertIn("segunda_LANE.png", str(exc.exception))

    def test_raw_sem_mascaras_e_dimensoes(self):
        pasta = self.amostra()
        for f in pasta.glob("*.png"):
            f.unlink()
        with self.assertRaisesRegex(ValueError, "amostra_CRACK.png"):
            _comum.carregar(pasta)
        pasta = self.amostra("dimensoes")
        cv2.imwrite(str(pasta / "amostra_LANE.png"), np.zeros((4, 4), np.uint8))
        with self.assertRaises(ValueError):
            _comum.validar_amostras([pasta])

    def executar(self, script, dataset, cache=None, argumentos=()):
        env = {**os.environ, "CAVALO_DATASET": str(dataset), "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(RAIZ / "scripts")}
        args = [sys.executable, str(RAIZ / "scripts" / script), *argumentos]
        if cache is not None:
            codigo = ("import runpy, sys; import _caminho; "
                      "_caminho.CACHE = sys.argv[1]; "
                      "script = sys.argv[2]; sys.argv = [script, *sys.argv[3:]]; "
                      "runpy.run_path(script, run_name='__main__')")
            args = [sys.executable, "-c", codigo, str(cache), str(RAIZ / "scripts" / script), *argumentos]
        return subprocess.run(args,
                              env=env, cwd=RAIZ, capture_output=True, text=True, timeout=30)

    def test_falha_limpa_sem_dataset(self):
        for script in ("ajustar.py", "avaliar.py", "diagnosticar.py", "calibrar_severidade.py", "ajustar_pixel.py"):
            with self.subTest(script=script):
                r = self.executar(script, self.raiz / "ausente")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("CAVALO_DATASET", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_scripts_abortam_amostra_invalida(self):
        nome = next(f"img{i}" for i in range(100) if particao(f"img{i}") == "treino")
        pasta = self.amostra(nome=nome)
        _comum.preparar(pasta)
        (pasta / f"{nome}_LANE.png").unlink()
        for script in ("ajustar.py", "avaliar.py", "diagnosticar.py", "calibrar_severidade.py", "ajustar_pixel.py"):
            with self.subTest(script=script):
                # Reproduz cache já existente sem escrever em outputs/cache.
                if script == "avaliar.py":
                    args = [sys.executable, str(RAIZ / "scripts" / script), "--particao", "treino"]
                    r = subprocess.run(args, cwd=RAIZ, env={**os.environ, "CAVALO_DATASET": str(pasta.parent),
                                        "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=30)
                else:
                    r = self.executar(script, pasta.parent, cache=self.cache if script == "ajustar_pixel.py" else None)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(f"{nome}_LANE.png", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_diagnosticar_importa_sem_executar(self):
        with patch.object(_caminho, "dataset_atual", side_effect=AssertionError("executou ao importar")):
            import diagnosticar
            importlib.reload(diagnosticar)


if __name__ == "__main__":
    unittest.main()
