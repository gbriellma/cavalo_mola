"""Consistência do contrato de parâmetros e das referências do pipeline."""
import json
import sys
import unittest
from dataclasses import fields
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from pavimento.parametros import Parametros

# Catálogo fixo: não derivar da documentação que o teste deve conferir.
TECNICAS = {
    "Histerese por componentes": "6_1 - processamento de imagens morfologicas.txt",
    "Otsu inteiro": "7_1 - segmentacao.txt",
    "Escurecimento local": "7_1 - segmentacao.txt",
    "Solidez": "9_2 - extracao de caracteristicas.txt",
    "Extensão": "9_2 - extracao de caracteristicas.txt",
    "Razão de aspecto": "9_2 - extracao de caracteristicas.txt",
    "EE lineares": "6_1 - processamento de imagens morfologicas.txt",
    "Dilatação condicional": "6_1 - processamento de imagens morfologicas.txt",
}


class TestDocumentacao(unittest.TestCase):
    def test_todos_parametros_estao_no_json(self):
        dados = json.loads((RAIZ / "params.json").read_text(encoding="utf-8"))
        for campo in fields(Parametros):
            with self.subTest(parametro=campo.name):
                self.assertIn(campo.name, dados)
                self.assertIs(type(dados[campo.name]), int)
        Parametros.carregar(str(RAIZ / "params.json"))

    def test_tecnicas_documentadas_com_origem(self):
        texto = (RAIZ / "docs/REFERENCIAS.md").read_text(encoding="utf-8")
        for tecnica, origem in TECNICAS.items():
            with self.subTest(tecnica=tecnica):
                linhas = [linha for linha in texto.splitlines()
                          if linha.startswith("| " + tecnica + " |")]
                self.assertEqual(len(linhas), 1, tecnica)
                self.assertIn("docs/texto_aulas/" + origem, linhas[0])
                self.assertIn("slide", linhas[0])

    @unittest.skipUnless((RAIZ / "docs/texto_aulas").is_dir(),
                         "Material integral das aulas não é redistribuído")
    def test_arquivos_das_aulas_locais(self):
        for origem in set(TECNICAS.values()):
            with self.subTest(arquivo=origem):
                self.assertTrue((RAIZ / "docs/texto_aulas" / origem).is_file())


if __name__ == "__main__":
    unittest.main()
