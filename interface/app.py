"""Inicialização segura da aplicação desktop PyQt5 + OpenCV."""
from __future__ import annotations

import os
import sys
from typing import List, Optional

# 1) Carrega PyQt5 antes de cv2 para preservar plugins gráficos e fontes nativas do sistema
from PyQt5.QtCore import QLibraryInfo, Qt
from PyQt5.QtWidgets import QApplication

_QT_PLUGINS = QLibraryInfo.location(QLibraryInfo.PluginsPath)

import cv2  # noqa: E402, F401

if _QT_PLUGINS:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _QT_PLUGINS
else:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

from .janela_principal import JanelaPrincipal  # noqa: E402


def criar_aplicacao(argv: Optional[List[str]] = None) -> QApplication:
    inst = QApplication.instance()
    if inst is not None:
        return inst  # type: ignore[return-value]
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    return QApplication(argv if argv is not None else sys.argv)


def executar_interface(caminho_inicial: Optional[str] = None, argv: Optional[List[str]] = None) -> int:
    app = criar_aplicacao(argv)
    janela = JanelaPrincipal(caminho_inicial=caminho_inicial)
    janela.show()
    return app.exec_()
