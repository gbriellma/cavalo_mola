"""Diálogo nativo para explorar e pré-visualizar amostras reais do Dataset DNIT."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pavimento.dataset import carregar as carregar_amostra_dnit

from .dados import definir_pasta_dataset, filtrar_dataset_dnit, resumo_dataset_dnit
from .tema import bgr_para_qimage


class DnitExplorerDialog(QDialog):
    """Explorador interativo das amostras reais da base Cracks and Potholes (DNIT)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Explorar Dataset Real DNIT (Cracks and Potholes)")
        self.resize(920, 580)
        self.amostra_escolhida: Optional[str] = None

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(16, 14, 16, 14)
        raiz.setSpacing(10)

        # Cabeçalho compacto
        topo = QHBoxLayout()
        lbl_t = QLabel("Amostras Reais do DNIT (Foto + ROI + Gabarito)")
        lbl_t.setStyleSheet("font-size:15px; font-weight:700; color:#0F172A;")
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("font-size:12px; color:#64748B; font-weight:600;")
        btn_pasta = QPushButton("Localizar Pasta v1…")
        btn_pasta.clicked.connect(self._escolher_pasta)
        topo.addWidget(lbl_t)
        topo.addStretch(1)
        topo.addWidget(self._lbl_status)
        topo.addWidget(btn_pasta)
        raiz.addLayout(topo)

        # Barra de filtros
        filtros = QHBoxLayout()
        filtros.setSpacing(8)
        self._txt_busca = QLineEdit()
        self._txt_busca.setPlaceholderText("Filtrar por rodovia ou código (ex.: DF_251, RS_386, 992851)…")
        self._txt_busca.textChanged.connect(self._atualizar_lista)

        self._cmb_part = QComboBox()
        self._cmb_part.addItem("Partição: Teste (Inéditas)", "teste")
        self._cmb_part.addItem("Partição: Treino", "treino")
        self._cmb_part.addItem("Todas as Partições", "todas")
        self._cmb_part.currentIndexChanged.connect(self._atualizar_lista)

        self._cmb_estado = QComboBox()
        self._cmb_estado.currentIndexChanged.connect(self._atualizar_lista)

        filtros.addWidget(self._txt_busca, 1)
        filtros.addWidget(self._cmb_part)
        filtros.addWidget(self._cmb_estado)
        raiz.addLayout(filtros)

        # Corpo: tabela à esquerda + pré-visualização à direita
        corpo = QHBoxLayout()
        corpo.setSpacing(12)

        self._tabela = QTableWidget(0, 3)
        self._tabela.setHorizontalHeaderLabels(["Rodovia", "Partição", "Amostra"])
        self._tabela.verticalHeader().setVisible(False)
        self._tabela.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tabela.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tabela.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tabela.setAlternatingRowColors(True)
        hdr = self._tabela.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        self._tabela.itemSelectionChanged.connect(self._atualizar_preview)
        self._tabela.cellDoubleClicked.connect(self._confirmar_atual)
        corpo.addWidget(self._tabela, 3)

        card_prev = QFrame()
        card_prev.setProperty("card", True)
        lay_prev = QVBoxLayout(card_prev)
        lay_prev.setContentsMargins(12, 12, 12, 12)
        lay_prev.setSpacing(8)

        lbl_prev_t = QLabel("Pré-visualização (RAW + ROI + Gabarito)")
        lbl_prev_t.setStyleSheet("font-size:12px; font-weight:700; color:#334155;")
        self._lbl_img = QLabel("Selecione uma amostra na lista")
        self._lbl_img.setAlignment(Qt.AlignCenter)
        self._lbl_img.setMinimumSize(340, 220)
        self._lbl_img.setStyleSheet("background:#0E131A; color:#94A3B8; border-radius:6px;")

        self._lbl_info_prev = QLabel("Laranja = Trinca real · Verde = Buraco real · Ciano = Pista")
        self._lbl_info_prev.setWordWrap(True)
        self._lbl_info_prev.setStyleSheet("font-size:11px; color:#64748B;")

        lay_prev.addWidget(lbl_prev_t)
        lay_prev.addWidget(self._lbl_img, 1)
        lay_prev.addWidget(self._lbl_info_prev)
        corpo.addWidget(card_prev, 2)
        raiz.addLayout(corpo, 1)

        # Rodapé
        rodape = QHBoxLayout()
        rodape.addStretch(1)
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.clicked.connect(self.reject)
        self._btn_abrir = QPushButton("Analisar Amostra Selecionada")
        self._btn_abrir.setProperty("variant", "primary")
        self._btn_abrir.setEnabled(False)
        self._btn_abrir.clicked.connect(self._confirmar_atual)
        rodape.addWidget(btn_cancelar)
        rodape.addWidget(self._btn_abrir)
        raiz.addLayout(rodape)

        self._recarregar_estados()
        self._atualizar_lista()

    def _recarregar_estados(self) -> None:
        info = resumo_dataset_dnit()
        self._cmb_estado.blockSignals(True)
        self._cmb_estado.clear()
        self._cmb_estado.addItem("Todos os Estados", "todos")
        for uf in info.get("estados", []):
            self._cmb_estado.addItem(f"Estado: {uf}", uf)
        self._cmb_estado.blockSignals(False)

    def _escolher_pasta(self) -> None:
        pasta = QFileDialog.getExistingDirectory(self, "Selecionar pasta Dataset/v1 do DNIT")
        if pasta:
            definir_pasta_dataset(Path(pasta))
            self._recarregar_estados()
            self._atualizar_lista()

    def _atualizar_lista(self) -> None:
        info = resumo_dataset_dnit()
        if not info["disponivel"]:
            self._lbl_status.setText("Pasta dataset/Dataset/v1 não localizada")
            self._tabela.setRowCount(0)
            self._btn_abrir.setEnabled(False)
            return

        itens = filtrar_dataset_dnit(
            busca=self._txt_busca.text(),
            particao_f=self._cmb_part.currentData() or "todas",
            estado_f=self._cmb_estado.currentData() or "todos",
        )
        exibidos = itens[:400]
        self._lbl_status.setText(
            f"{len(itens)} encontradas (de {info['total']} no dataset)"
            + (f" · exibindo 400" if len(itens) > 400 else "")
        )

        self._tabela.setRowCount(len(exibidos))
        for row, it in enumerate(exibidos):
            c0 = QTableWidgetItem(it["rodovia"])
            c1 = QTableWidgetItem(it["particao"].upper())
            c2 = QTableWidgetItem(it["nome"])
            c0.setData(Qt.UserRole, it["caminho"])
            self._tabela.setItem(row, 0, c0)
            self._tabela.setItem(row, 1, c1)
            self._tabela.setItem(row, 2, c2)

        if exibidos:
            self._tabela.selectRow(0)
        else:
            self._btn_abrir.setEnabled(False)
            self._lbl_img.setPixmap(QPixmap())
            self._lbl_img.setText("Nenhuma amostra para o filtro atual")

    def _caminho_selecionado(self) -> Optional[str]:
        sel = self._tabela.selectedItems()
        if not sel:
            return None
        item0 = self._tabela.item(sel[0].row(), 0)
        return item0.data(Qt.UserRole) if item0 else None

    def _atualizar_preview(self) -> None:
        cam = self._caminho_selecionado()
        if not cam:
            self._btn_abrir.setEnabled(False)
            return
        self._btn_abrir.setEnabled(True)
        try:
            am = carregar_amostra_dnit(cam)
            vis = cv2.resize(am.imagem, (360, 225), interpolation=cv2.INTER_AREA)
            roi = cv2.resize(am.mascaras["LANE"].astype(np.uint8), (360, 225), interpolation=cv2.INTER_NEAREST) > 0
            gt_t = cv2.resize(am.mascaras["CRACK"].astype(np.uint8), (360, 225), interpolation=cv2.INTER_NEAREST) > 0
            gt_b = cv2.resize(am.mascaras["POTHOLE"].astype(np.uint8), (360, 225), interpolation=cv2.INTER_NEAREST) > 0

            vis[~roi] = (vis[~roi] // 3)
            vis[gt_t & roi] = (30, 175, 255)
            vis[gt_b & roi] = (70, 230, 80)
            cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (235, 195, 50), 1)

            qimg = bgr_para_qimage(vis)
            self._lbl_img.setPixmap(QPixmap.fromImage(qimg))
            px_t = int((am.mascaras["CRACK"] & am.mascaras["LANE"]).sum())
            px_b = int((am.mascaras["POTHOLE"] & am.mascaras["LANE"]).sum())
            self._lbl_info_prev.setText(
                f"{am.nome}\nGabarito na pista: {px_t:,} px trinca · {px_b:,} px buraco".replace(",", ".")
            )
        except Exception as exc:
            self._lbl_img.setPixmap(QPixmap())
            self._lbl_img.setText(f"Erro na prévia: {exc}")

    def _confirmar_atual(self, *_args) -> None:
        cam = self._caminho_selecionado()
        if cam:
            self.amostra_escolhida = cam
            self.accept()
