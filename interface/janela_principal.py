"""Janela principal em PyQt5 do Sistema de Detecção e Quantificação de Danos em Pavimentos."""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .conteudo import (
    ETAPAS_PIPELINE,
    GRUPOS_PARAMETROS,
    LEGENDAS_SUBETAPA,
    LIMITACOES_RESUMO,
    METRICAS_OFICIAIS_TESTE,
    MOSAICO_QUADROS,
    ORDEM_ETAPAS,
    PONTOS_FORTES_RESUMO,
    REQUISITOS_DISCIPLINA,
    SEVERIDADE_ROTULO,
    TIPO_TRINCA_ROTULO,
)
from .dados import (
    EntradaAtiva,
    carregar_entrada_arquivo,
    carregar_entrada_cenario,
    carregar_entrada_dnit,
    carregar_presets_params,
    construir_parametros,
    filtrar_dataset_dnit,
    listar_cenarios,
    montar_relatorio_json,
    resolver_roi,
    resumo_dataset_dnit,
)
from .dialogo_dnit import DnitExplorerDialog
from .pipeline_didatico import (
    executar_pipeline_didatico,
    gerar_painel_oficial_png,
    inspecionar_pixel,
)
from .tema import CORES, escolher_familia_mono, escolher_familia_ui, gerar_folha_estilo
from .widgets import (
    ImageCanvasWidget,
    MosaicTileWidget,
    SeverityGaugeWidget,
    TextureHistogramWidget,
)


class JanelaPrincipal(QMainWindow):
    """Interface gráfica desktop para inspeção etapa por etapa do pipeline clássico."""

    def __init__(self, caminho_inicial: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Detecção e Quantificação de Danos em Pavimentos — PI 2026.2 · UFRPE")
        self.resize(1380, 860)
        self.setMinimumSize(1100, 700)

        self._familia_ui = escolher_familia_ui()
        self._familia_mono = escolher_familia_mono()
        self.setStyleSheet(gerar_folha_estilo(self._familia_ui, self._familia_mono))

        # Estado da aplicação
        self._presets = carregar_presets_params()
        self._params_dict: Dict[str, int] = dict(self._presets["final"])
        self._entrada: EntradaAtiva = carregar_entrada_cenario("rodovia_dnit_01")
        self._roi_modo: str = "cenario"  # "cenario" | "inteira" | "poligono"
        self._poligono_roi: List[Tuple[float, float]] = list(self._entrada.poligono_inicial)
        self._reduzir_2x: bool = self._entrada.reduzir_padrao

        self._etapa_atual: str = "5"
        self._subetapa_atual: str = "5_final"
        self._resultado: Optional[dict] = None

        self._sliders_params: Dict[str, Tuple[QSlider, QLabel, int, str]] = {}
        self._bloqueando_sliders: bool = False

        self._timer_analise = QTimer(self)
        self._timer_analise.setSingleShot(True)
        self._timer_analise.setInterval(35)
        self._timer_analise.timeout.connect(self._executar_analise)

        # Construção da UI
        central = QWidget(self)
        self.setCentralWidget(central)
        lay_raiz = QVBoxLayout(central)
        lay_raiz.setContentsMargins(0, 0, 0, 0)
        lay_raiz.setSpacing(0)

        lay_raiz.addWidget(self._construir_header())

        corpo = QWidget()
        lay_corpo = QVBoxLayout(corpo)
        lay_corpo.setContentsMargins(14, 10, 14, 12)
        lay_corpo.setSpacing(8)

        self._barra_entrada = self._construir_barra_entrada()
        lay_corpo.addWidget(self._barra_entrada)

        self._stack_views = QStackedWidget()
        self._stack_views.addWidget(self._construir_view_pipeline())     # idx 0
        self._stack_views.addWidget(self._construir_view_mosaico())      # idx 1
        self._stack_views.addWidget(self._construir_view_resultados())   # idx 2
        lay_corpo.addWidget(self._stack_views, 1)

        lay_raiz.addWidget(corpo, 1)

        self._configurar_atalhos()
        self._popular_combo_entradas()

        if caminho_inicial:
            self.carregar_caminho_externo(caminho_inicial)
        else:
            # Se o dataset DNIT estiver presente, seleciona o 1º destaque real ou mantém o cenário real
            self._aplicar_entrada(self._entrada)

    # ================================================================= #
    # CONSTRUÇÃO DOS BLOCOS VISUAIS                                     #
    # ================================================================= #
    def _construir_header(self) -> QFrame:
        barra = QFrame()
        barra.setObjectName("HeaderBar")
        barra.setFixedHeight(52)
        lay = QHBoxLayout(barra)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(12)

        badge = QLabel("PI 2026.2 · UFRPE")
        badge.setObjectName("BrandBadge")
        badge.setFixedHeight(24)
        badge.setAlignment(Qt.AlignCenter)
        titulo = QLabel("Detecção e Quantificação de Danos em Pavimentos")
        titulo.setObjectName("HeaderTitle")
        lay.addWidget(badge, 0, Qt.AlignVCenter)
        lay.addWidget(titulo, 0, Qt.AlignVCenter)
        lay.addStretch(1)

        self._grupo_nav = QButtonGroup(self)
        self._grupo_nav.setExclusive(True)
        abas = [
            ("1. Pipeline Passo a Passo", 0),
            ("2. Painel das Etapas (Mosaico)", 1),
            ("3. Resultados && Requisitos", 2),
        ]
        self._botoes_nav: List[QPushButton] = []
        for texto, idx in abas:
            btn = QPushButton(texto)
            btn.setCheckable(True)
            btn.setProperty("navTab", True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _c, i=idx: self.alternar_aba_principal(i))
            self._grupo_nav.addButton(btn, idx)
            self._botoes_nav.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)
        self._lbl_perf = QLabel("Núcleo: — ms · Só Inteiros")
        self._lbl_perf.setObjectName("PerfPill")
        lay.addWidget(self._lbl_perf)
        return barra

    def _construir_barra_entrada(self) -> QFrame:
        card = QFrame()
        card.setProperty("card", True)
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(8)

        lbl_in = QLabel("Amostra:")
        lbl_in.setStyleSheet("font-weight:700; font-size:12px; color:#334155;")
        self._cmb_entrada = QComboBox()
        self._cmb_entrada.setMinimumWidth(290)
        self._cmb_entrada.activated.connect(self._ao_escolher_combo_entrada)

        self._btn_explorar_dnit = QPushButton("Explorar DNIT…")
        self._btn_explorar_dnit.setProperty("variant", "primary")
        self._btn_explorar_dnit.clicked.connect(self._abrir_explorador_dnit)

        self._btn_sorteio_dnit = QPushButton("Sorteio Teste")
        self._btn_sorteio_dnit.setToolTip("Carrega uma amostra aleatória da partição inédita de teste do DNIT")
        self._btn_sorteio_dnit.clicked.connect(self._sortear_amostra_teste)

        btn_abrir = QPushButton("Abrir Foto…")
        btn_abrir.setToolTip("Abrir imagem avulsa (.jpg, .png, .bmp) ou *_RAW.jpg do DNIT")
        btn_abrir.clicked.connect(self._dialogo_abrir_foto)

        self._btn_abrir_roi = QPushButton("+ Máscara ROI")
        self._btn_abrir_roi.setToolTip("Anexar máscara binária de pista (*_LANE.png) à foto aberta")
        self._btn_abrir_roi.setVisible(False)
        self._btn_abrir_roi.clicked.connect(self._dialogo_abrir_roi)

        lay.addWidget(lbl_in)
        lay.addWidget(self._cmb_entrada)
        lay.addWidget(self._btn_explorar_dnit)
        lay.addWidget(self._btn_sorteio_dnit)
        lay.addWidget(btn_abrir)
        lay.addWidget(self._btn_abrir_roi)

        lay.addWidget(self._separador_vertical())

        lbl_roi = QLabel("ROI:")
        lbl_roi.setStyleSheet("font-weight:700; font-size:12px; color:#334155;")
        lay.addWidget(lbl_roi)

        self._grupo_roi = QButtonGroup(self)
        self._grupo_roi.setExclusive(True)
        self._botoes_roi: Dict[str, QPushButton] = {}
        for chave, rotulo in (("cenario", "Pista"), ("inteira", "Inteira"), ("poligono", "Editar")):
            b = QPushButton(rotulo)
            b.setCheckable(True)
            b.setProperty("pill", True)
            if chave == "cenario":
                b.setChecked(True)
            b.clicked.connect(lambda _c, k=chave: self.definir_modo_roi(k))
            self._grupo_roi.addButton(b)
            self._botoes_roi[chave] = b
            lay.addWidget(b)

        self._chk_reduzir = QCheckBox("Redução 2×")
        self._chk_reduzir.setChecked(True)
        self._chk_reduzir.toggled.connect(self._ao_alternar_reducao)
        lay.addWidget(self._chk_reduzir)

        lay.addStretch(1)

        btn_png = QPushButton("Salvar Painel PNG")
        btn_png.clicked.connect(self._dialogo_exportar_png)
        btn_json = QPushButton("Salvar JSON")
        btn_json.clicked.connect(self._dialogo_exportar_json)
        lay.addWidget(btn_png)
        lay.addWidget(btn_json)
        return card

    def _construir_view_pipeline(self) -> QWidget:
        pag = QWidget()
        lay = QVBoxLayout(pag)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # 1) Barra visual de etapas (Stepper)
        stepper = QFrame()
        stepper.setProperty("card", True)
        lay_step = QHBoxLayout(stepper)
        lay_step.setContentsMargins(10, 6, 10, 6)
        lay_step.setSpacing(6)

        self._grupo_etapas = QButtonGroup(self)
        self._grupo_etapas.setExclusive(True)
        self._botoes_etapa: Dict[str, QPushButton] = {}

        for idx, eid in enumerate(ORDEM_ETAPAS):
            info = ETAPAS_PIPELINE[eid]
            btn = QPushButton(f"{info['codigo']} · {info['nome_curto']}\n{info['sub_curto']}")
            btn.setCheckable(True)
            btn.setProperty("stepBtn", True)
            btn.setProperty("branch", info["cor_ramo"])
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda _c, k=eid: self.selecionar_etapa(k))
            self._grupo_etapas.addButton(btn)
            self._botoes_etapa[eid] = btn
            lay_step.addWidget(btn, 1)

            if idx < len(ORDEM_ETAPAS) - 1:
                sep_txt = "‖" if eid == "3a" else "→"
                lbl_s = QLabel(sep_txt)
                lbl_s.setAlignment(Qt.AlignCenter)
                lbl_s.setStyleSheet("color:#94A3B8; font-weight:800; font-size:14px; padding:0 2px;")
                lay_step.addWidget(lbl_s)

        lay.addWidget(stepper)

        # 2) Corpo dividido: Esquerda = Visualizador | Direita = Diagnóstico + Detalhes da Etapa
        corpo = QHBoxLayout()
        corpo.setSpacing(10)

        # ---------------- Coluna Esquerda (Visualizador) ---------------- #
        card_palco = QFrame()
        card_palco.setProperty("card", True)
        lay_palco = QVBoxLayout(card_palco)
        lay_palco.setContentsMargins(12, 10, 12, 10)
        lay_palco.setSpacing(7)

        # Cabeçalho da etapa
        cab_etapa = QHBoxLayout()
        cab_etapa.setSpacing(10)
        col_tit = QVBoxLayout()
        col_tit.setSpacing(1)
        self._lbl_eyebrow = QLabel("")
        self._lbl_eyebrow.setStyleSheet("font-size:10px; font-weight:800; color:#0F4C5C; letter-spacing:0.6px;")
        self._lbl_titulo_etapa = QLabel("")
        self._lbl_titulo_etapa.setStyleSheet("font-size:14px; font-weight:700; color:#0F172A;")
        col_tit.addWidget(self._lbl_eyebrow)
        col_tit.addWidget(self._lbl_titulo_etapa)
        cab_etapa.addLayout(col_tit, 1)

        btn_ant = QPushButton("← Anterior")
        btn_ant.clicked.connect(lambda: self._navegar_etapa(-1))
        btn_prox = QPushButton("Próxima →")
        btn_prox.setProperty("variant", "primary")
        btn_prox.clicked.connect(lambda: self._navegar_etapa(1))
        cab_etapa.addWidget(btn_ant)
        cab_etapa.addWidget(btn_prox)
        lay_palco.addLayout(cab_etapa)

        # Barra de sub-etapas + modos de visualização + zoom
        barra_sub = QHBoxLayout()
        barra_sub.setSpacing(5)
        self._wrap_subetapas = QWidget()
        self._lay_subetapas = QHBoxLayout(self._wrap_subetapas)
        self._lay_subetapas.setContentsMargins(0, 0, 0, 0)
        self._lay_subetapas.setSpacing(4)
        barra_sub.addWidget(self._wrap_subetapas)
        barra_sub.addStretch(1)

        self._grupo_cmp = QButtonGroup(self)
        self._grupo_cmp.setExclusive(True)
        self._botoes_cmp: Dict[str, QPushButton] = {}
        for modo, rot in (("cortina", "Antes × Depois"), ("unico", "Única"), ("lado", "Lado a Lado")):
            b = QPushButton(rot)
            b.setCheckable(True)
            b.setProperty("pill", True)
            if modo == "cortina":
                b.setChecked(True)
            b.clicked.connect(lambda _c, m=modo: self.definir_modo_comparacao(m))
            self._grupo_cmp.addButton(b)
            self._botoes_cmp[modo] = b
            barra_sub.addWidget(b)
        lay_palco.addLayout(barra_sub)

        # Faixa de edição de ROI (oculta por padrão)
        self._banner_roi = QFrame()
        self._banner_roi.setStyleSheet("background:#FEF3C7; border:1px solid #F59E0B; border-radius:6px;")
        self._banner_roi.setVisible(False)
        lay_broi = QHBoxLayout(self._banner_roi)
        lay_broi.setContentsMargins(10, 5, 10, 5)
        lbl_broi = QLabel("Arraste os vértices âmbar sobre a imagem para delimitar a pista (ROI).")
        lbl_broi.setStyleSheet("color:#92400E; font-size:12px; font-weight:600; border:none;")
        btn_aplicar_roi = QPushButton("Aplicar ROI")
        btn_aplicar_roi.setProperty("variant", "primary")
        btn_aplicar_roi.clicked.connect(self._aplicar_poligono_editado)
        btn_reset_roi = QPushButton("Resetar")
        btn_reset_roi.clicked.connect(self._resetar_poligono_roi)
        lay_broi.addWidget(lbl_broi, 1)
        lay_broi.addWidget(btn_aplicar_roi)
        lay_broi.addWidget(btn_reset_roi)
        lay_palco.addWidget(self._banner_roi)

        # Canvas interativo principal
        self._canvas = ImageCanvasWidget(familia_mono=self._familia_mono, parent=card_palco)
        self._canvas.zoomAlterado.connect(lambda z: self._lbl_zoom.setText(f"{int(round(z * 100))}%"))
        self._canvas.pixelInspecionado.connect(self._atualizar_barra_pixel)
        self._canvas.poligonoAlterado.connect(self._ao_arrastar_poligono)
        lay_palco.addWidget(self._canvas, 1)

        # Barra inferior do canvas: Legenda de cores (esquerda) + Zoom e Lupa (direita)
        barra_leg_zoom = QHBoxLayout()
        barra_leg_zoom.setContentsMargins(2, 0, 2, 0)
        barra_leg_zoom.setSpacing(6)

        self._wrap_legenda = QWidget()
        self._lay_legenda = QHBoxLayout(self._wrap_legenda)
        self._lay_legenda.setContentsMargins(0, 0, 0, 0)
        self._lay_legenda.setSpacing(12)
        barra_leg_zoom.addWidget(self._wrap_legenda, 1)

        btn_z_out = QPushButton("−")
        btn_z_out.setProperty("iconBtn", True)
        btn_z_out.clicked.connect(lambda: self._canvas.aplicar_zoom(self._canvas._zoom / 1.25))
        self._lbl_zoom = QLabel("100%")
        self._lbl_zoom.setFixedWidth(40)
        self._lbl_zoom.setAlignment(Qt.AlignCenter)
        self._lbl_zoom.setStyleSheet(f"font-family:'{self._familia_mono}'; font-size:11px; font-weight:700; color:#475569;")
        btn_z_in = QPushButton("+")
        btn_z_in.setProperty("iconBtn", True)
        btn_z_in.clicked.connect(lambda: self._canvas.aplicar_zoom(self._canvas._zoom * 1.25))
        btn_z_fit = QPushButton("Ajustar")
        btn_z_fit.clicked.connect(lambda: self._canvas.resetar_zoom())

        self._btn_lupa = QPushButton("Lupa 6×")
        self._btn_lupa.setCheckable(True)
        self._btn_lupa.setProperty("pill", True)
        self._btn_lupa.toggled.connect(self._canvas_definir_lupa)

        barra_leg_zoom.addWidget(btn_z_out)
        barra_leg_zoom.addWidget(self._lbl_zoom)
        barra_leg_zoom.addWidget(btn_z_in)
        barra_leg_zoom.addWidget(btn_z_fit)
        barra_leg_zoom.addWidget(self._btn_lupa)
        lay_palco.addLayout(barra_leg_zoom)

        # Barra compacta de inspeção de pixel em 1 linha
        barra_px = QFrame()
        barra_px.setProperty("subtleBox", True)
        lay_px = QHBoxLayout(barra_px)
        lay_px.setContentsMargins(10, 5, 10, 5)
        lay_px.setSpacing(10)
        self._lbl_pixel_info = QLabel(
            "Passe o cursor sobre a imagem para inspecionar os valores inteiros do pixel (clique para fixar)."
        )
        self._lbl_pixel_info.setStyleSheet(
            f"font-family:'{self._familia_mono}'; font-size:11px; color:#334155;"
        )
        lay_px.addWidget(self._lbl_pixel_info, 1)
        lay_palco.addWidget(barra_px)

        corpo.addWidget(card_palco, 7)

        # ---------------- Coluna Direita (Diagnóstico + Etapa) ---------------- #
        col_dir = QWidget()
        col_dir.setMinimumWidth(355)
        col_dir.setMaximumWidth(420)
        lay_dir = QVBoxLayout(col_dir)
        lay_dir.setContentsMargins(0, 0, 0, 0)
        lay_dir.setSpacing(8)

        # Card 1: Diagnóstico da Via
        card_diag = QFrame()
        card_diag.setProperty("card", True)
        lay_diag = QVBoxLayout(card_diag)
        lay_diag.setContentsMargins(12, 10, 12, 10)
        lay_diag.setSpacing(6)

        topo_diag = QHBoxLayout()
        lbl_d = QLabel("Diagnóstico da Via")
        lbl_d.setStyleSheet("font-size:13px; font-weight:700; color:#0F172A;")
        self._lbl_sev_badge = QLabel("—")
        self._lbl_sev_badge.setStyleSheet("padding:2px 10px; border-radius:10px; font-weight:800; font-size:11px;")
        topo_diag.addWidget(lbl_d)
        topo_diag.addStretch(1)
        topo_diag.addWidget(self._lbl_sev_badge)
        lay_diag.addLayout(topo_diag)

        self._gauge = SeverityGaugeWidget()
        lay_diag.addWidget(self._gauge)

        grade_kpi = QGridLayout()
        grade_kpi.setSpacing(6)
        self._lbl_kpi_indice = self._criar_caixa_kpi(grade_kpi, 0, 0, "Índice / 100")
        self._lbl_kpi_trincas = self._criar_caixa_kpi(grade_kpi, 0, 1, "Trincas")
        self._lbl_kpi_buracos = self._criar_caixa_kpi(grade_kpi, 0, 2, "Buracos")
        lay_diag.addLayout(grade_kpi)

        self._lbl_tipos_dnit = QLabel("")
        self._lbl_tipos_dnit.setStyleSheet("font-size:11px; color:#475569; font-weight:600;")
        lay_diag.addWidget(self._lbl_tipos_dnit)

        self._lbl_gt_strip = QLabel("")
        self._lbl_gt_strip.setWordWrap(True)
        self._lbl_gt_strip.setVisible(False)
        self._lbl_gt_strip.setStyleSheet(
            "background:#E6F3F6; color:#0F4C5C; border:1px solid #99D5E3; "
            "border-radius:6px; padding:5px 8px; font-size:11px; font-weight:700;"
        )
        lay_diag.addWidget(self._lbl_gt_strip)

        lay_dir.addWidget(card_diag)

        # Card 2: Abas laterais (Etapa Atual | Regiões | Parâmetros)
        self._abas_dir = QTabWidget()
        self._abas_dir.addTab(self._construir_aba_etapa(), "Etapa Atual")
        self._abas_dir.addTab(self._construir_aba_regioes(), "Regiões")
        self._abas_dir.addTab(self._construir_aba_parametros(), "Parâmetros")
        lay_dir.addWidget(self._abas_dir, 1)

        corpo.addWidget(col_dir, 3)
        lay.addLayout(corpo, 1)
        return pag

    def _construir_aba_etapa(self) -> QWidget:
        aba = QWidget()
        lay = QVBoxLayout(aba)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(7)

        self._lbl_topico_disc = QLabel("")
        self._lbl_topico_disc.setStyleSheet(
            "background:#F1F5F9; color:#334155; border-radius:5px; padding:3px 8px; "
            "font-size:11px; font-weight:700;"
        )
        lay.addWidget(self._lbl_topico_disc)

        self._lbl_formula = QLabel("")
        self._lbl_formula.setWordWrap(True)
        self._lbl_formula.setStyleSheet(
            f"background:#F8FAFC; color:#0F4C5C; border:1px solid #CBD5E1; border-radius:6px; "
            f"padding:5px 8px; font-family:'{self._familia_mono}'; font-size:11px; font-weight:700;"
        )
        lay.addWidget(self._lbl_formula)

        # 3 pontos objetivos (sem parágrafos longos)
        self._lbls_pontos: List[QLabel] = []
        for _ in range(3):
            lbl = QLabel("")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size:12px; color:#1E293B; padding-left:2px;")
            self._lbls_pontos.append(lbl)
            lay.addWidget(lbl)

        # Caixa de métricas pré-alocadas da etapa na imagem atual
        caixa_num = QFrame()
        caixa_num.setProperty("subtleBox", True)
        lay_metricas = QVBoxLayout(caixa_num)
        lay_metricas.setContentsMargins(10, 7, 10, 7)
        lay_metricas.setSpacing(4)
        self._linhas_metricas_etapa: List[Tuple[QWidget, QLabel, QLabel]] = []
        for _ in range(5):
            w_row = QWidget()
            r_lay = QHBoxLayout(w_row)
            r_lay.setContentsMargins(0, 0, 0, 0)
            r_lay.setSpacing(6)
            l_r = QLabel("")
            l_r.setStyleSheet("font-size:11px; color:#475569; border:none;")
            l_v = QLabel("")
            l_v.setStyleSheet(
                f"font-family:'{self._familia_mono}'; font-size:11px; font-weight:700; color:#0F172A; border:none;"
            )
            r_lay.addWidget(l_r)
            r_lay.addStretch(1)
            r_lay.addWidget(l_v)
            lay_metricas.addWidget(w_row)
            self._linhas_metricas_etapa.append((w_row, l_r, l_v))
        lay.addWidget(caixa_num)

        # Gráfico nativo de histograma (visível na Etapa 3B)
        self._hist_widget = TextureHistogramWidget()
        self._hist_widget.setVisible(False)
        lay.addWidget(self._hist_widget)

        # Botões rápidos de experimentação
        self._wrap_exp = QWidget()
        self._lay_exp = QHBoxLayout(self._wrap_exp)
        self._lay_exp.setContentsMargins(0, 0, 0, 0)
        self._lay_exp.setSpacing(6)
        lay.addWidget(self._wrap_exp)

        lay.addStretch(1)
        return aba

    def _construir_aba_regioes(self) -> QWidget:
        aba = QWidget()
        lay = QVBoxLayout(aba)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        topo = QHBoxLayout()
        topo.setSpacing(5)
        self._cmb_tipo_reg = QComboBox()
        self._cmb_tipo_reg.addItem("Trincas Aceitas", "trinca_ok")
        self._cmb_tipo_reg.addItem("Buracos Aceitos", "buraco_ok")
        self._cmb_tipo_reg.addItem("Trincas Rejeitadas", "trinca_rej")
        self._cmb_tipo_reg.addItem("Buracos Rejeitados", "buraco_rej")
        self._cmb_tipo_reg.currentIndexChanged.connect(self._popular_tabela_regioes)

        btn_limpar = QPushButton("Limpar Mira")
        btn_limpar.clicked.connect(lambda: self._canvas.destacar_bbox(None))

        topo.addWidget(self._cmb_tipo_reg, 1)
        topo.addWidget(btn_limpar)
        lay.addLayout(topo)

        lbl_dica = QLabel("Clique em uma linha para destacar a região na imagem.")
        lbl_dica.setStyleSheet("font-size:11px; color:#64748B;")
        lay.addWidget(lbl_dica)

        self._tab_regioes = QTableWidget(0, 4)
        self._tab_regioes.verticalHeader().setVisible(False)
        self._tab_regioes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tab_regioes.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tab_regioes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tab_regioes.setAlternatingRowColors(True)
        self._tab_regioes.itemSelectionChanged.connect(self._ao_selecionar_regiao)
        lay.addWidget(self._tab_regioes, 1)
        return aba

    def _construir_aba_parametros(self) -> QWidget:
        aba = QWidget()
        lay = QVBoxLayout(aba)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        barra_p = QHBoxLayout()
        barra_p.setSpacing(5)
        b_fin = QPushButton("Oficial (params.json)")
        b_fin.setProperty("variant", "primary")
        b_fin.clicked.connect(lambda: self.aplicar_preset_parametros("final"))
        b_ant = QPushButton("Anterior (r=4)")
        b_ant.clicked.connect(lambda: self.aplicar_preset_parametros("slides_anterior"))
        b_pad = QPushButton("Padrão")
        b_pad.clicked.connect(lambda: self.aplicar_preset_parametros("padrao"))
        barra_p.addWidget(b_fin)
        barra_p.addWidget(b_ant)
        barra_p.addWidget(b_pad)
        lay.addLayout(barra_p)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        miolo = QWidget()
        lay_m = QVBoxLayout(miolo)
        lay_m.setContentsMargins(4, 2, 6, 4)
        lay_m.setSpacing(8)

        for nome_grupo, lista in GRUPOS_PARAMETROS:
            lbl_g = QLabel(nome_grupo)
            lbl_g.setStyleSheet(
                "font-size:11px; font-weight:800; color:#0F4C5C; "
                "padding-top:4px; border-bottom:1px solid #E2E8F0;"
            )
            lay_m.addWidget(lbl_g)

            for chave, rotulo, vmin, vmax, passo, div, sufixo in lista:
                bloco = QVBoxLayout()
                bloco.setSpacing(2)
                linha_t = QHBoxLayout()
                lbl_n = QLabel(rotulo)
                lbl_n.setStyleSheet("font-size:11px; font-weight:600; color:#334155;")
                lbl_v = QLabel("")
                lbl_v.setStyleSheet(
                    f"font-family:'{self._familia_mono}'; font-size:11px; font-weight:700; color:#0F4C5C;"
                )
                linha_t.addWidget(lbl_n)
                linha_t.addStretch(1)
                linha_t.addWidget(lbl_v)

                sld = QSlider(Qt.Horizontal)
                sld.setMinimum(vmin)
                sld.setMaximum(vmax)
                sld.setSingleStep(passo)
                sld.setPageStep(passo * 2)
                val_ini = self._params_dict.get(chave, vmin)
                sld.setValue(max(vmin, min(vmax, val_ini)))
                lbl_v.setText(self._formatar_valor_param(val_ini, div, sufixo))

                sld.valueChanged.connect(lambda val, k=chave: self._ao_mudar_slider(k, val))
                self._sliders_params[chave] = (sld, lbl_v, div, sufixo)

                bloco.addLayout(linha_t)
                bloco.addWidget(sld)
                lay_m.addLayout(bloco)

        lay_m.addStretch(1)
        scroll.setWidget(miolo)
        lay.addWidget(scroll, 1)
        return aba

    def _construir_view_mosaico(self) -> QWidget:
        pag = QWidget()
        lay = QVBoxLayout(pag)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        cab = QFrame()
        cab.setProperty("card", True)
        lay_c = QHBoxLayout(cab)
        lay_c.setContentsMargins(14, 8, 14, 8)
        lbl = QLabel("Visão Completa do Pipeline — Clique em qualquer etapa para inspecionar em detalhe")
        lbl.setStyleSheet("font-size:13px; font-weight:700; color:#0F172A;")
        lay_c.addWidget(lbl, 1)
        lay.addWidget(cab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grade_wrap = QWidget()
        grade = QGridLayout(grade_wrap)
        grade.setContentsMargins(2, 2, 2, 2)
        grade.setSpacing(10)

        self._tiles_mosaico: Dict[str, MosaicTileWidget] = {}
        for idx, (eid, chave_img, cod, tit) in enumerate(MOSAICO_QUADROS):
            tile = MosaicTileWidget(eid, chave_img, cod, tit)
            tile.clicado.connect(self._abrir_do_mosaico)
            self._tiles_mosaico[chave_img] = tile
            grade.addWidget(tile, idx // 4, idx % 4)

        scroll.setWidget(grade_wrap)
        lay.addWidget(scroll, 1)
        return pag

    def _construir_view_resultados(self) -> QWidget:
        pag = QWidget()
        lay = QVBoxLayout(pag)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # 1) 4 KPIs principais da avaliação oficial
        linha_kpi = QHBoxLayout()
        linha_kpi.setSpacing(10)
        kpis = [
            ("1.451", "Amostras Reais DNIT", "785 treino · 666 teste inédito (MD5)", "#0F4C5C"),
            ("0,928", "F1 Trincas (Imagem)", "F1 tol. 2 px: 0,531 (+66% sobre inicial)", "#0F4C5C"),
            ("57,4%", "Severidade Exata", "93,2% até 1 nível (4 classes · acaso 25%)", "#B45309"),
            ("~45 ms", "Tempo por Imagem", "100% aritmética inteira · NumPy do zero", "#161B22"),
        ]
        for num, tit, sub, cor in kpis:
            c = QFrame()
            c.setProperty("card", True)
            lc = QVBoxLayout(c)
            lc.setContentsMargins(14, 10, 14, 10)
            lc.setSpacing(2)
            l_num = QLabel(num)
            l_num.setStyleSheet(f"font-size:22px; font-weight:800; color:{cor};")
            l_tit = QLabel(tit)
            l_tit.setStyleSheet("font-size:12px; font-weight:700; color:#0F172A;")
            l_sub = QLabel(sub)
            l_sub.setStyleSheet("font-size:11px; color:#64748B;")
            lc.addWidget(l_num)
            lc.addWidget(l_tit)
            lc.addWidget(l_sub)
            linha_kpi.addWidget(c, 1)
        lay.addLayout(linha_kpi)

        # 2) Duas tabelas lado a lado: Métricas (666 Testes) | Requisitos da Disciplina
        meio = QHBoxLayout()
        meio.setSpacing(10)

        card_met = QFrame()
        card_met.setProperty("card", True)
        lm = QVBoxLayout(card_met)
        lm.setContentsMargins(12, 10, 12, 10)
        lm.setSpacing(6)
        lbl_mt = QLabel("Resultados Quantitativos — 666 Imagens Inéditas de Teste")
        lbl_mt.setStyleSheet("font-size:13px; font-weight:700; color:#0F172A;")
        lm.addWidget(lbl_mt)

        tab_m = QTableWidget(len(METRICAS_OFICIAIS_TESTE), 5)
        tab_m.setHorizontalHeaderLabels(["Métrica", "Inicial", "r=4", "Final (r=10)", "Destaque"])
        tab_m.verticalHeader().setVisible(False)
        tab_m.verticalHeader().setDefaultSectionSize(36)
        tab_m.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tab_m.setSelectionMode(QAbstractItemView.NoSelection)
        tab_m.setAlternatingRowColors(True)
        for r, (met, ini, ant, fin, obs) in enumerate(METRICAS_OFICIAIS_TESTE):
            tab_m.setItem(r, 0, QTableWidgetItem(met))
            tab_m.setItem(r, 1, QTableWidgetItem(ini))
            tab_m.setItem(r, 2, QTableWidgetItem(ant))
            it_f = QTableWidgetItem(fin)
            f_bold = QFont(tab_m.font())
            f_bold.setBold(True)
            it_f.setFont(f_bold)
            it_f.setForeground(Qt.darkCyan)
            tab_m.setItem(r, 3, it_f)
            tab_m.setItem(r, 4, QTableWidgetItem(obs))
        hm = tab_m.horizontalHeader()
        hm.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            hm.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        lm.addWidget(tab_m, 1)
        meio.addWidget(card_met, 6)

        card_req = QFrame()
        card_req.setProperty("card", True)
        lr = QVBoxLayout(card_req)
        lr.setContentsMargins(12, 10, 12, 10)
        lr.setSpacing(6)
        lbl_rt = QLabel("Tópicos da Disciplina Atendidos (Descricao de projeto.pdf)")
        lbl_rt.setStyleSheet("font-size:13px; font-weight:700; color:#0F172A;")
        lr.addWidget(lbl_rt)

        tab_r = QTableWidget(len(REQUISITOS_DISCIPLINA), 4)
        tab_r.setHorizontalHeaderLabels(["Tópico", "Implementação", "Módulo", "Ver"])
        tab_r.verticalHeader().setVisible(False)
        tab_r.setWordWrap(True)
        tab_r.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tab_r.setSelectionMode(QAbstractItemView.NoSelection)
        tab_r.setAlternatingRowColors(True)
        for r, (_item, topico, impl, mod, eid, sub) in enumerate(REQUISITOS_DISCIPLINA):
            tab_r.setItem(r, 0, QTableWidgetItem(topico))
            tab_r.setItem(r, 1, QTableWidgetItem(impl))
            tab_r.setItem(r, 2, QTableWidgetItem(mod))
            cell_w = QWidget()
            cell_l = QHBoxLayout(cell_w)
            cell_l.setContentsMargins(4, 4, 4, 4)
            btn_ir = QPushButton(f"Etapa {ETAPAS_PIPELINE[eid]['codigo']} →")
            btn_ir.setProperty("variant", "primary")
            btn_ir.clicked.connect(lambda _c, e=eid, s=sub: self._ir_para_etapa_sub(e, s))
            cell_l.addWidget(btn_ir)
            tab_r.setCellWidget(r, 3, cell_w)
        hr = tab_r.horizontalHeader()
        hr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hr.setSectionResizeMode(1, QHeaderView.Stretch)
        hr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        tab_r.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lr.addWidget(tab_r, 1)
        meio.addWidget(card_req, 6)

        lay.addLayout(meio, 1)

        # 3) Rodapé conciso de Análise Crítica
        rodape = QHBoxLayout()
        rodape.setSpacing(10)
        rodape.addWidget(self._card_lista_curta("Pontos Fortes Validados", PONTOS_FORTES_RESUMO, "#0F4C5C"), 1)
        rodape.addWidget(
            self._card_lista_curta("Limitações do Método Local 2D & Evolução", LIMITACOES_RESUMO, "#B45309"), 1
        )
        lay.addLayout(rodape)
        return pag

    # ================================================================= #
    # AUXILIARES DE LAYOUT                                              #
    # ================================================================= #
    def _separador_vertical(self) -> QFrame:
        lin = QFrame()
        lin.setFrameShape(QFrame.VLine)
        lin.setStyleSheet("color:#CBD5E1;")
        return lin

    def _criar_caixa_kpi(self, grade: QGridLayout, lin: int, col: int, rotulo: str) -> QLabel:
        box = QFrame()
        box.setProperty("subtleBox", True)
        lb = QVBoxLayout(box)
        lb.setContentsMargins(6, 5, 6, 5)
        lb.setSpacing(1)
        val = QLabel("—")
        val.setAlignment(Qt.AlignCenter)
        val.setStyleSheet("font-size:13px; font-weight:800; color:#0F172A;")
        sub = QLabel(rotulo)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("font-size:10px; color:#64748B; font-weight:600;")
        lb.addWidget(val)
        lb.addWidget(sub)
        grade.addWidget(box, lin, col)
        return val

    def _card_lista_curta(self, titulo: str, itens: List[str], cor_acento: str) -> QFrame:
        c = QFrame()
        c.setProperty("card", True)
        lay = QVBoxLayout(c)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(4)
        lbl_t = QLabel(titulo)
        lbl_t.setStyleSheet(f"font-size:12px; font-weight:800; color:{cor_acento};")
        lay.addWidget(lbl_t)
        for txt in itens:
            l_it = QLabel(f"•  {txt}")
            l_it.setWordWrap(True)
            l_it.setStyleSheet("font-size:11px; color:#334155;")
            lay.addWidget(l_it)
        return c

    def _configurar_atalhos(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._navegar_etapa(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._navegar_etapa(1))
        for i, eid in enumerate(ORDEM_ETAPAS, start=1):
            QShortcut(QKeySequence(str(i)), self, activated=lambda k=eid: self.selecionar_etapa(k))

    # ================================================================= #
    # GESTÃO DE ENTRADAS E PARÂMETROS                                   #
    # ================================================================= #
    def _popular_combo_entradas(self) -> None:
        self._cmb_entrada.blockSignals(True)
        self._cmb_entrada.clear()

        info_ds = resumo_dataset_dnit()
        self._btn_sorteio_dnit.setEnabled(info_ds["disponivel"])
        for dest in info_ds.get("destaques", []):
            self._cmb_entrada.addItem(dest["titulo"], ("dnit", dest["nome"]))

        for cen in listar_cenarios():
            prefixo = "Real · " if cen["categoria"] == "real" else ""
            self._cmb_entrada.addItem(f"{prefixo}{cen['titulo']}", ("cenario", cen["id"]))

        self._cmb_entrada.blockSignals(False)

    def _ao_escolher_combo_entrada(self, idx: int) -> None:
        dado = self._cmb_entrada.itemData(idx)
        if not dado:
            return
        tipo, ident = dado[0], dado[1]
        try:
            if tipo == "dnit":
                self._aplicar_entrada(carregar_entrada_dnit(ident))
            elif tipo == "cenario":
                self._aplicar_entrada(carregar_entrada_cenario(ident))
            elif tipo == "arquivo":
                self._aplicar_entrada(carregar_entrada_arquivo(ident))
        except Exception as exc:
            QMessageBox.warning(self, "Erro ao carregar amostra", str(exc))

    def _aplicar_entrada(self, entrada: EntradaAtiva) -> None:
        self._entrada = entrada
        self._sincronizar_combo_com_entrada(entrada)
        self._poligono_roi = list(entrada.poligono_inicial)
        self._roi_modo = "cenario" if entrada.tem_roi_propria else "inteira"
        for k, b in self._botoes_roi.items():
            b.setChecked(k == self._roi_modo)
        self._banner_roi.setVisible(False)
        self._canvas.definir_edicao_roi(False, self._poligono_roi)

        self._chk_reduzir.blockSignals(True)
        self._chk_reduzir.setChecked(entrada.reduzir_padrao)
        self._reduzir_2x = entrada.reduzir_padrao
        self._chk_reduzir.blockSignals(False)

        self._btn_abrir_roi.setVisible(entrada.tipo == "arquivo")
        self._executar_analise()

    def _sincronizar_combo_com_entrada(self, ent: EntradaAtiva) -> None:
        chave_alvo = (ent.tipo, ent.identificador)
        self._cmb_entrada.blockSignals(True)
        encontrou = False
        for i in range(self._cmb_entrada.count()):
            dado = self._cmb_entrada.itemData(i)
            if dado == chave_alvo or (isinstance(dado, tuple) and len(dado) == 3 and dado[:2] == chave_alvo):
                self._cmb_entrada.setCurrentIndex(i)
                encontrou = True
                break
        if not encontrou:
            dado0 = self._cmb_entrada.itemData(0) if self._cmb_entrada.count() > 0 else None
            marcador = (ent.tipo, ent.identificador, "custom")
            if isinstance(dado0, tuple) and len(dado0) == 3 and dado0[2] == "custom":
                self._cmb_entrada.setItemText(0, ent.titulo)
                self._cmb_entrada.setItemData(0, marcador)
            else:
                self._cmb_entrada.insertItem(0, ent.titulo, marcador)
            self._cmb_entrada.setCurrentIndex(0)
        self._cmb_entrada.blockSignals(False)

    def carregar_caminho_externo(self, caminho: str) -> None:
        p = Path(caminho).expanduser().resolve()
        if p.is_dir():
            ent = carregar_entrada_dnit(str(p))
        else:
            ent = carregar_entrada_arquivo(str(p))
        self._aplicar_entrada(ent)

    def _abrir_explorador_dnit(self) -> None:
        dlg = DnitExplorerDialog(self)
        if dlg.exec_() == DnitExplorerDialog.Accepted and dlg.amostra_escolhida:
            try:
                ent = carregar_entrada_dnit(dlg.amostra_escolhida)
                self._popular_combo_entradas()
                self._aplicar_entrada(ent)
            except Exception as exc:
                QMessageBox.warning(self, "Erro ao abrir amostra DNIT", str(exc))

    def _sortear_amostra_teste(self) -> None:
        candidatas = filtrar_dataset_dnit(particao_f="teste")
        if not candidatas:
            QMessageBox.information(self, "Dataset DNIT", "Nenhuma amostra de teste localizada.")
            return
        escolhida = random.choice(candidatas)
        ent = carregar_entrada_dnit(escolhida["caminho"])
        self._aplicar_entrada(ent)

    def _dialogo_abrir_foto(self) -> None:
        arq, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir foto de pavimento",
            "",
            "Imagens (*.jpg *.jpeg *.png *.bmp *.webp);;Todos os arquivos (*)",
        )
        if not arq:
            return
        try:
            ent = carregar_entrada_arquivo(arq)
            self._aplicar_entrada(ent)
        except Exception as exc:
            QMessageBox.warning(self, "Erro ao abrir imagem", str(exc))

    def _dialogo_abrir_roi(self) -> None:
        if self._entrada.tipo != "arquivo":
            return
        arq, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar máscara binária da pista (ROI)",
            "",
            "Imagens (*.png *.jpg *.bmp);;Todos os arquivos (*)",
        )
        if not arq:
            return
        try:
            ent = carregar_entrada_arquivo(self._entrada.identificador, caminho_roi=arq)
            self._aplicar_entrada(ent)
        except Exception as exc:
            QMessageBox.warning(self, "Erro na máscara ROI", str(exc))

    def definir_modo_roi(self, modo: str) -> None:
        self._roi_modo = modo
        for k, b in self._botoes_roi.items():
            b.setChecked(k == modo)
        editando = modo == "poligono"
        self._banner_roi.setVisible(editando)
        self._canvas.definir_edicao_roi(editando, self._poligono_roi)
        if not editando:
            self._executar_analise()

    def _ao_arrastar_poligono(self, novo_poly: List[Tuple[float, float]]) -> None:
        self._poligono_roi = novo_poly

    def _aplicar_poligono_editado(self) -> None:
        self._poligono_roi = self._canvas.obter_poligono_roi()
        self._banner_roi.setVisible(False)
        self._canvas.definir_edicao_roi(False)
        self._executar_analise()

    def _resetar_poligono_roi(self) -> None:
        self._poligono_roi = list(self._entrada.poligono_inicial)
        self._canvas.definir_edicao_roi(True, self._poligono_roi)
        self._executar_analise()

    def _ao_alternar_reducao(self, ativo: bool) -> None:
        self._reduzir_2x = ativo
        self._executar_analise()

    def _canvas_definir_lupa(self, ativa: bool) -> None:
        self._canvas.definir_lupa(ativa)

    def aplicar_preset_parametros(self, nome_preset: str) -> None:
        if nome_preset not in self._presets:
            return
        self._params_dict = dict(self._presets[nome_preset])
        self._sincronizar_sliders_com_params()
        self._executar_analise()

    def _sincronizar_sliders_com_params(self) -> None:
        self._bloqueando_sliders = True
        for chave, (sld, lbl_v, div, suf) in self._sliders_params.items():
            val = self._params_dict.get(chave, sld.minimum())
            sld.setValue(max(sld.minimum(), min(sld.maximum(), val)))
            lbl_v.setText(self._formatar_valor_param(val, div, suf))
        self._bloqueando_sliders = False

    def _ao_mudar_slider(self, chave: str, valor: int) -> None:
        if self._bloqueando_sliders:
            return
        self._params_dict[chave] = int(valor)
        if chave in self._sliders_params:
            _sld, lbl_v, div, suf = self._sliders_params[chave]
            lbl_v.setText(self._formatar_valor_param(valor, div, suf))
        self._timer_analise.start()

    @staticmethod
    def _formatar_valor_param(valor: int, div: int, sufixo: str) -> str:
        if div == 1:
            return f"{valor}{sufixo}"
        if div in (4, 10, 16):
            return f"{valor / div:.2f}".rstrip("0").rstrip(".") + sufixo
        return f"{valor / div:.3f}".rstrip("0").rstrip(".") + sufixo

    # ================================================================= #
    # EXECUÇÃO DO NÚCLEO E ATUALIZAÇÃO DA TELA                          #
    # ================================================================= #
    def _executar_analise(self) -> None:
        roi_full = resolver_roi(self._entrada, self._roi_modo, self._poligono_roi)
        params = construir_parametros(self._params_dict)
        self._resultado = executar_pipeline_didatico(
            self._entrada.img_bgr,
            roi_full,
            params,
            aplicar_reducao_2x=self._reduzir_2x,
            gt_trinca_full=self._entrada.gt_trinca,
            gt_buraco_full=self._entrada.gt_buraco,
        )

        dim = self._resultado["dimensoes"]
        w, h = dim["processamento"]
        self._chk_reduzir.setText(f"Redução 2× ({w}×{h})")
        self._lbl_perf.setText(
            f"Núcleo: {self._resultado['tempo_nucleo_ms']:.1f} ms · Só Inteiros".replace(".", ",")
        )

        self._atualizar_card_diagnostico()
        self._atualizar_mosaico()
        self._popular_tabela_regioes()
        self._renderizar_etapa_ativa()

    def alternar_aba_principal(self, idx: int) -> None:
        self._stack_views.setCurrentIndex(idx)
        self._barra_entrada.setVisible(idx in (0, 1))
        for i, b in enumerate(self._botoes_nav):
            b.setChecked(i == idx)

    def selecionar_etapa(self, etapa_id: str, subetapa: Optional[str] = None) -> None:
        if etapa_id not in ETAPAS_PIPELINE:
            return
        self._etapa_atual = etapa_id
        info = ETAPAS_PIPELINE[etapa_id]
        opcoes = self._subetapas_disponiveis(etapa_id)
        chaves_disp = { item[0] for item in opcoes }
        if subetapa and subetapa in chaves_disp:
            self._subetapa_atual = subetapa
        elif self._subetapa_atual not in chaves_disp:
            self._subetapa_atual = info["padrao_sub"]

        for k, b in self._botoes_etapa.items():
            b.setChecked(k == etapa_id)

        self._canvas.destacar_bbox(None)
        self._renderizar_etapa_ativa()

    def _navegar_etapa(self, delta: int) -> None:
        if self._stack_views.currentIndex() != 0:
            self.alternar_aba_principal(0)
        pos = ORDEM_ETAPAS.index(self._etapa_atual)
        novo = (pos + delta) % len(ORDEM_ETAPAS)
        self.selecionar_etapa(ORDEM_ETAPAS[novo])

    def _ir_para_etapa_sub(self, etapa_id: str, subetapa: str) -> None:
        self.alternar_aba_principal(0)
        self.selecionar_etapa(etapa_id, subetapa)

    def _abrir_do_mosaico(self, etapa_id: str, chave_img: str) -> None:
        self.alternar_aba_principal(0)
        self.selecionar_etapa(etapa_id, chave_img)

    def definir_modo_comparacao(self, modo: str) -> None:
        self._canvas.definir_modo_comparacao(modo)
        for k, b in self._botoes_cmp.items():
            b.setChecked(k == modo)

    def _subetapas_disponiveis(self, etapa_id: str) -> List[Tuple[str, str, str, str, str]]:
        info = ETAPAS_PIPELINE[etapa_id]
        lista = list(info["subetapas"])
        if etapa_id == "5" and self._resultado and self._resultado["metricas_gt"] is not None:
            lista.extend(info.get("subetapas_gt", []))
        return lista

    def _renderizar_etapa_ativa(self) -> None:
        if self._resultado is None:
            return
        eid = self._etapa_atual
        info = ETAPAS_PIPELINE[eid]
        for k, b in self._botoes_etapa.items():
            b.setChecked(k == eid)

        cor_eyebrow = "#B45309" if info["cor_ramo"] == "amber" else "#0F4C5C"
        self._lbl_eyebrow.setStyleSheet(
            f"font-size:10px; font-weight:800; color:{cor_eyebrow}; letter-spacing:0.6px;"
        )
        self._lbl_eyebrow.setText(info["eyebrow"])
        self._lbl_titulo_etapa.setText(info["titulo"])
        self._lbl_formula.setText(info["formula"])
        self._lbl_topico_disc.setText(info["topico"])

        for lbl, pt in zip(self._lbls_pontos, info["pontos"]):
            lbl.setText(f"•  {pt}")

        # Reconstrói pílulas de sub-etapas
        self._limpar_layout(self._lay_subetapas)
        opcoes = self._subetapas_disponiveis(eid)
        chaves = [o[0] for o in opcoes]
        if self._subetapa_atual not in chaves:
            self._subetapa_atual = info["padrao_sub"]

        sub_meta = opcoes[0]
        for item in opcoes:
            chave_img, rot_btn, _antes_k, _antes_l, _depois_l = item
            b = QPushButton(rot_btn)
            b.setCheckable(True)
            b.setProperty("pill", True)
            ativo = chave_img == self._subetapa_atual
            b.setChecked(ativo)
            if ativo:
                sub_meta = item
            b.clicked.connect(lambda _c, k=chave_img: self._escolher_subetapa(k))
            self._lay_subetapas.addWidget(b)

        # Atualiza imagens no canvas
        chave_depois, _rot_btn, chave_antes, rot_antes, rot_depois = sub_meta
        imgs = self._resultado["imagens_bgr"]
        bgr_depois = imgs[chave_depois]
        bgr_antes = imgs.get(chave_antes, imgs["1_original"])
        # Caso especial 5_gt_rotulo: compara Rótulo Humano (antes) vs Nossa Predição (depois)
        if chave_depois == "5_gt_rotulo" and "5_gt_rotulo" in imgs:
            bgr_antes = imgs["5_gt_rotulo"]
            bgr_depois = imgs["5_final"]
        self._canvas.definir_imagens(bgr_antes, bgr_depois, rot_antes, rot_depois)

        # Legenda de cores da sub-etapa
        self._limpar_layout(self._lay_legenda)
        for cor_hex, txt_leg in LEGENDAS_SUBETAPA.get(chave_depois, []):
            item_w = QWidget()
            li = QHBoxLayout(item_w)
            li.setContentsMargins(0, 0, 0, 0)
            li.setSpacing(5)
            bolinha = QLabel()
            bolinha.setFixedSize(10, 10)
            bolinha.setStyleSheet(
                f"background:{cor_hex}; border:1px solid #475569; border-radius:5px;"
            )
            lbl_l = QLabel(txt_leg)
            lbl_l.setStyleSheet("font-size:11px; color:#334155; font-weight:600;")
            li.addWidget(bolinha)
            li.addWidget(lbl_l)
            self._lay_legenda.addWidget(item_w)
        self._lay_legenda.addStretch(1)

        # Métricas numéricas da etapa atual
        self._atualizar_metricas_da_etapa(eid)

        # Histograma visível somente na Etapa 3B
        if eid == "3b":
            est = self._resultado["estatisticas"]
            self._hist_widget.atualizar(
                self._resultado["histogramas"]["textura"],
                est["mediana_textura"],
                est["limiar_textura"],
            )
            self._hist_widget.setVisible(True)
        else:
            self._hist_widget.setVisible(False)

        # Botões rápidos de teste da etapa
        self._limpar_layout(self._lay_exp)
        for rot_exp, tipo_exp, payload in info.get("experimentos", []):
            be = QPushButton(rot_exp)
            be.setProperty("variant", "amber")
            be.clicked.connect(lambda _c, t=tipo_exp, pl=payload: self._executar_experimento(t, pl))
            self._lay_exp.addWidget(be)
        if self._params_dict != self._presets["final"] or self._roi_modo != "cenario":
            b_rest = QPushButton("Restaurar Oficial")
            b_rest.clicked.connect(self._restaurar_config_oficial)
            self._lay_exp.addWidget(b_rest)
        self._lay_exp.addStretch(1)

    def _escolher_subetapa(self, chave_img: str) -> None:
        self._subetapa_atual = chave_img
        self._renderizar_etapa_ativa()

    def _executar_experimento(self, tipo: str, payload) -> None:
        if tipo == "roi":
            self.definir_modo_roi(str(payload))
        elif tipo == "reducao":
            self._chk_reduzir.setChecked(bool(payload))
        elif tipo == "otsu_lado":
            self._subetapa_atual = "3a_otsu"
            self.definir_modo_comparacao("lado")
            self._renderizar_etapa_ativa()
        elif tipo == "params":
            self._params_dict.update(payload)
            self._sincronizar_sliders_com_params()
            self._executar_analise()
        elif tipo == "params_sub":
            novos, sub = payload
            self._params_dict.update(novos)
            self._subetapa_atual = sub
            self._sincronizar_sliders_com_params()
            self._executar_analise()

    def _restaurar_config_oficial(self) -> None:
        self._params_dict = dict(self._presets["final"])
        self._sincronizar_sliders_com_params()
        self._roi_modo = "cenario" if self._entrada.tem_roi_propria else "inteira"
        for k, b in self._botoes_roi.items():
            b.setChecked(k == self._roi_modo)
        self._chk_reduzir.blockSignals(True)
        self._chk_reduzir.setChecked(self._entrada.reduzir_padrao)
        self._reduzir_2x = self._entrada.reduzir_padrao
        self._chk_reduzir.blockSignals(False)
        self._executar_analise()

    def _atualizar_metricas_da_etapa(self, eid: str) -> None:
        if not self._resultado:
            return
        dim = self._resultado["dimensoes"]
        est = self._resultado["estatisticas"]
        res = self._resultado["resumo_didatico"]
        q = self._resultado["quant"]

        def fmt_int(n: int) -> str:
            return f"{n:,}".replace(",", ".")

        if eid == "1":
            linhas = [
                ("Resolução original", f"{dim['original'][0]} × {dim['original'][1]}"),
                ("Resolução processada", f"{dim['processamento'][0]} × {dim['processamento'][1]} ({res['etapa1']['fator_pixels']}× menos px)"),
                ("Pixels na pista (ROI)", f"{fmt_int(dim['pixels_roi'])} ({dim['roi_pct']}%)"),
                ("Ignorados fora da ROI", f"{fmt_int(res['etapa1']['pixels_fora_roi'])} ({res['etapa1']['pct_fora_roi']}%)"),
            ]
        elif eid == "2":
            linhas = [
                ("Janela trincas (2r+1)", f"{est['janela_trinca']} × {est['janela_trinca']} px"),
                ("Janela buracos (2r_b+1)", f"{est['janela_buraco']} × {est['janela_buraco']} px"),
                ("Brilho médio na ROI", f"{est['media_roi']} (mediana {est['mediana_g']})"),
                ("Desvio s_xy (mediana / P95)", f"{res['etapa2']['s_xy_mediana_roi']} / {res['etapa2']['s_xy_p95_roi']}"),
            ]
        elif eid == "3a":
            linhas = [
                ("Sementes fortes", f"{fmt_int(est['px_trinca_forte'])} px"),
                ("Candidatas pós-fecho", f"{fmt_int(est['px_trinca_fraca_pos_fecho'])} px"),
                ("Mantidas pela histerese", f"{fmt_int(est['px_trinca_pos_histerese'])} px"),
                ("Ruído fraco descartado", f"{fmt_int(est['px_trinca_descartados_histerese'])} px ({res['etapa3a']['pct_ruido_removido']}%)"),
                ("Comparação Otsu global", f"{fmt_int(est['px_otsu'])} px marcados"),
            ]
        elif eid == "3b":
            linhas = [
                ("Mediana da textura s_b", f"{est['mediana_textura']}"),
                ("Desvio robusto (MAD)", f"{est['mad_textura']}"),
                ("Limiar t_tex calculado", f"{est['limiar_textura']}"),
                ("Candidatos cand_b", f"{fmt_int(est['px_buraco_cand'])} px ({res['etapa3b']['pct_pista_cand']}% da ROI)"),
            ]
        elif eid == "4":
            linhas = [
                ("Apagados pela Abertura", f"{fmt_int(est['px_buraco_removido_abertura'])} px (-{res['etapa4']['pct_removido_abertura']}%)"),
                ("Mantidos pós-Abertura", f"{fmt_int(est['px_buraco_pos_abertura'])} px"),
                ("Preenchidos no miolo", f"+{fmt_int(est['px_buraco_adicionado_preench'])} px"),
                ("Lacunas religadas (trincas)", f"+{fmt_int(est['px_trinca_fecho_adicionados'])} px"),
            ]
        else:
            linhas = [
                ("Trincas aceitas / rejeitadas", f"{est['n_comp_trinca_aceitos']} aceitas · {est['n_comp_trinca_rejeitados']} rej."),
                ("Esqueleto / bifurcações", f"{fmt_int(est['px_esqueleto'])} px · {est['px_bifurcacoes']} bifurc."),
                ("Buracos aceitos / rejeitados", f"{est['n_comp_buraco_aceitos']} aceitos · {est['n_comp_buraco_rejeitados']} rej."),
                ("Índice final de dano", f"{q['indice_dano_1000'] / 10:.1f} / 100 ({res['etapa5']['severidade_label']})".replace(".", ",")),
            ]

        for idx, (w_row, l_r, l_v) in enumerate(self._linhas_metricas_etapa):
            if idx < len(linhas):
                rot, val = linhas[idx]
                l_r.setText(rot)
                l_v.setText(val)
                w_row.setVisible(True)
            else:
                w_row.setVisible(False)

    def _atualizar_card_diagnostico(self) -> None:
        if not self._resultado:
            return
        q = self._resultado["quant"]
        p = self._resultado["params"]
        sev = q["severidade"]

        bg_s = CORES.get(f"sev_{sev}_bg", "#E2E8F0")
        fg_s = CORES.get(f"sev_{sev}_fg", "#0F172A")
        self._lbl_sev_badge.setText(SEVERIDADE_ROTULO.get(sev, sev.upper()))
        self._lbl_sev_badge.setStyleSheet(
            f"background:{bg_s}; color:{fg_s}; padding:3px 10px; border-radius:10px; font-weight:800; font-size:11px;"
        )

        self._gauge.atualizar(q["indice_dano_1000"], p["sev_media_1000"], p["sev_alta_1000"])
        self._lbl_kpi_indice.setText(f"{q['indice_dano_1000'] / 10:.1f}".replace(".", ","))
        self._lbl_kpi_trincas.setText(f"{q['n_trincas']} ({q['comprimento_trincas_px']} px)")
        pct_b = f"{q['area_buracos_pct_100'] / 100:.2f}%".replace(".", ",")
        self._lbl_kpi_buracos.setText(f"{q['n_buracos']} ({pct_b})")

        por_tipo = q.get("tipos_trinca", {})
        self._lbl_tipos_dnit.setText(
            f"Tipos DNIT:  Long {por_tipo.get('longitudinal', 0)}  ·  "
            f"Transv {por_tipo.get('transversal', 0)}  ·  "
            f"Diag {por_tipo.get('diagonal', 0)}  ·  "
            f"Jacaré {por_tipo.get('couro_de_jacare', 0)}"
        )

        gt = self._resultado["metricas_gt"]
        if gt is not None:
            f1_t = gt["trinca_tol2"]["f1"]
            iou_t = gt["trinca_exato"]["iou"]
            iou_b = gt["buraco"]["iou"]
            txt_b = f"IoU {iou_b:.2f}" if gt["px_gt_buraco"] > 0 or q["n_buracos"] > 0 else "—"
            self._lbl_gt_strip.setText(
                f"Gabarito: Trinca F1(2px) {f1_t:.2f} · IoU {iou_t:.2f} · Buraco {txt_b}".replace(".", ",")
            )
            self._lbl_gt_strip.setVisible(True)
        else:
            self._lbl_gt_strip.setVisible(False)

    def _atualizar_mosaico(self) -> None:
        if not self._resultado:
            return
        imgs = self._resultado["imagens_bgr"]
        dim = self._resultado["dimensoes"]
        est = self._resultado["estatisticas"]
        res = self._resultado["resumo_didatico"]
        q = self._resultado["quant"]

        legendas_curtas = {
            "1_original": f"{dim['original'][0]}×{dim['original'][1]}",
            "1_roi": f"ROI {dim['roi_pct']}%",
            "2_media": f"Janela {est['janela_trinca']}×{est['janela_trinca']}",
            "2_desvio": f"P95={res['etapa2']['s_xy_p95_roi']}",
            "3a_sementes": f"{est['px_trinca_forte']} px fortes",
            "3a_histerese": f"-{res['etapa3a']['pct_ruido_removido']}% ruído",
            "3a_otsu": f"Corte {est['limiar_otsu']}",
            "3b_textura": f"t_tex={est['limiar_textura']}",
            "4_abertura": f"-{res['etapa4']['pct_removido_abertura']}% limpo",
            "4_fecho_preench": f"+{est['px_buraco_adicionado_preench']} px miolo",
            "5_esqueleto": f"{est['px_esqueleto']} px · {est['px_bifurcacoes']} bif.",
            "5_final": f"Índice {q['indice_dano_1000']/10:.1f} ({res['etapa5']['severidade_label']})".replace(".", ","),
        }
        for chave, tile in self._tiles_mosaico.items():
            if chave in imgs:
                tile.atualizar(imgs[chave], legendas_curtas.get(chave, ""))

    def _popular_tabela_regioes(self) -> None:
        if not self._resultado:
            return
        modo = self._cmb_tipo_reg.currentData() or "trinca_ok"
        self._tab_regioes.clearSelection()

        if modo == "trinca_ok":
            itens = self._resultado["regioes_trinca"]
            self._tab_regioes.setColumnCount(4)
            self._tab_regioes.setHorizontalHeaderLabels(["#", "Tipo DNIT", "Comp.", "Larg. / Bif."])
            self._tab_regioes.setRowCount(len(itens))
            for r, it in enumerate(itens):
                c0 = QTableWidgetItem(str(it["id"]))
                c0.setData(Qt.UserRole, it["bbox"])
                self._tab_regioes.setItem(r, 0, c0)
                self._tab_regioes.setItem(r, 1, QTableWidgetItem(TIPO_TRINCA_ROTULO.get(it["tipo"], it["tipo"])))
                self._tab_regioes.setItem(r, 2, QTableWidgetItem(f"{it['comprimento']} px"))
                self._tab_regioes.setItem(r, 3, QTableWidgetItem(f"{it['largura_px']} px · {it['bifurcacoes']}b"))
        elif modo == "buraco_ok":
            itens = self._resultado["regioes_buraco"]
            self._tab_regioes.setColumnCount(4)
            self._tab_regioes.setHorizontalHeaderLabels(["#", "Área", "Circ.", "Aspecto"])
            self._tab_regioes.setRowCount(len(itens))
            for r, it in enumerate(itens):
                c0 = QTableWidgetItem(str(it["id"]))
                c0.setData(Qt.UserRole, it["bbox"])
                self._tab_regioes.setItem(r, 0, c0)
                self._tab_regioes.setItem(r, 1, QTableWidgetItem(f"{it['area']} px"))
                self._tab_regioes.setItem(r, 2, QTableWidgetItem(f"{it['circularidade_1000']/1000:.2f}"))
                self._tab_regioes.setItem(r, 3, QTableWidgetItem(f"{it['aspecto_1000']/1000:.2f}×"))
        elif modo == "trinca_rej":
            itens = self._resultado["rejeitados_trinca"]
            self._tab_regioes.setColumnCount(3)
            self._tab_regioes.setHorizontalHeaderLabels(["#", "Área/Comp.", "Motivo da Rejeição"])
            self._tab_regioes.setRowCount(len(itens))
            for r, it in enumerate(itens):
                c0 = QTableWidgetItem(str(it["id"]))
                c0.setData(Qt.UserRole, it["bbox"])
                self._tab_regioes.setItem(r, 0, c0)
                self._tab_regioes.setItem(r, 1, QTableWidgetItem(f"{it['area']}px / {it['comprimento']}px"))
                self._tab_regioes.setItem(r, 2, QTableWidgetItem("; ".join(it["motivos"])))
        else:
            itens = self._resultado["rejeitados_buraco"]
            self._tab_regioes.setColumnCount(3)
            self._tab_regioes.setHorizontalHeaderLabels(["#", "Área", "Motivo da Rejeição"])
            self._tab_regioes.setRowCount(len(itens))
            for r, it in enumerate(itens):
                c0 = QTableWidgetItem(str(it["id"]))
                c0.setData(Qt.UserRole, it["bbox"])
                self._tab_regioes.setItem(r, 0, c0)
                self._tab_regioes.setItem(r, 1, QTableWidgetItem(f"{it['area']} px"))
                self._tab_regioes.setItem(r, 2, QTableWidgetItem("; ".join(it["motivos"])))

        hdr = self._tab_regioes.horizontalHeader()
        for col in range(self._tab_regioes.columnCount() - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self._tab_regioes.columnCount() - 1, QHeaderView.Stretch)

    def _ao_selecionar_regiao(self) -> None:
        sel = self._tab_regioes.selectedItems()
        if not sel:
            self._canvas.destacar_bbox(None)
            return
        item0 = self._tab_regioes.item(sel[0].row(), 0)
        bbox = item0.data(Qt.UserRole) if item0 else None
        self._canvas.destacar_bbox(bbox)

    def _atualizar_barra_pixel(self, x: int, y: int, fixado: bool) -> None:
        if not self._resultado:
            return
        info = inspecionar_pixel(self._resultado, x, y)
        if info is None:
            return
        pino = " [FIXO]" if fixado else ""
        self._lbl_pixel_info.setText(
            f"({info['x']:3d}, {info['y']:3d}){pino}  │  "
            f"I={info['intensidade']:3d}  m_xy={info['media']:3d}  s_xy={info['desvio']:2d}  │  "
            f"Trinca: {info['status_trinca']}  │  "
            f"Buraco: {info['status_buraco']}  │  "
            f"{info['estado']}"
        )

    # ================================================================= #
    # EXPORTAÇÃO                                                        #
    # ================================================================= #
    def _dialogo_exportar_png(self) -> None:
        sugestao = f"{Path(self._entrada.identificador).stem}_painel.png"
        arq, _ = QFileDialog.getSaveFileName(
            self, "Salvar Painel Oficial de Etapas (PNG)", sugestao, "Imagem PNG (*.png)"
        )
        if not arq:
            return
        if not arq.lower().endswith(".png"):
            arq += ".png"
        self.exportar_painel_png(arq)

    def exportar_painel_png(self, caminho_destino: str) -> None:
        roi_full = resolver_roi(self._entrada, self._roi_modo, self._poligono_roi)
        params = construir_parametros(self._params_dict)
        png_bytes = gerar_painel_oficial_png(
            self._entrada.img_bgr,
            roi_full,
            params,
            aplicar_reducao_2x=self._reduzir_2x,
            gt_trinca_full=self._entrada.gt_trinca,
            gt_buraco_full=self._entrada.gt_buraco,
        )
        Path(caminho_destino).write_bytes(png_bytes)

    def _dialogo_exportar_json(self) -> None:
        if not self._resultado:
            return
        sugestao = f"{Path(self._entrada.identificador).stem}_relatorio.json"
        arq, _ = QFileDialog.getSaveFileName(
            self, "Salvar Relatório Quantitativo (JSON)", sugestao, "Arquivo JSON (*.json)"
        )
        if not arq:
            return
        if not arq.lower().endswith(".json"):
            arq += ".json"
        self.exportar_relatorio_json(arq)

    def exportar_relatorio_json(self, caminho_destino: str) -> None:
        if not self._resultado:
            return
        rel = montar_relatorio_json(self._entrada, self._resultado)
        Path(caminho_destino).write_text(
            json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def _limpar_layout(cls, lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    cls._limpar_layout(sub)
                    sub.setParent(None)
                    sub.deleteLater()
