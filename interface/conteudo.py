"""Metadados concisos das etapas, legendas visuais, parâmetros e resultados."""
from __future__ import annotations

from typing import Dict, List, Tuple

ORDEM_ETAPAS: Tuple[str, ...] = ("1", "2", "3a", "3b", "4", "5")

SEVERIDADE_ROTULO = {
    "nenhum": "NENHUM",
    "baixa": "BAIXA",
    "media": "MÉDIA",
    "alta": "ALTA",
}

TIPO_TRINCA_ROTULO = {
    "longitudinal": "Longitudinal",
    "transversal": "Transversal",
    "diagonal": "Diagonal",
    "couro_de_jacare": "Couro de Jacaré",
}

# Cada sub-etapa: (chave_img, rotulo_botao, chave_antes, rotulo_antes, rotulo_depois)
ETAPAS_PIPELINE: Dict[str, dict] = {
    "1": {
        "codigo": "01",
        "nome_curto": "Pré-proc.",
        "sub_curto": "HSI · 2× · ROI",
        "cor_ramo": "teal",
        "eyebrow": "ETAPA 01 · AQUISIÇÃO E PRÉ-PROCESSAMENTO",
        "titulo": "Intensidade HSI, Redução 2× e Recorte da Pista (ROI)",
        "topico": "Cores (HSI) & Resolução Espacial",
        "formula": "I = ⌊(R + G + B) / 3⌋   ·   Redução 2×2 (shift >> 2)",
        "pontos": [
            "Descarta cor redundante convertendo RGB para Intensidade HSI inteira.",
            "Reduz resolução em 2× por média 2×2 (4× menos pixels, suaviza ruído fino).",
            "Isola a faixa de rolamento (ROI), ignorando acostamento, mato e veículos.",
        ],
        "padrao_sub": "1_roi",
        "subetapas": [
            ("1_roi", "HSI + ROI", "1_original", "Original RGB", "HSI + ROI"),
            ("1_hsi", "Intensidade HSI", "1_original", "Original RGB", "Intensidade HSI"),
            ("1_original", "Original RGB", "1_original", "Original RGB", "Original RGB"),
        ],
        "experimentos": [
            ("Ver sem ROI (imagem inteira)", "roi", "inteira"),
            ("Desligar Redução 2×", "reducao", False),
        ],
    },
    "2": {
        "codigo": "02",
        "nome_curto": "Filtragem O(1)",
        "sub_curto": "Média · Desvio",
        "cor_ramo": "teal",
        "eyebrow": "ETAPA 02 · FILTRAGEM ESPACIAL EM O(1)",
        "titulo": "Média Local (m_xy) e Desvio Padrão Local (s_xy / s_b)",
        "topico": "Filtro de Caixa por Somas Acumuladas",
        "formula": "m_xy = ⌊S₁ / n⌋   ·   s_xy = √⌊(n·S₂ − S₁²) / n²⌋",
        "pontos": [
            "Somas acumuladas (imagem integral) calculam qualquer janela em custo O(1).",
            "m_xy estima a iluminação local (sol/sombra), apagando trincas finas.",
            "s_xy e s_b medem a rugosidade local com janelas independentes.",
        ],
        "padrao_sub": "2_desvio",
        "subetapas": [
            ("2_media", "Média m_xy", "1_roi", "HSI + ROI", "Média m_xy"),
            ("2_desvio", "Desvio s_xy", "1_roi", "HSI + ROI", "Desvio s_xy ×6"),
            ("2_desvio_b", "Textura s_b", "1_roi", "HSI + ROI", "Textura s_b ×6"),
            ("2_contraste", "Contraste m_xy−I", "1_roi", "HSI + ROI", "Contraste ×4"),
        ],
        "experimentos": [
            ("Janela 9×9 (r=4)", "params", {"raio": 4}),
            ("Janela 3×3 (r=1 · ruído)", "params", {"raio": 1}),
        ],
    },
    "3a": {
        "codigo": "3A",
        "nome_curto": "Seg. Trincas",
        "sub_curto": "Limiar + Histerese",
        "cor_ramo": "teal",
        "eyebrow": "ETAPA 3A · SEGMENTAÇÃO ADAPTATIVA DE TRINCAS",
        "titulo": "Limiar Variável em 2 Níveis + Histerese 8-Conexa",
        "topico": "Segmentação Local & Crescimento por Sementes",
        "formula": "I(x,y) < m_xy − (a · s_xy + b)   ·   Forte ⊂ Fraca",
        "pontos": [
            "Compara cada pixel com sua vizinhança — imune a sombras de árvores.",
            "Sementes fortes marcam o centro escuro; máscara fraca pega pontas rasas.",
            "Histerese mantém só componentes fracos conectados a ≥ 1 semente forte.",
        ],
        "padrao_sub": "3a_histerese",
        "subetapas": [
            ("3a_sementes", "Sementes", "1_roi", "HSI + ROI", "Sementes + Fracas"),
            ("3a_fecho_dir", "Fecho 4-Âng.", "3a_sementes", "Pré-Fecho", "Pós-Fecho 4 Ângulos"),
            ("3a_histerese", "Pós-Histerese", "3a_sementes", "Candidatos Brutos", "Pós-Histerese"),
            ("3a_otsu", "vs. Otsu Global", "3a_histerese", "Limiar Variável (Nosso)", "Otsu Global Fixo"),
        ],
        "experimentos": [
            ("Lado a Lado c/ Otsu", "otsu_lado", None),
            ("Sem Histerese", "params", {"trinca_a16": 16, "trinca_b16": 32}),
        ],
    },
    "3b": {
        "codigo": "3B",
        "nome_curto": "Seg. Buracos",
        "sub_curto": "Mediana + MAD",
        "cor_ramo": "amber",
        "eyebrow": "ETAPA 3B · SEGMENTAÇÃO ROBUSTA DE BURACOS",
        "titulo": "Mapa de Textura com Limiar por Histograma (Mediana + MAD)",
        "topico": "Estatísticas de Histograma & Textura",
        "formula": "t_tex = Mediana(s_b) + k · MAD(s_b)   (k = k₄ / 4)",
        "pontos": [
            "Buracos nem sempre são escuros, mas apresentam alta aspereza (s_b).",
            "Mediana e MAD pelo histograma não se distorcem com faixas brancas.",
            "Marca toda região com rugosidade acima do padrão normal da pista.",
        ],
        "padrao_sub": "3b_textura",
        "subetapas": [
            ("3b_textura", "Textura > t_tex", "2_desvio_b", "Mapa s_b", "Acima de t_tex"),
            ("3b_cand", "Máscara cand_b", "1_roi", "HSI + ROI", "Candidatos cand_b"),
        ],
        "experimentos": [
            ("Limiar Sensível (k=3,0)", "params", {"buraco_k4": 12}),
        ],
    },
    "4": {
        "codigo": "04",
        "nome_curto": "Morfologia",
        "sub_curto": "Do Zero · NumPy",
        "cor_ramo": "amber",
        "eyebrow": "ETAPA 04 · MORFOLOGIA MATEMÁTICA IMPLEMENTADA DO ZERO",
        "titulo": "Abertura, Fechamento, Preenchimento e EE Lineares 4-Ângulos",
        "topico": "Morfologia Binária Pura (sem cv2.morphologyEx)",
        "formula": "Abertura = (A ⊖ B) ⊕ B   ·   EE Lineares: (—, |, /, \\)",
        "pontos": [
            "Abertura apaga faixas pintadas e riscos estreitos do mapa de buracos.",
            "Dilatação condicional preenche miolos lisos (água/poeira) cercados.",
            "Fechamento direcional em 4 ângulos emenda trincas sem borrar a largura.",
        ],
        "padrao_sub": "4_abertura",
        "subetapas": [
            ("4_abertura", "Abertura (Buracos)", "3b_cand", "Pré-Abertura", "Pós-Abertura"),
            ("4_fecho_preench", "Fecho + Preench.", "4_abertura", "Pós-Abertura", "Miolo Preenchido"),
            ("3a_fecho_dir", "Fecho 4-Âng. (Trincas)", "3a_sementes", "Pré-Fecho", "Pós-Fecho Direcional"),
        ],
        "experimentos": [
            ("Ver sem Abertura", "params_sub", ({"buraco_abertura": 1}, "4_abertura")),
            ("Sem Fecho Direcional", "params_sub", ({"trinca_fecho": 1}, "3a_fecho_dir")),
        ],
    },
    "5": {
        "codigo": "05",
        "nome_curto": "Quantificação",
        "sub_curto": "Esqueleto · Nota",
        "cor_ramo": "teal",
        "eyebrow": "ETAPA 05 · ESQUELETO ZHANG-SUEN, FORMA E QUANTIFICAÇÃO",
        "titulo": "Componentes 8-Conexos, Classificação DNIT e Índice de Dano",
        "topico": "Extração de Características & Classificação",
        "formula": "Comp. = |Esqueleto|   ·   Larg. = Área/Comp.   ·   Circ. = 4πA/P²",
        "pontos": [
            "Esqueleto Zhang-Suen (1 px) mede comprimento, largura e bifurcações.",
            "Classifica trincas pela norma DNIT (couro de jacaré recebe peso 2×).",
            "Filtra buracos por área, circularidade e aspecto; gera Índice (0–100).",
        ],
        "padrao_sub": "5_final",
        "subetapas": [
            ("5_final", "Resultado Final", "1_original", "Original RGB", "Resultado Final"),
            ("5_esqueleto", "Esqueleto 1 px", "3a_histerese", "Pós-Histerese", "Esqueleto 1 px"),
            ("5_buracos_forma", "Forma (Buracos)", "4_fecho_preench", "Pré-Filtro", "Aceitos vs Rejeitados"),
        ],
        "subetapas_gt": [
            ("5_gt_comparar", "vs. Gabarito", "5_final", "Nossa Predição", "Comparação Gabarito"),
            ("5_gt_rotulo", "Rótulo × Predição", "5_gt_rotulo", "Rótulo Humano DNIT", "Nossa Predição"),
        ],
        "experimentos": [
            (
                "Desligar Filtro de Forma",
                "params_sub",
                ({"buraco_area_min": 1, "buraco_circ_min_1000": 0, "buraco_aspecto_max_1000": 20000}, "5_buracos_forma"),
            ),
        ],
    },
}

LEGENDAS_SUBETAPA: Dict[str, List[Tuple[str, str]]] = {
    "1_original": [("#64748B", "Foto RGB de entrada"), ("#E2E8F0", "Cores originais da via")],
    "1_hsi": [("#9CA3AF", "Intensidade I = ⌊(R+G+B)/3⌋"), ("#1F2937", "Escuro (trincas/sombras)"), ("#F3F4F6", "Claro (faixas/sol)")],
    "1_roi": [("#32C3EB", "Contorno da pista (ROI)"), ("#9CA3AF", "Pista analisada"), ("#374151", "Fora da ROI (ignorado)")],
    "2_media": [("#9CA3AF", "Média local m_xy (iluminação)"), ("#3C8CA0", "Contorno ROI")],
    "2_desvio": [("#1F2937", "Liso (s_xy baixo)"), ("#F3F4F6", "Áspero / bordas (s_xy ×6)"), ("#3C8CA0", "Contorno ROI")],
    "2_desvio_b": [("#1F2937", "Uniforme"), ("#F3F4F6", "Rugoso na janela de buraco (s_b ×6)"), ("#3C8CA0", "Contorno ROI")],
    "2_contraste": [("#F3F4F6", "Mais escuro que vizinhança ((m_xy − I) ×4)"), ("#1F2937", "Asfalto normal / claro")],
    "3a_sementes": [("#FFC319", "Semente forte (alta certeza)"), ("#37B9E6", "Candidata fraca (pontas + ruído)")],
    "3a_fecho_dir": [("#FFC319", "Semente forte"), ("#379BCD", "Candidata pré-fecho"), ("#5FEB50", "Religado em 4 ângulos")],
    "3a_histerese": [("#FFD71E", "Semente forte"), ("#F55A41", "Fraca mantida por conexão"), ("#9B55AF", "Ruído descartado")],
    "3a_otsu": [("#F05F46", "Otsu global fixo (marca sombras!)"), ("#4B5563", "Acima do corte global")],
    "3b_textura": [("#FAA514", "Rugosidade s_b > t_tex"), ("#4B5563", "Asfalto normal ≤ t_tex")],
    "3b_cand": [("#F5F5F5", "Candidato cand_b (pré-abertura)"), ("#111827", "Fundo")],
    "4_abertura": [("#F5F5F5", "Preservado pós-Abertura"), ("#EB4B4B", "Removido pela Abertura (faixas finas)")],
    "4_fecho_preench": [("#EBEBEB", "Base pós-Abertura"), ("#6EEB50", "Fechamento de borda"), ("#2DC3F5", "Miolo preenchido")],
    "5_final": [("#DC0000", "Trinca"), ("#FFFF00", "Esqueleto 1 px"), ("#FF00FF", "Buraco"), ("#00C8FF", "ROI")],
    "5_esqueleto": [("#DC2828", "Trinca aceita"), ("#FFF019", "Esqueleto Zhang-Suen"), ("#28EBFF", "Bifurcação"), ("#734B6E", "Rejeitada")],
    "5_buracos_forma": [("#DC50E6", "Buraco aceito"), ("#DC504B", "Rejeitado (forma/área)")],
    "5_gt_comparar": [("#FFFF00", "Acerto (TP)"), ("#00C800", "Não detectado (FN)"), ("#FF0000", "Falso alarme (FP)")],
    "5_gt_rotulo": [("#FFAF1E", "Trinca real (CRACK)"), ("#50E646", "Buraco real (POTHOLE)"), ("#32C3EB", "Pista (LANE)")],
}

# Quadros exibidos na aba "Painel das Etapas (Mosaico)"
# (etapa_id, chave_img, codigo_badge, titulo_quadro)
MOSAICO_QUADROS: List[Tuple[str, str, str, str]] = [
    ("1", "1_original", "01", "Original RGB"),
    ("1", "1_roi", "01", "Intensidade HSI + ROI"),
    ("2", "2_media", "02", "Média Local m_xy (O(1))"),
    ("2", "2_desvio", "02", "Desvio Padrão s_xy (Textura)"),
    ("3a", "3a_sementes", "3A", "Trincas: Sementes + Fracas"),
    ("3a", "3a_histerese", "3A", "Trincas: Pós-Histerese"),
    ("3a", "3a_otsu", "3A", "Comparação: Otsu Global"),
    ("3b", "3b_textura", "3B", "Buracos: Textura > Mediana+MAD"),
    ("4", "4_abertura", "04", "Morfologia: Abertura (Limpa Faixas)"),
    ("4", "4_fecho_preench", "04", "Morfologia: Fecho + Preenchimento"),
    ("5", "5_esqueleto", "05", "Esqueleto Zhang-Suen (1 px)"),
    ("5", "5_final", "05", "Resultado Final Quantificado"),
]

# Controles de parâmetros agrupados: (grupo, [(chave, nome_curto, min, max, passo, divisor_exibicao, sufixo)])
GRUPOS_PARAMETROS = [
    (
        "Etapa 02 · Filtragem Espacial",
        [
            ("raio", "Raio janela trincas", 1, 18, 1, 1, " px"),
            ("raio_buraco", "Raio janela buracos", 1, 18, 1, 1, " px"),
        ],
    ),
    (
        "Etapa 3A & 04 · Trincas",
        [
            ("trinca_a16", "Peso desvio forte (a)", 0, 64, 2, 16, "×"),
            ("trinca_b16", "Margem escura forte (b)", 0, 320, 8, 16, ""),
            ("trinca_fraca_a16", "Peso desvio fraco", 0, 48, 2, 16, "×"),
            ("trinca_fraca_b16", "Margem escura fraca", 0, 160, 8, 16, ""),
            ("trinca_fecho", "Fecho direcional (EE)", 1, 9, 1, 1, " px"),
            ("trinca_area_min", "Área mínima trinca", 1, 100, 1, 1, " px"),
            ("trinca_comp_min", "Comprimento mín.", 1, 60, 1, 1, " px"),
            ("trinca_larg_max16", "Largura máxima", 16, 160, 4, 16, " px"),
        ],
    ),
    (
        "Etapa 3B & 04 · Buracos",
        [
            ("buraco_k4", "Multiplicador MAD (k)", 4, 60, 2, 4, "×"),
            ("buraco_abertura", "Abertura morfológica", 1, 15, 1, 1, " px"),
            ("buraco_fecho", "Fechamento morfológico", 1, 15, 1, 1, " px"),
            ("buraco_area_min", "Área mínima buraco", 10, 800, 15, 1, " px"),
            ("buraco_circ_min_1000", "Circularidade mín.", 0, 700, 25, 1000, ""),
            ("buraco_aspecto_max_1000", "Aspecto máximo", 1000, 10000, 250, 1000, "×"),
        ],
    ),
    (
        "Etapa 05 · Índice e Severidade",
        [
            ("dano_peso_trinca", "Peso trincas", 0, 120, 5, 1, ""),
            ("dano_peso_buraco", "Peso buracos", 0, 100, 5, 1, ""),
            ("sev_media_1000", "Corte Sev. Média", 20, 500, 10, 10, "/100"),
            ("sev_alta_1000", "Corte Sev. Alta", 200, 950, 10, 10, "/100"),
        ],
    ),
]

METRICAS_OFICIAIS_TESTE = [
    ("Trincas — Detecção por imagem (F1)", "0,927", "0,928", "0,928", "Revocação 100,0% (0 FN)"),
    ("Trincas — Pixel tol. 2 px (F1)", "0,321", "0,430", "0,531", "+66% sobre o inicial"),
    ("Trincas — Pixel exato (F1 / IoU)", "0,200 / 0,111", "0,234 / 0,132", "0,326 / 0,194", "+63% em F1 exato"),
    ("Comprimento de trincas (Spearman)", "0,554", "0,683", "0,712", "Forte ordenação relativa"),
    ("Buracos — Detecção por imagem (F1)", "0,330", "0,380", "0,380", "Revocação 33,8% → 53,8%"),
    ("Buracos — Pixel (F1 / IoU)", "0,085 / 0,044", "0,112 / 0,059", "0,112 / 0,059", "Apenas 0,69% dos pixels"),
    ("Severidade — Acerto exato (4 classes)", "48,5%", "56,3%", "57,4%", "Acaso = 25%"),
    ("Severidade — Erro de até 1 nível", "90,1%", "94,1%", "93,2%", "Triagem consistente"),
]

# Mapeamento conciso dos tópicos de Descricao de projeto.pdf -> etapa interativa
REQUISITOS_DISCIPLINA = [
    ("Item 5", "Cores (HSI) & Fundamentos", "RGB → Intensidade HSI inteira, redução 2×2 e máscara ROI", "nucleo.intensidade_hsi · reduzir_2x", "1", "1_roi"),
    ("Item 5", "Melhorias & Filtragem Espacial", "Filtro de caixa O(1) via somas acumuladas (m_xy e s_xy)", "nucleo.estatisticas_locais", "2", "2_desvio"),
    ("Item 5", "Segmentação (Limiar Variável)", "Dois níveis + histerese 8-conexa vs. Otsu global inteiro", "nucleo.histerese_componentes · otsu_inteiro", "3a", "3a_histerese"),
    ("Item 5", "Textura & Histograma", "Mediana e MAD por histograma na ROI (t_tex = med + k·MAD)", "nucleo.mediana_mad", "3b", "3b_textura"),
    ("Itens 5 & 10", "Morfologia Matemática do Zero", "Erosão, dilatação, abertura, fecho 4-ângulos e preenchimento em NumPy", "nucleo.abertura · preencher_buracos", "4", "4_abertura"),
    ("Itens 5 & 11", "Descrição, Esqueleto & Padrões", "Zhang-Suen (1 px), circularidade 4πA/P², tipos DNIT e severidade", "nucleo.afinar · quantificar", "5", "5_esqueleto"),
]

PONTOS_FORTES_RESUMO = [
    "Núcleo 100% inteiro e determinístico do zero em NumPy (~45 ms/imagem em CPU).",
    "Salto de +66% no F1 tol. 2 px de trincas (0,321 → 0,531) e Spearman 0,712.",
    "Severidade acerta 93,2% em até 1 nível (57,4% exato em 4 classes vs. 25% acaso).",
    "Validação limpa por hash MD5: 785 treino / 666 teste avaliado uma única vez.",
]

LIMITACOES_RESUMO = [
    "Buracos somam só 0,69% dos pixels da pista — falsos positivos pesam forte no IoU (0,059).",
    "Filtros locais 2D confundem remendos escuros, manchas de óleo e sombras recortadas.",
    "Polígonos manuais aproximados no dataset penalizam métricas pixel a pixel.",
    "Próximos passos naturais: CNNs (contexto semântico) ou visão estéreo (profundidade 3D).",
]
