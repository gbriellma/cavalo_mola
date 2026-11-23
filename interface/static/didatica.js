/* =========================================================================
   Detecção e Quantificação de Danos em Pavimentos — Conteúdo Didático
   100% fiel ao núcleo inteiro determinístico (src/pavimento/nucleo.py)
   ========================================================================= */
"use strict";

window.PAVIMENTO_INFO = {
  SEVERIDADE_LABEL: {
    nenhum: "Nenhum",
    baixa: "Baixa",
    media: "Média",
    alta: "Alta",
  },

  ETAPAS: {
    "1": {
      requisitoTag: "Item 5 · Cores (HSI) & Fundamentos",
      resumoCurto:
        "Converte RGB para a componente Intensidade do modelo HSI (I = ⌊(R+G+B)/3⌋), reduz 2× por média de blocos 2×2 e aplica a máscara ROI da pista. Descarta cor redundante, corta 75% dos pixels e evita falsos alarmes no entorno.",
      dicaVisualPorSub: {
        "1_roi": "Contorno azul-ciano delimita a pista analisada (ROI); área externa fica 3× mais escura e é ignorada pelo núcleo.",
        "1_hsi": "Intensidade I em tons de cinza sem recorte de ROI; note como vegetação e acostamento ainda competem com trincas.",
        "1_original": "Imagem RGB na resolução de processamento antes da conversão para cinza (reduzida 2× quando a opção está ativa).",
      },
      provaRapida: [
        { id: "sem_roi", label: "Ver sem ROI (pista inteira)", tipo: "roi", payload: "inteira" },
        { id: "sem_2x", label: "Desligar redução 2×", tipo: "reducao", payload: false },
      ],
    },

    "2": {
      requisitoTag: "Item 5 · Filtragem Espacial O(1)",
      resumoCurto:
        "Filtro de caixa em O(1) via somas acumuladas calcula a média local m_xy e os desvios padrão locais s_xy e s_b em aritmética inteira. m_xy acompanha sol e sombra como referência do asfalto, enquanto s_xy e s_b medem a rugosidade.",
      dicaVisualPorSub: {
        "2_media": "Referência borrada m_xy: trincas finas desaparecem, mantendo sombras e variação suave de iluminação da pista.",
        "2_desvio": "Desvio s_xy (×6): asfalto liso fica escuro; bordas de trincas, faixas pintadas e trechos ásperos acendem em claro.",
        "2_desvio_b": "Textura s_b (×6) em janela própria (raio_buraco): destaca regiões amplas e esfareladas para detectar buracos.",
        "2_contraste": "Contraste local m_xy − I (×4): apenas pixels mais escuros que a vizinhança brilham; faixas claras ficam pretas.",
      },
      provaRapida: [
        { id: "r4_antigo", label: "Testar janela 9×9 (r=4)", tipo: "params", payload: { raio: 4 } },
        { id: "r1_ruido", label: "Janela 3×3 (r=1 — ruído)", tipo: "params", payload: { raio: 1 } },
      ],
    },

    "3a": {
      requisitoTag: "Item 5 · Segmentação (Limiar Variável)",
      resumoCurto:
        "Limiarização variável em dois níveis detecta pixels mais escuros que m_xy − (a·s_xy + b) sem ponto flutuante. Após fechamento direcional, a histerese mantém apenas regiões fracas ligadas a pelo menos uma semente forte.",
      dicaVisualPorSub: {
        "3a_sementes": "Sementes fortes (âmbar) marcam o núcleo confiável da trinca; candidatos fracos (ciano) incluem pontas rasas e ruído.",
        "3a_fecho_dir": "Pixels em verde mostram falhas curtas religadas em 4 direções (—, |, /, \\) antes da histerese, sem engrossar a trinca.",
        "3a_histerese": "Componentes com semente forte sobrevivem (vermelho/amarelo); candidatos fracos isolados (roxo) são descartados.",
        "3a_otsu": "Comparação com Otsu global inteiro: um corte único na pista confunde sombras inteiras com defeito.",
      },
      provaRapida: [
        { id: "ver_otsu", label: "Comparar com Otsu Global", tipo: "substage", payload: { substep: "3a_otsu", compareMode: "lado" } },
        { id: "sem_hist", label: "Sem Histerese (só limiar fraco)", tipo: "params", payload: { trinca_a16: 16, trinca_b16: 32 } },
      ],
    },

    "3b": {
      requisitoTag: "Item 5 · Segmentação (Textura + Histograma)",
      resumoCurto:
        "Extrai mediana e MAD diretamente do histograma do desvio s_b na ROI para definir o corte t_tex = mediana + k·MAD. Isola áreas muito mais ásperas que o asfalto típico sem se distorcer por faixas pintadas.",
      dicaVisualPorSub: {
        "3b_textura": "Pixels laranja superam t_tex no mapa de textura; confira no histograma a mediana (azul) separada do limiar (vermelho).",
        "3b_cand": "Máscara binária cand_b: buracos aparecem como blocos largos, enquanto faixas finas ainda aguardam a abertura da Etapa 4.",
      },
      provaRapida: [
        { id: "k_frouxo", label: "Limiar sensível (k=3,0)", tipo: "params", payload: { buraco_k4: 12 } },
      ],
    },

    "4": {
      requisitoTag: "Itens 5 & 10 · Morfologia do Zero (NumPy)",
      resumoCurto:
        "Morfologia binária do zero em NumPy (translações + & / |): abertura remove faixas finas em buracos, fechamento une bordas e dilatação condicional preenche miolos ocos. Nas trincas, fechamento direcional em 4 ângulos emenda lacunas curtas.",
      dicaVisualPorSub: {
        "4_abertura": "Vermelho mostra riscos e faixas finas apagados pela Abertura; branco mostra regiões largas preservadas.",
        "4_fecho_preench": "Branco é a base pós-abertura, verde liga quebras de borda (fechamento) e ciano preenche miolos lisos cercados.",
        "3a_fecho_dir": "No ramo de trincas, pontos verdes mostram quebras de 1–3 px religadas nas 4 direções lineares (—, |, /, \\).",
      },
      provaRapida: [
        { id: "sem_abertura", label: "Ver sem Abertura", tipo: "params", payload: { buraco_abertura: 1 }, substep: "4_abertura" },
        { id: "sem_fecho_dir", label: "Sem Fecho Direcional", tipo: "params", payload: { trinca_fecho: 1 }, substep: "3a_fecho_dir" },
      ],
    },

    "5": {
      requisitoTag: "Itens 5 & 11.5 · Descritores & Classificação",
      resumoCurto:
        "Rotula componentes 8-conexos, filtra buracos por área, circularidade e aspecto, e afina trincas até 1 px (Zhang-Suen) para medir comprimento e bifurcações. Classifica o tipo DNIT (peso 2× em couro de jacaré) e gera Índice e Severidade.",
      dicaVisualPorSub: {
        "5_final": "Resultado final na pista: trincas em vermelho, esqueleto 1 px em amarelo e contorno de buracos em magenta.",
        "5_esqueleto": "Esqueleto Zhang-Suen (amarelo) e bifurcações (ciano) dentro das trincas aceitas; fragmentos rejeitados ficam em roxo.",
        "5_buracos_forma": "Buracos aprovados pelos filtros geométricos em magenta vs candidatos rejeitados em vermelho-coral.",
        "5_gt_comparar": "Comparação pixel a pixel com o gabarito DNIT: amarelo = acerto (TP), verde = perdido (FN), vermelho = alarme falso (FP).",
        "5_gt_rotulo": "Anotação humana do DNIT na ROI: trincas (CRACK) em laranja-âmbar e buracos (POTHOLE) em verde com contorno amarelo.",
      },
      provaRapida: [
        { id: "sem_forma", label: "Desligar filtro de forma (buracos)", tipo: "params", payload: { buraco_area_min: 1, buraco_circ_min_1000: 0, buraco_aspecto_max_1000: 20000 }, substep: "5_buracos_forma" },
      ],
    },
  },

  LEGENDAS_POR_SUBETAPA: {
    "1_original": [
      { color: "#64748B", label: "Foto RGB original na resolução de entrada" },
      { color: "#E2E8F0", label: "Cores naturais da pista, faixas e entorno" },
    ],
    "1_hsi": [
      { color: "#9CA3AF", label: "Intensidade HSI: I = ⌊(R+G+B)/3⌋" },
      { color: "#1F2937", label: "Tons escuros (trincas e sombras)" },
      { color: "#E5E7EB", label: "Tons claros (faixas pintadas e sol)" },
    ],
    "1_roi": [
      { color: "#32C3EB", label: "Contorno da ROI (pista analisada)" },
      { color: "#9CA3AF", label: "Intensidade HSI dentro da ROI" },
      { color: "#374151", label: "Fora da ROI (escurecido 3× e ignorado)" },
    ],
    "2_media": [
      { color: "#9CA3AF", label: "Média local m_xy (suaviza riscos finos)" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "2_desvio": [
      { color: "#1F2937", label: "Escuro = asfalto liso (s_xy baixo)" },
      { color: "#F3F4F6", label: "Claro = borda de trinca ou área áspera (s_xy ×6)" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "2_desvio_b": [
      { color: "#1F2937", label: "Escuro = piso uniforme na janela de buraco" },
      { color: "#F3F4F6", label: "Claro = região rugosa ou borda de faixa (s_b ×6)" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "2_contraste": [
      { color: "#F3F4F6", label: "Claro = pixel mais escuro que m_xy ((m_xy − I) ×4)" },
      { color: "#1F2937", label: "Preto = pixel ≥ m_xy (asfalto normal / faixas)" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "3a_sementes": [
      { color: "#FFC319", label: "Semente forte (alta certeza de trinca)" },
      { color: "#37B9E6", label: "Candidato fraco bruto (pontas e ruídos)" },
      { color: "#286478", label: "Contorno da ROI sobre fundo escurecido" },
    ],
    "3a_fecho_dir": [
      { color: "#FFC319", label: "Semente forte (âncora)" },
      { color: "#379BCD", label: "Candidato fraco pré-fecho" },
      { color: "#5FEB50", label: "Religado pelo fecho direcional (4 ângulos)" },
      { color: "#286478", label: "Contorno da ROI" },
    ],
    "3a_histerese": [
      { color: "#FFD71E", label: "Semente forte em componente mantido" },
      { color: "#F55A41", label: "Candidato fraco mantido por conexão" },
      { color: "#9B55AF", label: "Ruído fraco descartado (sem semente)" },
      { color: "#286478", label: "Contorno da ROI" },
    ],
    "3a_otsu": [
      { color: "#F05F46", label: "Marcado pelo Otsu global (falha em sombras)" },
      { color: "#4B5563", label: "Asfalto acima do corte global" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "3b_textura": [
      { color: "#FAA514", label: "Rugosidade s_b > t_tex (candidato a buraco)" },
      { color: "#374151", label: "Asfalto normal abaixo de t_tex" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "3b_cand": [
      { color: "#F5F5F5", label: "Máscara binária cand_b (pré-abertura)" },
      { color: "#000000", label: "Fundo (s_b ≤ t_tex ou fora da ROI)" },
      { color: "#2D788C", label: "Contorno da ROI" },
    ],
    "4_abertura": [
      { color: "#F5F5F5", label: "Mantido pós-Abertura (regiões largas)" },
      { color: "#EB4B4B", label: "Apagado pela Abertura (faixas e riscos finos)" },
      { color: "#286478", label: "Contorno da ROI" },
    ],
    "4_fecho_preench": [
      { color: "#EBEBEB", label: "Base mantida após Abertura" },
      { color: "#6EEB50", label: "Adicionado pelo Fechamento (bordas)" },
      { color: "#2DC3F5", label: "Miolo preenchido por dilatação condicional" },
      { color: "#286478", label: "Contorno da ROI" },
    ],
    "5_final": [
      { color: "#DC0000", label: "Trinca aceita" },
      { color: "#FFFF00", label: "Esqueleto Zhang-Suen (1 px)" },
      { color: "#FF00FF", label: "Contorno de buraco aceito" },
      { color: "#00C8FF", label: "Contorno da ROI" },
    ],
    "5_esqueleto": [
      { color: "#DC2828", label: "Trinca aceita pelos filtros" },
      { color: "#FFF019", label: "Esqueleto 1 px (Zhang-Suen)" },
      { color: "#28EBFF", label: "Bifurcações (≥ 3 vizinhos-8)" },
      { color: "#734B6E", label: "Candidato rejeitado (forma/tamanho)" },
      { color: "#2D96B4", label: "Contorno da ROI" },
    ],
    "5_buracos_forma": [
      { color: "#DC50E6", label: "Buraco aceito (filtros de forma)" },
      { color: "#FF78FF", label: "Contorno de buraco aceito" },
      { color: "#DC504B", label: "Candidato rejeitado (motivo na tabela)" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    "5_gt_comparar": [
      { color: "#FFFF00", label: "Acerto (TP: predição = gabarito)" },
      { color: "#00C800", label: "Falso negativo (FN: dano real perdido)" },
      { color: "#FF0000", label: "Falso positivo (FP: alarme fora do gabarito)" },
      { color: "#32C3EB", label: "Contorno real da pista (LANE)" },
    ],
    "5_gt_rotulo": [
      { color: "#FFAF1E", label: "Trinca do gabarito DNIT (CRACK)" },
      { color: "#50E646", label: "Buraco do gabarito DNIT (POTHOLE)" },
      { color: "#FFEB28", label: "Contorno do buraco no gabarito" },
      { color: "#32C3EB", label: "Contorno real da pista (LANE)" },
    ],
  },

  REQUISITOS_PROFESSOR: [
    {
      item: "Item 5",
      topico: "Cores (HSI) & Fundamentos",
      comoAtende: "Converte BGR/RGB em Intensidade HSI inteira, reduz 2× por média de blocos 2×2 e isola a faixa de rodagem via ROI.",
      codigo: "nucleo.intensidade_hsi · nucleo.reduzir_2x · nucleo.reduzir_mascara_2x",
      labelBotao: "Ir para Etapa 1",
      acao: { tipo: "step", step: "1", substep: "1_roi" },
    },
    {
      item: "Item 5",
      topico: "Melhorias & Filtragem Espacial",
      comoAtende: "Filtro de caixa O(1) por somas acumuladas calcula média local m_xy e desvio padrão s_xy / s_b recortados nas bordas.",
      codigo: "nucleo.estatisticas_locais · nucleo._soma_janela_1d · nucleo.raiz_inteira",
      labelBotao: "Ir para Etapa 2",
      acao: { tipo: "step", step: "2", substep: "2_desvio" },
    },
    {
      item: "Item 5",
      topico: "Segmentação de Trincas (Limiar Variável)",
      comoAtende: "Aplica limiar variável I < m_xy − (a·s_xy + b) em dois níveis com histerese 8-conexa, superando Otsu global em sombras.",
      codigo: "nucleo.analisar (c) · nucleo.histerese_componentes · nucleo.otsu_inteiro",
      labelBotao: "Comparar vs Otsu",
      acao: { tipo: "step", step: "3a", substep: "3a_otsu", compareMode: "lado" },
    },
    {
      item: "Item 5",
      topico: "Segmentação por Textura + Histograma",
      comoAtende: "Calcula mediana e MAD em O(n) via histograma do desvio s_b na ROI para definir t_tex = mediana + k·MAD resistente a outliers.",
      codigo: "nucleo.mediana_mad · nucleo.analisar (b)",
      labelBotao: "Ver Etapa 3B",
      acao: { tipo: "step", step: "3b", substep: "3b_textura" },
    },
    {
      item: "Itens 5 & 10",
      topico: "Morfologia Matemática do Zero (NumPy)",
      comoAtende: "Erosão, dilatação, abertura, fechamento, borda, preenchimento condicional e fecho direcional 4-ângulos implementados do zero sem cv2.morphologyEx.",
      codigo: "nucleo._desloc · erodir · dilatar · abertura · fechamento · preencher_buracos · morfologia_direcional",
      labelBotao: "Ver Etapa 4",
      acao: { tipo: "step", step: "4", substep: "4_abertura" },
    },
    {
      item: "Itens 5 & 11.3",
      topico: "Descrição, Esqueleto & Classificação",
      comoAtende: "Rotula ilhas 8-conexas, filtra buracos (área, circularidade 4πA/P², aspecto), afina trincas (Zhang-Suen), classifica tipos DNIT e gera Índice/Severidade.",
      codigo: "nucleo.rotular · borda · afinar · contar_vizinhos8 · quantificar · severidade",
      labelBotao: "Ver Esqueleto (5)",
      acao: { tipo: "step", step: "5", substep: "5_esqueleto" },
    },
    {
      item: "Itens 9 & 11.4",
      topico: "Implementação Modular Python/NumPy/OpenCV",
      comoAtende: "Módulos separados para núcleo inteiro, parâmetros, métricas, dataset e visualização, usando NumPy no cálculo e OpenCV apenas em I/O, painéis e rotular().",
      codigo: "src/pavimento/{nucleo,parametros,metricas,dataset,visualizacao}.py",
      labelBotao: "Ver Painel Completo",
      acao: { tipo: "view", view: "mosaico" },
    },
    {
      item: "Itens 11.2 & 11.5",
      topico: "Base DNIT (1.451 img · 785/666) · Métricas & Análise",
      comoAtende: "Split determinístico por hash (785 treino / 666 teste), avaliação única no teste com Precisão, Revocação, F1, IoU, Spearman e análise crítica.",
      codigo: "pavimento.dataset · pavimento.metricas · scripts/avaliar.py",
      labelBotao: "Ver Resultados (666)",
      acao: { tipo: "view", view: "resultados" },
    },
  ],

  ANALISE_CRITICA: {
    pontosFortes: [
      "Núcleo 100% determinístico em aritmética inteira e NumPy do zero, rodando em ~45–52 ms/imagem em CPU sem GPU.",
      "Trincas sólidas nas 666 imagens de teste: F1 por imagem de 0,928 (revocação 100,0%), salto de +66% no F1 tol. 2 px (0,321 → 0,531) e Spearman de 0,712.",
      "Classificação de severidade consistente para triagem: 93,2% de acerto dentro de até 1 nível de diferença (57,4% exato em 4 classes vs 25% do acaso).",
      "Validação sem vazamento de dados: divisão determinística por hash (785 treino / 666 teste) com ajuste exclusivo no treino e avaliação única no teste.",
    ],
    limitacoes: [
      "IoU e F1 de buracos por pixel baixos (0,059 / 0,112): buracos ocupam só 0,69% dos pixels da pista, tornando a métrica muito sensível a falsos positivos.",
      "Falta de contexto semântico em filtros locais 2D: brilho e rugosidade confundem remendos antigos, manchas de óleo, sombras recortadas e faixas gastas.",
      "Rótulos manuais em polígonos aproximados penalizam o IoU exato, e fotos avulsas sem máscara de pista dependem de ROI manual ou poligonal.",
      "Evolução natural: combinar o pipeline clássico com CNNs (U-Net/YOLO-Seg) para semântica ou visão estéreo para confirmar profundidade física real.",
    ],
  },

  PARAM_TRADUCAO: {
    raio: { nomeSimples: "Lupa das trincas", esquerda: "3×3 vê detalhes, capta ruído", direita: "Suaviza mais, perde contraste fino" },
    raio_buraco: { nomeSimples: "Lupa de textura (buracos)", esquerda: "Mede rugosidade pontual", direita: "Captura áreas esfareladas amplas" },
    trinca_a16: { nomeSimples: "Peso do desvio (forte)", esquerda: "Mais sementes em piso áspero", direita: "Exige sinal bem acima do ruído" },
    trinca_b16: { nomeSimples: "Margem escura (forte)", esquerda: "Aceita trincas mais rasas", direita: "Só marca centros bem escuros" },
    trinca_fraca_a16: { nomeSimples: "Peso do desvio (fraca)", esquerda: "Estica pontas em piso áspero", direita: "Freia crescimento em rugosidade" },
    trinca_fraca_b16: { nomeSimples: "Margem escura (fraca)", esquerda: "Captura pontas suaves e ruído", direita: "Restringe pontas a riscos escuros" },
    buraco_k4: { nomeSimples: "Sensibilidade MAD (buracos)", esquerda: "Corte frouxo: mais candidatos", direita: "Corte rígido: só extrema aspereza" },
    buraco_modo: { nomeSimples: "Modo de buracos", esquerda: "0 · Textura + Escuro (histerese)", direita: "1 · Textura Pura (oficial)" },
    buraco_abertura: { nomeSimples: "Peneira Abertura (buracos)", esquerda: "1 px: não limpa faixas finas", direita: "Apaga faixas mais largas e riscos" },
    buraco_fecho: { nomeSimples: "Fechamento (buracos)", esquerda: "1 px: mantém bordas partidas", direita: "Une quebras próximas na borda" },
    trinca_fecho: { nomeSimples: "Fecho direcional (trincas)", esquerda: "1 px: não religa falhas curtas", direita: "3–7 px: emenda falhas em 4 ângulos" },
    trinca_abertura_linha: { nomeSimples: "Suporte linear (trincas)", esquerda: "1 px: aceita manchas curtas", direita: "Exige traço alinhado mais longo" },
    trinca_area_min: { nomeSimples: "Área mínima (trinca)", esquerda: "Aceita fragmentos minúsculos", direita: "Descarta fragmentos pequenos" },
    trinca_comp_min: { nomeSimples: "Comprimento mín. (esqueleto)", esquerda: "Conta riscos bem curtos", direita: "Exige fissuras mais longas" },
    trinca_larg_max16: { nomeSimples: "Largura máxima (trinca)", esquerda: "Só aceita riscos muito finos", direita: "Aceita marcas e manchas largas" },
    buraco_area_min: { nomeSimples: "Área mínima (buraco)", esquerda: "Aceita regiões pequenas", direita: "Só aceita buracos extensos" },
    buraco_circ_min_1000: { nomeSimples: "Circularidade mín. (buraco)", esquerda: "0: aceita qualquer formato", direita: "Exige forma compacta/arredondada" },
    buraco_aspecto_max_1000: { nomeSimples: "Razão de aspecto máx.", esquerda: "Rígido (1,5×): quase quadrado", direita: "Folgado (até 10×): aceita alongados" },
    dano_peso_trinca: { nomeSimples: "Peso das trincas no Índice", esquerda: "Menor impacto na nota final", direita: "Eleva rápido severidade por trincas" },
    dano_peso_buraco: { nomeSimples: "Peso dos buracos no Índice", esquerda: "Menor impacto de % de buraco", direita: "Cada % de buraco pesa mais na nota" },
  },
};
