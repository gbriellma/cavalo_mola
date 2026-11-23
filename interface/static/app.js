/* =========================================================================
   Detecção de Danos em Pavimentos — Frontend Interativo (100% offline)
   PI 2026.2 · UFRPE
   ========================================================================= */
"use strict";

const STEP_ORDER = ["1", "2", "3a", "3b", "4", "5"];

const STAGE_CONFIG = {
  "1": {
    eyebrow: "ETAPA 01 · PRÉ-PROCESSAMENTO",
    accent: "teal",
    title: "Intensidade HSI, Redução 2× e Recorte da ROI",
    substages: [
      { key: "1_roi", label: "Intensidade HSI + ROI", antes: "1_original", antesLbl: "Original RGB", depoisLbl: "HSI + ROI" },
      { key: "1_hsi", label: "Intensidade HSI (sem ROI)", antes: "1_original", antesLbl: "Original RGB", depoisLbl: "Intensidade HSI" },
      { key: "1_original", label: "Imagem Original (RGB)", antes: "1_original", antesLbl: "Original RGB", depoisLbl: "Original RGB" },
    ],
    legend: [
      { color: "#32C3EB", label: "Contorno da ROI (pista analisada)" },
      { color: "#9CA3AF", label: "Intensidade HSI dentro da pista" },
      { color: "#374151", label: "Fora da ROI (ignorado)" },
    ],
    oQueFaz:
      "Converte a foto RGB para Intensidade do modelo HSI (tons de cinza), reduz a resolução em 2× pela média de blocos 2×2 e aplica a máscara ROI da pista.",
    formula: "I = floor((R + G + B) / 3)\nreduzir_2x: (a + b + c + d + 2) >> 2",
    formulaNote: "100% aritmética inteira — determinismo exato sem ponto flutuante.",
    ref: "7_1 Cores (slides 60–73 · HSI) e 2_1 Fundamentos (slides 49–88). Código: nucleo.py → intensidade_hsi(), reduzir_2x(), reduzir_mascara_2x().",
    paramGroup: "etapa1",
  },
  "2": {
    eyebrow: "ETAPA 02 · FILTRAGEM ESPACIAL O(1)",
    accent: "teal",
    title: "Média Local (m_xy) e Desvios Locais (s_xy / s_b)",
    substages: [
      { key: "2_media", label: "Média Local (m_xy)", antes: "1_roi", antesLbl: "Intensidade I", depoisLbl: "Média m_xy" },
      { key: "2_desvio", label: "Desvio Trincas (s_xy)", antes: "1_roi", antesLbl: "Intensidade I", depoisLbl: "Desvio s_xy (×6)" },
      { key: "2_desvio_b", label: "Textura Buracos (s_b)", antes: "1_roi", antesLbl: "Intensidade I", depoisLbl: "Textura s_b (×6)" },
      { key: "2_contraste", label: "Contraste Local (m_xy − I)", antes: "1_roi", antesLbl: "Intensidade I", depoisLbl: "m_xy − I (×4)" },
    ],
    legend: [
      { color: "#1F2937", label: "Escuro no desvio = asfalto liso" },
      { color: "#F3F4F6", label: "Claro no desvio = superfície áspera / borda" },
      { color: "#3C8CA0", label: "Contorno da ROI" },
    ],
    oQueFaz:
      "Calcula média local m_xy e desvio padrão local s_xy (janela de trincas) e s_b (janela de buracos) em O(1) por pixel via somas acumuladas.",
    formula: "m_xy = floor(S1 / n)\ns²_xy = floor((n·S2 − S1²) / n²)\ns_xy = isqrt(s²_xy)",
    formulaNote: "S1 = Σg e S2 = Σg² via np.cumsum em duas passadas 1D (custo independe do raio).",
    ref: "4_3 Filtragem (slides 31–36 · filtro de caixa) e 7_1 Segmentação (slides 19–24, 98–99). Código: nucleo.py → estatisticas_locais(), raiz_inteira().",
    paramGroup: "etapa2",
  },
  "3a": {
    eyebrow: "ETAPA 3A · SEGMENTAÇÃO DE TRINCAS",
    accent: "teal",
    title: "Limiar Variável, Fecho Direcional e Histerese",
    substages: [
      { key: "3a_sementes", label: "1. Sementes Fortes × Fracas", antes: "1_roi", antesLbl: "Intensidade I", depoisLbl: "Fortes + Fracas" },
      { key: "3a_fecho_dir", label: "2. Fecho Direcional", antes: "3a_sementes", antesLbl: "Antes do Fecho", depoisLbl: "Após Fecho Direcional" },
      { key: "3a_histerese", label: "3. Histerese (Mantidos × Ruído)", antes: "3a_fecho_dir", antesLbl: "Fracas + Fortes", depoisLbl: "Pós-Histerese" },
      { key: "3a_otsu", label: "Comparar: Otsu Global", antes: "3a_histerese", antesLbl: "Método Local", depoisLbl: "Otsu Global" },
    ],
    legend: [
      { color: "#FFC319", label: "Semente forte (núcleo da trinca)" },
      { color: "#37B9E6", label: "Candidato fraco (pontas + ruído)" },
      { color: "#5FEB50", label: "Religado pelo fecho direcional (4 ângulos)" },
      { color: "#F55A41", label: "Componente mantido pela histerese" },
      { color: "#9B55AF", label: "Ruído fraco descartado" },
    ],
    oQueFaz:
      "Limiarização variável em dois níveis detecta sementes fortes e candidatos fracos; após fechamento direcional, a histerese preserva apenas componentes ligados a sementes fortes.",
    formula: "I(x,y) < m_xy − (a · s_xy + b)\nInteiro: [16·(m_xy − I) − b16]² > a16² · var",
    formulaNote: "Elevar ao quadrado elimina raiz quadrada e floats na decisão pixel a pixel.",
    ref: "7_1 Segmentação (slides 96–100 · limiar variável) e 7_2 Segmentação (slides 4–5 · crescimento por sementes). Código: nucleo.py → analisar() parte (c), histerese_componentes().",
    paramGroup: "etapa3a",
  },
  "3b": {
    eyebrow: "ETAPA 3B · SEGMENTAÇÃO DE BURACOS",
    accent: "amber",
    title: "Textura Local com Limiar Robusto (Mediana + k·MAD)",
    substages: [
      { key: "3b_textura", label: "Mapa de Textura + Limiar", antes: "2_desvio_b", antesLbl: "Desvio s_b", depoisLbl: "Acima de t_tex" },
      { key: "3b_cand", label: "Máscara Binária (cand_b)", antes: "1_roi", antesLbl: "Intensidade I", depoisLbl: "Candidatos cand_b" },
    ],
    legend: [
      { color: "#FAA514", label: "Rugosidade s_b > t_tex (candidato a buraco)" },
      { color: "#F5F5F5", label: "Máscara binária cand_b" },
    ],
    oQueFaz:
      "Calcula mediana e MAD em O(n) via histograma de s_b na ROI e define o corte robusto t_tex = mediana + k·MAD.",
    formula: "t_tex = mediana(s_b) + floor(buraco_k4 · MAD(s_b) / 4)\ncand_b = (s_b > t_tex) & roi",
    formulaNote: "Mediana e MAD são obtidos diretamente via np.bincount (histograma inteiro de 256 níveis).",
    ref: "4_2 Histograma (slides 2–9) e 7_1 Segmentação (slides 19–24 · textura; slides 85–88). Código: nucleo.py → mediana_mad().",
    paramGroup: "etapa3b",
  },
  "4": {
    eyebrow: "ETAPA 04 · MORFOLOGIA DO ZERO (NUMPY)",
    accent: "amber",
    title: "Abertura, Fechamento, Preenchimento e Fecho Direcional",
    substages: [
      { key: "4_abertura", label: "Buracos: Abertura (Limpa Faixas)", antes: "3b_cand", antesLbl: "Candidatos cand_b", depoisLbl: "Pós-Abertura" },
      { key: "4_fecho_preench", label: "Buracos: Fecho + Preenchimento", antes: "4_abertura", antesLbl: "Pós-Abertura", depoisLbl: "Pós-Preenchimento" },
      { key: "3a_fecho_dir", label: "Trincas: Fecho Direcional (4 EEs)", antes: "3a_sementes", antesLbl: "Antes do Fecho", depoisLbl: "Após Fecho 4-Ângulos" },
    ],
    legend: [
      { color: "#F5F5F5", label: "Mantido pós-Abertura" },
      { color: "#EB4B4B", label: "Apagado pela Abertura (faixas e riscos finos)" },
      { color: "#6EEB50", label: "Adicionado pelo Fechamento" },
      { color: "#2DC3F5", label: "Miolo preenchido por dilatação condicional" },
    ],
    oQueFaz:
      "Operações morfológicas binárias do zero em NumPy (translações + & / |): Abertura limpa faixas estreitas, Fechamento une bordas, Preenchimento fecha miolos e Fecho Direcional religa trincas.",
    formula: "Abertura:   A ∘ B = (A ⊖ B) ⊕ B\nFechamento: A • B = (A ⊕ B) ⊖ B\nPreench.:   X_k = (X_{k-1} ⊕ B_cruz) ∩ A^c",
    formulaNote: "Implementado sem cv2.morphologyEx — apenas _desloc() e operadores lógicos bit a bit.",
    ref: "6_1 Morfologia (slides 22–42, 61–66). Código: nucleo.py → erodir(), dilatar(), abertura(), fechamento(), preencher_buracos(), morfologia_direcional().",
    paramGroup: "etapa4",
  },
  "5": {
    eyebrow: "ETAPA 05 · RESULTADO E QUANTIFICAÇÃO",
    accent: "teal",
    title: "Esqueleto Zhang-Suen, Filtros de Forma e Índice",
    substages: [
      { key: "5_final", label: "Resultado Final", antes: "1_original", antesLbl: "Original RGB", depoisLbl: "Resultado Final" },
      { key: "5_esqueleto", label: "Trincas + Esqueleto + Bifurcações", antes: "3a_histerese", antesLbl: "Pós-Histerese", depoisLbl: "Esqueleto Zhang-Suen" },
      { key: "5_buracos_forma", label: "Buracos: Aceitos × Rejeitados", antes: "4_fecho_preench", antesLbl: "Pós-Preenchimento", depoisLbl: "Filtros de Forma" },
    ],
    legend: [
      { color: "#DC2828", label: "Trinca aceita" },
      { color: "#FFF019", label: "Esqueleto Zhang-Suen (1 px)" },
      { color: "#28EBFF", label: "Bifurcação (≥3 vizinhos-8) / ROI" },
      { color: "#E650DC", label: "Buraco aceito" },
      { color: "#DC504B", label: "Componente rejeitado" },
    ],
    oQueFaz:
      "Rotula componentes 8-conexos, filtra buracos por forma, afina trincas até 1 px (Zhang-Suen), classifica tipos DNIT e calcula Índice de Dano e Severidade.",
    formula: "Circularidade: C = 4πA / P²   (P = |A − (A ⊖ B_3×3)|)\nComprimento = |Esqueleto|  ·  Largura = Área / Comp.\nÍndice = min(1000, w_t·1000·L_pond/N_roi + w_b·pct_b/10)",
    formulaNote: "L_pond aplica peso 2× para trincas classificadas como couro de jacaré.",
    ref: "6_1 (slides 59–60, 67–73), 9_2 Características (slides 49, 55–68) e Zhang-Suen. Código: nucleo.py → rotular(), borda(), afinar(), quantificar().",
    paramGroup: "etapa5",
  },
};

const PARAM_GROUPS = [
  {
    id: "etapa2",
    title: "Etapa 2 · Janelas do Filtro de Caixa",
    items: [
      { key: "raio", label: "Raio (Trincas)", min: 1, max: 18, step: 1, fmt: (v) => `${v} (${2 * v + 1}×${2 * v + 1})`, help: "Janela menor preserva fissuras finas." },
      { key: "raio_buraco", label: "Raio (Buracos)", min: 2, max: 16, step: 1, fmt: (v) => `${v} (${2 * v + 1}×${2 * v + 1})`, help: "Janela maior captura áreas esfareladas." },
    ],
  },
  {
    id: "etapa3a",
    title: "Etapa 3A · Limiar Variável e Histerese",
    items: [
      { key: "trinca_a16", label: "Forte: a16", min: 0, max: 64, step: 2, fmt: (v) => `${v} (a=${(v / 16).toFixed(2)})`, help: "Exigência sobre o desvio local." },
      { key: "trinca_b16", label: "Forte: b16", min: 0, max: 320, step: 8, fmt: (v) => `${v} (b=${(v / 16).toFixed(1)})`, help: "Margem mínima de escurecimento." },
      { key: "trinca_fraca_a16", label: "Fraca: a16", min: 0, max: 48, step: 2, fmt: (v) => `${v} (a=${(v / 16).toFixed(2)})`, help: "Crescimento das pontas rasas." },
      { key: "trinca_fraca_b16", label: "Fraca: b16", min: 0, max: 160, step: 4, fmt: (v) => `${v} (b=${(v / 16).toFixed(1)})`, help: "Limiar mínimo para candidatos fracos." },
    ],
  },
  {
    id: "etapa3b",
    title: "Etapa 3B · Textura e Limiar (Buracos)",
    items: [
      { key: "buraco_k4", label: "Fator k (buraco_k4)", min: 4, max: 60, step: 2, fmt: (v) => `${v} (k=${(v / 4).toFixed(2)})`, help: "t_tex = mediana + k·MAD." },
      { key: "buraco_modo", label: "Modo de buracos", min: 0, max: 1, step: 1, fmt: (v) => (v === 1 ? "1 · Textura" : "0 · Textura+Escuro"), help: "Modo 1 é o oficial de params.json." },
    ],
  },
  {
    id: "etapa4",
    title: "Etapa 4 · Morfologia Matemática",
    items: [
      { key: "buraco_abertura", label: "Abertura de buracos", min: 1, max: 15, step: 1, fmt: (v) => `${v} px`, help: "Remove faixas finas e ruídos estreitos." },
      { key: "buraco_fecho", label: "Fechamento de buracos", min: 1, max: 15, step: 1, fmt: (v) => `${v} px`, help: "Une quebras na borda do buraco." },
      { key: "trinca_fecho", label: "Fecho direcional (trincas)", min: 1, max: 7, step: 2, fmt: (v) => (v === 1 ? "1 (desligado)" : `${v} px (4 ângulos)`), help: "Religa falhas alinhadas nas 4 direções." },
      { key: "trinca_abertura_linha", label: "Suporte linear (trincas)", min: 1, max: 7, step: 2, fmt: (v) => `${v} px`, help: "Exige traço alinhado mínimo." },
    ],
  },
  {
    id: "etapa5",
    title: "Etapa 5 · Filtros Geométricos e Índice",
    items: [
      { key: "trinca_area_min", label: "Trinca: área mínima", min: 5, max: 100, step: 1, fmt: (v) => `${v} px`, help: "Descarta fragmentos minúsculos." },
      { key: "trinca_comp_min", label: "Trinca: comp. mínimo", min: 2, max: 50, step: 1, fmt: (v) => `${v} px`, help: "Comprimento mínimo do esqueleto." },
      { key: "trinca_larg_max16", label: "Trinca: largura máx.", min: 24, max: 160, step: 4, fmt: (v) => `${v} (${(v / 16).toFixed(2)} px)`, help: "Rejeita manchas largas demais." },
      { key: "buraco_area_min", label: "Buraco: área mínima", min: 20, max: 600, step: 10, fmt: (v) => `${v} px`, help: "Filtra regiões ásperas pequenas." },
      { key: "buraco_circ_min_1000", label: "Buraco: circularidade mín.", min: 0, max: 800, step: 25, fmt: (v) => `${v} (${(v / 1000).toFixed(3)})`, help: "4πA/P² mínimo." },
      { key: "buraco_aspecto_max_1000", label: "Buraco: aspecto máx.", min: 1500, max: 10000, step: 250, fmt: (v) => `${(v / 1000).toFixed(2)}×`, help: "Razão maior/menor lado da caixa." },
      { key: "dano_peso_trinca", label: "Peso das trincas", min: 0, max: 120, step: 5, fmt: (v) => `${v}`, help: "Peso da densidade de trincas no Índice." },
      { key: "dano_peso_buraco", label: "Peso dos buracos", min: 0, max: 100, step: 5, fmt: (v) => `${v}`, help: "Peso da % de buracos no Índice." },
    ],
  },
];

// Estado global da aplicação
const state = {
  view: "etapas",
  cenarios: [],
  dataset: { disponivel: false, total: 0, destaques: [], estados: [] },
  presets: {},
  metricasOficiais: [],
  cenarioId: "rodovia_dnit_01",
  amostraDnit: null,
  uploadId: null,
  uploadNome: "",
  temRoiUpload: false,
  dnitModal: { q: "", particao: "teste", estado: "todos", offset: 0, limit: 48, totalFiltrado: 0 },
  roiModo: "cenario", // "cenario" | "inteira" | "poligono"
  roiPoligono: [[0.295, 0.0], [0.656, 0.0], [1.0, 0.80], [1.0, 1.0], [0.0, 1.0], [0.0, 0.24]],
  reduzir2x: false,
  params: {},
  step: "5",
  substepByStep: {
    "1": "1_roi",
    "2": "2_media",
    "3a": "3a_histerese",
    "3b": "3b_textura",
    "4": "4_abertura",
    "5": "5_final",
  },
  compareMode: "cortina", // "unico" | "cortina" | "lado"
  curtainPos: 0.5,
  zoom: 1.0,
  panX: 0,
  panY: 0,
  loupe: false,
  inspectorOpen: false,
  techDetailsOpen: false,
  activeExperimentId: null,
  experimentBackup: null,
  openParamGroups: {},
  lastParamStep: null,
  pinnedPixel: null,
  hoverPixel: null,
  highlightBbox: null,
  selectedRegionKey: null,
  histMode: "textura", // "textura" | "intensidade"
  regTab: "trincas",   // "trincas" | "buracos" | "rej_trincas" | "rej_buracos"
  result: null,
  images: {},          // key -> HTMLImageElement
  pack1: null,         // ImageData (R=g, G=media, B=desvio)
  pack2: null,         // ImageData (R=desvio_b, G=flags_t, B=flags_b)
  dragMode: null,      // "maybe_pan" | "curtain" | "roi_pt"
  dragRoiIndex: -1,
  dragStart: null,
  analyzeTimer: null,
  requestSeq: 0,
};

// Elementos DOM principais
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const mainCanvas = $("#mainCanvas");
const mainCtx = mainCanvas.getContext("2d");
const loupeCanvas = $("#loupeCanvas");
const loupeCtx = loupeCanvas.getContext("2d");

// ==================== UTILITÁRIOS ====================
function fmtInt(n) {
  return Number(n || 0).toLocaleString("pt-BR");
}

function escHtml(str) {
  return String(str == null ? "" : str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatarSeveridade(sev) {
  return window.PAVIMENTO_INFO?.SEVERIDADE_LABEL?.[sev] || sev || "—";
}

function atualizarCompareModeUI() {
  $$("#compareModeGroup .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.cmp === state.compareMode)
  );
}

function showToast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 2800);
}

function setLoading(flag) {
  $("#loadingOverlay").classList.toggle("hidden", !flag);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function extractImageData(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height);
}

function getSubstagesParaEtapa(step) {
  const base = STAGE_CONFIG[step].substages;
  if (step === "5" && state.result && state.result.metricas_gt) {
    return [
      ...base,
      { key: "5_gt_comparar", label: "vs. Gabarito DNIT", antes: "5_final", antesLbl: "Nossa Saída", depoisLbl: "Acerto / Erro vs. DNIT" },
      { key: "5_gt_rotulo", label: "Gabarito DNIT", antes: "1_roi", antesLbl: "Pista + ROI", depoisLbl: "Rótulo Real DNIT" },
    ];
  }
  return base;
}

function getCurrentSubstageObj() {
  const subs = getSubstagesParaEtapa(state.step);
  const key = state.substepByStep[state.step] || subs[0].key;
  return subs.find((s) => s.key === key) || subs[0];
}

// ==================== API & ANÁLISE ====================
function montarPayloadAnalise() {
  return {
    cenario_id: state.cenarioId,
    amostra_dnit: state.amostraDnit,
    upload_id: state.uploadId,
    roi_modo: state.roiModo,
    roi_poligono: state.roiPoligono,
    reduzir_2x: state.reduzir2x,
    params: state.params,
  };
}

async function executarAnalise(silencioso = false) {
  const seq = ++state.requestSeq;
  if (!silencioso) setLoading(true);
  try {
    const resp = await fetch("/api/analisar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(montarPayloadAnalise()),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.erro || "Erro ao analisar imagem");
    if (seq !== state.requestSeq) return;

    const entries = Object.entries(data.imagens);
    const decoded = await Promise.all(entries.map(async ([k, src]) => [k, await loadImage(src)]));
    const [p1Img, p2Img] = await Promise.all([
      loadImage(data.data_packs.pack1),
      loadImage(data.data_packs.pack2),
    ]);
    if (seq !== state.requestSeq) return;

    state.images = Object.fromEntries(decoded);
    state.pack1 = extractImageData(p1Img);
    state.pack2 = extractImageData(p2Img);
    state.result = data;

    $("#perfText").textContent = `Núcleo: ${data.tempo_nucleo_ms.toFixed(1)} ms`;
    const [w0, h0] = data.dimensoes.original;
    const [wp, hp] = data.dimensoes.processamento;
    $("#lblResolucao").textContent = data.dimensoes.reduzida_2x
      ? `Redução 2× (${w0}×${h0} → ${wp}×${hp})`
      : `Sem redução (${wp}×${hp})`;

    atualizarResumoMini();
    renderEtapaAtual();
    if (state.view === "mosaico") renderMosaico();
  } catch (err) {
    showToast(`Erro: ${err.message}`);
  } finally {
    if (seq === state.requestSeq) setLoading(false);
  }
}

function agendarAnalise(ms = 90) {
  clearTimeout(state.analyzeTimer);
  state.analyzeTimer = setTimeout(() => executarAnalise(true), ms);
}

// ==================== RENDERIZAÇÃO DO CANVAS ====================
function calcularLayoutCanvas() {
  if (!state.result) return null;
  const stage = $("#canvasStage");
  const sw = stage.clientWidth || 760;
  const sh = stage.clientHeight || 460;
  if (mainCanvas.width !== sw || mainCanvas.height !== sh) {
    mainCanvas.width = sw;
    mainCanvas.height = sh;
  }

  const [iw, ih] = state.result.dimensoes.processamento;
  const pad = 12;

  if (state.compareMode === "lado") {
    const gap = 10;
    const halfW = (sw - pad * 2 - gap) / 2;
    const baseScale = Math.min(halfW / iw, (sh - pad * 2) / ih);
    const scale = baseScale * state.zoom;
    const dw = iw * scale;
    const dh = ih * scale;
    const leftX = (sw - gap) / 2 - dw + state.panX;
    const rightX = (sw + gap) / 2 + state.panX;
    const y = (sh - dh) / 2 + state.panY;
    return { mode: "lado", sw, sh, iw, ih, scale, dw, dh, leftX, rightX, y };
  }

  const baseScale = Math.min((sw - pad * 2) / iw, (sh - pad * 2) / ih);
  const scale = baseScale * state.zoom;
  const dw = iw * scale;
  const dh = ih * scale;
  const x = (sw - dw) / 2 + state.panX;
  const y = (sh - dh) / 2 + state.panY;
  return { mode: state.compareMode, sw, sh, iw, ih, scale, dw, dh, x, y };
}

function desenharCanvas() {
  const lay = calcularLayoutCanvas();
  if (!lay) return;
  const sub = getCurrentSubstageObj();
  const imgMain = state.images[sub.key];
  const imgAntes = state.images[sub.antes] || state.images["1_original"];
  if (!imgMain) return;

  mainCtx.save();
  mainCtx.clearRect(0, 0, lay.sw, lay.sh);
  mainCtx.fillStyle = "#0D1117";
  mainCtx.fillRect(0, 0, lay.sw, lay.sh);
  mainCtx.imageSmoothingEnabled = lay.scale < 1.5;

  const badgeL = $("#cmpBadgeLeft");
  const badgeR = $("#cmpBadgeRight");
  const curtain = $("#curtainHandle");

  if (lay.mode === "lado") {
    curtain.classList.add("hidden");
    badgeL.classList.remove("hidden");
    badgeR.classList.remove("hidden");
    badgeL.textContent = sub.antesLbl;
    badgeR.textContent = sub.depoisLbl;

    mainCtx.drawImage(imgAntes, lay.leftX, lay.y, lay.dw, lay.dh);
    mainCtx.drawImage(imgMain, lay.rightX, lay.y, lay.dw, lay.dh);

    mainCtx.strokeStyle = "rgba(255,255,255,0.18)";
    mainCtx.lineWidth = 1;
    mainCtx.strokeRect(lay.leftX, lay.y, lay.dw, lay.dh);
    mainCtx.strokeRect(lay.rightX, lay.y, lay.dw, lay.dh);

    desenharOverlaysSobreImagem(lay.rightX, lay.y, lay.scale);
    desenharOverlaysSobreImagem(lay.leftX, lay.y, lay.scale, true);
  } else if (lay.mode === "cortina") {
    const splitX = lay.x + lay.dw * state.curtainPos;
    curtain.classList.remove("hidden");
    curtain.style.left = `${Math.max(14, Math.min(lay.sw - 14, splitX))}px`;
    badgeL.classList.remove("hidden");
    badgeR.classList.remove("hidden");
    badgeL.textContent = `← ${sub.antesLbl}`;
    badgeR.textContent = `${sub.depoisLbl} →`;

    mainCtx.save();
    mainCtx.beginPath();
    mainCtx.rect(lay.x, lay.y, Math.max(0, splitX - lay.x), lay.dh);
    mainCtx.clip();
    mainCtx.drawImage(imgAntes, lay.x, lay.y, lay.dw, lay.dh);
    mainCtx.restore();

    mainCtx.save();
    mainCtx.beginPath();
    mainCtx.rect(splitX, lay.y, Math.max(0, lay.x + lay.dw - splitX), lay.dh);
    mainCtx.clip();
    mainCtx.drawImage(imgMain, lay.x, lay.y, lay.dw, lay.dh);
    mainCtx.restore();

    desenharOverlaysSobreImagem(lay.x, lay.y, lay.scale);
  } else {
    curtain.classList.add("hidden");
    badgeL.classList.add("hidden");
    badgeR.classList.add("hidden");
    mainCtx.drawImage(imgMain, lay.x, lay.y, lay.dw, lay.dh);
    desenharOverlaysSobreImagem(lay.x, lay.y, lay.scale);
  }

  mainCtx.restore();
}

function desenharOverlaysSobreImagem(ox, oy, scale, somenteCrosshair = false) {
  const [iw, ih] = state.result.dimensoes.processamento;

  if (!somenteCrosshair && state.highlightBbox) {
    const [bx, by, bw, bh] = state.highlightBbox;
    mainCtx.save();
    mainCtx.strokeStyle = "#F59E0B";
    mainCtx.lineWidth = 2.5;
    mainCtx.setLineDash([5, 3]);
    mainCtx.strokeRect(ox + bx * scale - 2, oy + by * scale - 2, bw * scale + 4, bh * scale + 4);
    mainCtx.fillStyle = "rgba(245, 158, 11, 0.16)";
    mainCtx.fillRect(ox + bx * scale, oy + by * scale, bw * scale, bh * scale);
    mainCtx.restore();
  }

  if (!somenteCrosshair && state.roiModo === "poligono") {
    const pts = state.roiPoligono.map(([px, py]) => [ox + px * (iw - 1) * scale, oy + py * (ih - 1) * scale]);
    mainCtx.save();
    mainCtx.beginPath();
    pts.forEach(([x, y], idx) => (idx === 0 ? mainCtx.moveTo(x, y) : mainCtx.lineTo(x, y)));
    mainCtx.closePath();
    mainCtx.fillStyle = "rgba(245, 158, 11, 0.14)";
    mainCtx.fill();
    mainCtx.strokeStyle = "#F59E0B";
    mainCtx.lineWidth = 2;
    mainCtx.stroke();
    pts.forEach(([x, y], idx) => {
      mainCtx.beginPath();
      mainCtx.arc(x, y, 7, 0, Math.PI * 2);
      mainCtx.fillStyle = "#F59E0B";
      mainCtx.fill();
      mainCtx.lineWidth = 2;
      mainCtx.strokeStyle = "#161B22";
      mainCtx.stroke();
      mainCtx.fillStyle = "#161B22";
      mainCtx.font = "bold 10px sans-serif";
      mainCtx.textAlign = "center";
      mainCtx.textBaseline = "middle";
      mainCtx.fillText(String(idx + 1), x, y);
    });
    mainCtx.restore();
  }

  const pt = state.pinnedPixel || state.hoverPixel;
  if (pt) {
    const cx = ox + (pt.x + 0.5) * scale;
    const cy = oy + (pt.y + 0.5) * scale;
    mainCtx.save();
    mainCtx.strokeStyle = state.pinnedPixel ? "#F59E0B" : "rgba(56, 189, 248, 0.85)";
    mainCtx.lineWidth = state.pinnedPixel ? 2 : 1;
    mainCtx.beginPath();
    mainCtx.moveTo(cx - 9, cy);
    mainCtx.lineTo(cx + 9, cy);
    mainCtx.moveTo(cx, cy - 9);
    mainCtx.lineTo(cx, cy + 9);
    mainCtx.stroke();
    if (state.pinnedPixel) {
      mainCtx.beginPath();
      mainCtx.arc(cx, cy, 5, 0, Math.PI * 2);
      mainCtx.stroke();
    }
    mainCtx.restore();
  }
}

function canvasParaPixel(clientX, clientY) {
  const lay = calcularLayoutCanvas();
  if (!lay) return null;
  const rect = mainCanvas.getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;

  let ox = lay.x;
  const oy = lay.y;
  if (lay.mode === "lado") {
    if (mx >= lay.leftX && mx <= lay.leftX + lay.dw) ox = lay.leftX;
    else if (mx >= lay.rightX && mx <= lay.rightX + lay.dw) ox = lay.rightX;
    else return null;
  }

  const px = Math.floor((mx - ox) / lay.scale);
  const py = Math.floor((my - oy) / lay.scale);
  if (px < 0 || py < 0 || px >= lay.iw || py >= lay.ih) return null;
  return { x: px, y: py, mx, my };
}

function desenharLupa(pt) {
  if (!state.loupe || !pt || !state.result) {
    loupeCanvas.classList.add("hidden");
    return;
  }
  const sub = getCurrentSubstageObj();
  const img = state.images[sub.key];
  if (!img) return;

  loupeCanvas.classList.remove("hidden");
  const stage = $("#canvasStage");
  const lw = 132;
  let lx = pt.mx + 20;
  let ly = pt.my - lw - 16;
  if (lx + lw > stage.clientWidth - 6) lx = pt.mx - lw - 20;
  if (ly < 6) ly = pt.my + 20;
  loupeCanvas.style.left = `${lx}px`;
  loupeCanvas.style.top = `${ly}px`;

  const raioPx = 11;
  loupeCtx.imageSmoothingEnabled = false;
  loupeCtx.fillStyle = "#0D1117";
  loupeCtx.fillRect(0, 0, lw, lw);
  loupeCtx.drawImage(
    img,
    pt.x - raioPx, pt.y - raioPx, raioPx * 2, raioPx * 2,
    0, 0, lw, lw
  );
  const cell = lw / (raioPx * 2);
  loupeCtx.strokeStyle = "#F59E0B";
  loupeCtx.lineWidth = 2;
  loupeCtx.strokeRect(raioPx * cell, raioPx * cell, cell, cell);
}

// ==================== INSPETOR DE PIXEL (60 FPS) ====================
function atualizarInspetorPixel(pt) {
  if (!state.inspectorOpen) return;

  const box = $("#pixelInspector");
  box.classList.toggle("pinned", Boolean(state.pinnedPixel));
  $("#piHint").textContent = state.pinnedPixel
    ? `Pixel (${state.pinnedPixel.x}, ${state.pinnedPixel.y}) FIXADO · Clique novamente na imagem para soltar`
    : "Passe o mouse sobre a imagem · Clique para fixar um pixel";

  if (!pt || !state.pack1 || !state.pack2 || !state.result) {
    $("#piCoord").textContent = "—";
    $("#piG").textContent = "—";
    $("#piM").textContent = "—";
    $("#piS").textContent = "—";
    $("#piTrincaTest").textContent = "—";
    $("#piBuracoTest").textContent = "—";
    $("#piEstado").textContent = "—";
    return;
  }

  const w = state.pack1.width;
  const idx = (pt.y * w + pt.x) * 4;
  const g = state.pack1.data[idx];
  const m = state.pack1.data[idx + 1];
  const s = state.pack1.data[idx + 2];

  const sb = state.pack2.data[idx];
  const ft = state.pack2.data[idx + 1];
  const fb = state.pack2.data[idx + 2];

  const inRoi = Boolean(ft & 1);
  const tForte = Boolean(ft & 2);
  const tFracaBruta = Boolean(ft & 4);
  const tFraca = Boolean(ft & 8);
  const tHist = Boolean(ft & 16);
  const tFinal = Boolean(ft & 32);
  const tEsq = Boolean(ft & 64);
  const tBif = Boolean(ft & 128);

  const bCand = Boolean(fb & 1);
  const bAberto = Boolean(fb & 2);
  const bPreench = Boolean(fb & 8);
  const bFinal = Boolean(fb & 16);

  const p = state.result.params;
  const est = state.result.estatisticas;

  $("#piCoord").innerHTML = `(${pt.x}, ${pt.y}) ${inRoi ? '<span class="tag-ok">[ROI]</span>' : '<span class="tag-danger">[Fora]</span>'}`;
  $("#piG").textContent = `${g} / 255`;
  $("#piM").textContent = `${m}`;
  $("#piS").textContent = `s=${s} · s_b=${sb}`;

  const diff = m - g;
  const limForte = ((p.trinca_a16 * s + p.trinca_b16) / 16).toFixed(1);
  const limFraca = ((Math.min(p.trinca_a16, p.trinca_fraca_a16) * s + Math.min(p.trinca_b16, p.trinca_fraca_b16)) / 16).toFixed(1);
  let trincaTag = '<span class="muted">Fundo</span>';
  if (!inRoi) trincaTag = '<span class="muted">Fora da ROI</span>';
  else if (tForte) trincaTag = '<span class="tag-warn">Semente Forte ✓</span>';
  else if (tFracaBruta) trincaTag = '<span class="tag-ok">Cand. Fraco ✓</span>';
  else if (tFraca) trincaTag = '<span class="tag-ok">Fecho Dir. ✓</span>';
  $("#piTrincaTest").innerHTML = `m−I=${diff} (forte>${limForte}, fraca>${limFraca}) → ${trincaTag}`;

  let burTag = '<span class="muted">Normal</span>';
  if (!inRoi) burTag = '<span class="muted">Fora da ROI</span>';
  else if (bFinal) burTag = '<span class="tag-warn">Buraco Aceito ✓</span>';
  else if (bPreench) burTag = '<span class="tag-danger">Rejeitado (forma)</span>';
  else if (bCand && !bAberto) burTag = '<span class="tag-danger">Removido p/ Abertura</span>';
  else if (bCand) burTag = '<span class="tag-ok">Cand. Textura ✓</span>';
  $("#piBuracoTest").innerHTML = `s_b=${sb} vs t_tex=${est.limiar_textura} → ${burTag}`;

  const tags = [];
  if (tFinal) tags.push("TRINCA");
  if (tEsq) tags.push("ESQUELETO");
  if (tBif) tags.push("BIFURCAÇÃO");
  if (bFinal) tags.push("BURACO");
  if (!tags.length && tHist) tags.push("Trinca rejeitada (forma)");
  if (!tags.length && tFraca && !tHist) tags.push("Ruído descartado (histerese)");
  $("#piEstado").innerHTML = tags.length
    ? `<span class="${bFinal || tFinal ? "tag-warn" : "muted"}">${tags.join(" · ")}</span>`
    : '<span class="muted">Asfalto normal</span>';
}

// ==================== RESUMO & RENDERIZAÇÃO DAS ETAPAS ====================
function atualizarResumoMini() {
  if (!state.result) return;
  const q = state.result.quant;
  const badge = $("#smSevBadge");
  badge.textContent = formatarSeveridade(q.severidade);
  badge.className = `sev-badge sev-${q.severidade}`;

  const pct = Math.min(100, Math.max(0, q.indice_dano_1000 / 10));
  const marker = $("#smGaugeMarker");
  if (marker) marker.style.left = `${pct}%`;

  $("#smIndice").textContent = `${(q.indice_dano_1000 / 10).toFixed(1)}`;
  $("#smTrincas").textContent = `${q.n_trincas} (${fmtInt(q.comprimento_trincas_px)} px)`;
  $("#smBuracos").textContent = `${q.n_buracos} (${(q.area_buracos_pct_100 / 100).toFixed(2)}%)`;

  const gtStrip = $("#gtLiveStrip");
  if (!gtStrip) return;
  const gt = state.result.metricas_gt;
  if (gt) {
    gtStrip.classList.remove("hidden");
    gtStrip.innerHTML = `
      <span>vs. Gabarito DNIT: Trinca tol.2px F1 = <strong>${gt.trinca_tol2.f1.toFixed(3)}</strong> · Buraco IoU = <strong>${gt.buraco.iou.toFixed(3)}</strong></span>
      <button type="button" class="btn btn-outline btn-xs" id="btnAbrirGtComparar">Ver mapa GT</button>
    `;
    $("#btnAbrirGtComparar")?.addEventListener("click", () => {
      state.step = "5";
      state.substepByStep["5"] = "5_gt_comparar";
      state.highlightBbox = null;
      renderEtapaAtual();
    });
  } else {
    gtStrip.classList.add("hidden");
    gtStrip.innerHTML = "";
  }
}

function renderEtapaAtual() {
  const cfg = STAGE_CONFIG[state.step];
  const sub = getCurrentSubstageObj();

  $$(".step-pill").forEach((b) => b.classList.toggle("active", b.dataset.step === state.step));

  const eyebrow = $("#stageEyebrow");
  eyebrow.textContent = cfg.eyebrow;
  eyebrow.className = `stage-eyebrow ${cfg.accent === "amber" ? "amber" : ""}`;
  $("#stageTitle").textContent = cfg.title;

  const idx = STEP_ORDER.indexOf(state.step);
  $("#btnPrevStep").disabled = idx <= 0;
  $("#btnNextStep").disabled = idx >= STEP_ORDER.length - 1;

  const subs = getSubstagesParaEtapa(state.step);
  const pillsBox = $("#substagePills");
  pillsBox.innerHTML = subs
    .map(
      (s) =>
        `<button class="sub-pill ${s.key === sub.key ? "active" : ""}" data-sub="${escHtml(s.key)}">${escHtml(s.label)}</button>`
    )
    .join("");
  pillsBox.querySelectorAll(".sub-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.substepByStep[state.step] = btn.dataset.sub;
      renderEtapaAtual();
    });
  });

  const legendItems = window.PAVIMENTO_INFO?.LEGENDAS_POR_SUBETAPA?.[sub.key] || cfg.legend;
  $("#legendStrip").innerHTML = legendItems
    .map(
      (it) =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${escHtml(it.color)}"></span>${escHtml(it.label)}</span>`
    )
    .join("");

  $("#roiEditBanner").classList.toggle("hidden", state.roiModo !== "poligono");

  renderExplicacaoLateral(cfg);
  renderParametrosLaterais();
  renderPainelContextual();
  desenharCanvas();
  atualizarInspetorPixel(state.pinnedPixel || state.hoverPixel);
}

function encontrarProvaPorId(expId) {
  if (!expId) return null;
  const etapas = window.PAVIMENTO_INFO?.ETAPAS || {};
  for (const stepKey of Object.keys(etapas)) {
    const lista = etapas[stepKey]?.provaRapida || [];
    const achado = lista.find((e) => e.id === expId);
    if (achado) return { step: stepKey, exp: achado };
  }
  return null;
}

function restaurarExperimento() {
  const b = state.experimentBackup;
  if (!b) return;
  state.params = { ...b.params };
  state.roiModo = b.roiModo;
  state.reduzir2x = b.reduzir2x;
  if (b.step && b.substep) {
    state.substepByStep[b.step] = b.substep;
  }
  state.compareMode = b.compareMode;
  state.experimentBackup = null;
  state.activeExperimentId = null;
  state.pinnedPixel = null;

  const chk = $("#chkReduzir2x");
  if (chk) chk.checked = state.reduzir2x;
  atualizarBotoesRoi();
  atualizarCompareModeUI();
  executarAnalise();
}

function aplicarExperimento(exp) {
  if (!exp) return;
  if (state.activeExperimentId === exp.id) {
    restaurarExperimento();
    return;
  }

  if (!state.experimentBackup) {
    state.experimentBackup = {
      params: { ...state.params },
      roiModo: state.roiModo,
      reduzir2x: state.reduzir2x,
      step: state.step,
      substep: state.substepByStep[state.step],
      compareMode: state.compareMode,
    };
  } else {
    const b = state.experimentBackup;
    state.params = { ...b.params };
    state.roiModo = b.roiModo;
    state.reduzir2x = b.reduzir2x;
    if (b.step === state.step && b.substep) {
      state.substepByStep[state.step] = b.substep;
    }
    state.compareMode = b.compareMode;
    const chk = $("#chkReduzir2x");
    if (chk) chk.checked = state.reduzir2x;
    atualizarBotoesRoi();
    atualizarCompareModeUI();
  }

  state.activeExperimentId = exp.id;
  state.pinnedPixel = null;

  if (exp.tipo === "roi") {
    state.roiModo = exp.payload;
    atualizarBotoesRoi();
  } else if (exp.tipo === "reducao") {
    state.reduzir2x = Boolean(exp.payload);
    const chk = $("#chkReduzir2x");
    if (chk) chk.checked = state.reduzir2x;
  } else if (exp.tipo === "params" && exp.payload) {
    Object.assign(state.params, exp.payload);
  } else if (exp.tipo === "substage" && exp.payload) {
    if (exp.payload.substep) state.substepByStep[state.step] = exp.payload.substep;
    if (exp.payload.compareMode) {
      state.compareMode = exp.payload.compareMode;
      atualizarCompareModeUI();
    }
  }

  if (exp.substep) state.substepByStep[state.step] = exp.substep;
  if (exp.compareMode) {
    state.compareMode = exp.compareMode;
    atualizarCompareModeUI();
  }

  if (exp.tipo === "substage") {
    renderEtapaAtual();
  } else {
    executarAnalise();
  }
}

function getMetricasEtapa(est, dim, q) {
  if (state.step === "1") {
    return [
      ["Original", `${dim.original[0]}×${dim.original[1]} px`],
      ["Processada", `${dim.processamento[0]}×${dim.processamento[1]} px`],
      ["Área ROI", `${fmtInt(dim.pixels_roi)} (${dim.roi_pct}%)`],
      ["Brilho médio", `${est.media_roi} / 255`],
    ];
  }
  if (state.step === "2") {
    return [
      ["Janela trincas", `${est.janela_trinca}×${est.janela_trinca} px`],
      ["Janela buracos", `${est.janela_buraco}×${est.janela_buraco} px`],
      ["Mediana brilho", `${est.mediana_g} (MAD ${est.mad_g})`],
      ["Tempo núcleo", `${state.result.tempo_nucleo_ms.toFixed(1)} ms`],
    ];
  }
  if (state.step === "3a") {
    return [
      ["Sementes fortes", `${fmtInt(est.px_trinca_forte)} px`],
      ["Cand. fracos", `${fmtInt(est.px_trinca_fraca_bruta)} px`],
      ["Pós-histerese", `${fmtInt(est.px_trinca_pos_histerese)} px`],
      ["Ruído removido", `${fmtInt(est.px_trinca_descartados_histerese)} px`],
    ];
  }
  if (state.step === "3b") {
    return [
      ["Mediana s_b", `${est.mediana_textura}`],
      ["MAD s_b", `${est.mad_textura}`],
      ["Corte t_tex", `${est.limiar_textura}`],
      ["Candidatos", `${fmtInt(est.px_buraco_cand)} px`],
    ];
  }
  if (state.step === "4") {
    return [
      ["Abertura", `−${fmtInt(est.px_buraco_removido_abertura)} px`],
      ["Preenchimento", `+${fmtInt(est.px_buraco_adicionado_preench)} px`],
      ["Fecho dir.", `+${fmtInt(est.px_trinca_fecho_adicionados)} px`],
      ["Buracos limpos", `${fmtInt(est.px_buraco_pos_preench)} px`],
    ];
  }
  return [
    ["Trincas (ac./rej.)", `${est.n_comp_trinca_aceitos} / ${est.n_comp_trinca_rejeitados}`],
    ["Esqueleto / bif.", `${fmtInt(est.px_esqueleto)} px / ${fmtInt(est.px_bifurcacoes)}`],
    ["Buracos (ac./rej.)", `${est.n_comp_buraco_aceitos} / ${est.n_comp_buraco_rejeitados}`],
    ["Índice / Sev.", `${(q.indice_dano_1000 / 10).toFixed(1)}/100 (${formatarSeveridade(q.severidade)})`],
  ];
}

function renderExplicacaoLateral(cfg) {
  if (!state.result) return;
  const est = state.result.estatisticas;
  const dim = state.result.dimensoes;
  const q = state.result.quant;
  const sub = getCurrentSubstageObj();
  const info = window.PAVIMENTO_INFO?.ETAPAS?.[state.step];

  const metricas = getMetricasEtapa(est, dim, q);
  const dicaSub = info?.dicaVisualPorSub?.[sub.key] || "";
  const provas = info?.provaRapida || [];
  const activeInfo = encontrarProvaPorId(state.activeExperimentId);

  const box = $("#explicacaoConteudo");
  box.innerHTML = `
    ${info?.requisitoTag ? `<span class="req-pill-tag">${escHtml(info.requisitoTag)}</span>` : ""}

    <p class="stage-summary-txt">${escHtml(info?.resumoCurto || cfg.oQueFaz)}</p>
    ${dicaSub ? `<div class="edu-block"><p><strong>Nesta visão:</strong> ${escHtml(dicaSub)}</p></div>` : ""}

    <div class="live-metrics-list">
      ${metricas.map(([k, v]) => `<div class="lm-item"><span class="lm-k">${escHtml(k)}</span><span class="lm-v">${escHtml(v)}</span></div>`).join("")}
    </div>

    ${(provas.length || activeInfo) ? `
      <div class="quick-proofs">
        <h4 class="quick-proofs-title">Provas rápidas</h4>
        ${provas.map((e) => `<button type="button" class="proof-btn ${state.activeExperimentId === e.id ? "active" : ""}" data-exp-id="${escHtml(e.id)}">${escHtml(e.label)}</button>`).join("")}
        ${state.activeExperimentId ? `<button type="button" class="proof-reset-btn" id="btnResetExperimento">↺ Voltar ao normal</button>` : ""}
      </div>
    ` : ""}

    <details class="tech-details" id="techDetailsBox" ${state.techDetailsOpen ? "open" : ""}>
      <summary>Fórmula inteira e referência da disciplina</summary>
      <div class="tech-details-body">
        <pre class="formula-box">${escHtml(cfg.formula)}</pre>
        <div class="formula-note">${escHtml(cfg.formulaNote)}</div>
        <div class="ref-box" style="margin-top:8px;">
          <strong>Referência:</strong> ${escHtml(cfg.ref)}
        </div>
      </div>
    </details>
  `;

  box.querySelectorAll(".proof-btn[data-exp-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const expObj = provas.find((e) => e.id === btn.dataset.expId);
      aplicarExperimento(expObj);
    });
  });

  $("#btnResetExperimento")?.addEventListener("click", restaurarExperimento);

  const detailsEl = $("#techDetailsBox");
  if (detailsEl) {
    detailsEl.addEventListener("toggle", () => {
      state.techDetailsOpen = detailsEl.open;
    });
  }
}

function renderParametrosLaterais() {
  const container = $("#parametrosConteudo");
  const currentGroup = STAGE_CONFIG[state.step].paramGroup;

  if (Object.keys(state.openParamGroups).length === 0) {
    PARAM_GROUPS.forEach((g) => {
      state.openParamGroups[g.id] = g.id === currentGroup;
    });
    state.lastParamStep = state.step;
  } else if (state.lastParamStep !== state.step) {
    if (currentGroup && currentGroup !== "etapa1") {
      state.openParamGroups[currentGroup] = true;
    }
    state.lastParamStep = state.step;
  }

  const ordenados = [...PARAM_GROUPS].sort((a, b) =>
    a.id === currentGroup ? -1 : b.id === currentGroup ? 1 : 0
  );

  const stage1Helper = currentGroup === "etapa1"
    ? `
      <div class="stage1-param-helper">
        <h4>Etapa 1 · Controles da Entrada (sem sliders)</h4>
        <p>O pré-processamento depende da <strong>Região de Interesse (ROI)</strong> e da <strong>Redução 2×</strong>:</p>
        <div class="inline-row" style="margin-top:8px;">
          <button type="button" class="btn btn-xs ${state.roiModo === "cenario" ? "btn-primary" : "btn-outline"}" id="btnQuickRoiCenario">Pista (ROI)</button>
          <button type="button" class="btn btn-xs ${state.roiModo === "inteira" ? "btn-primary" : "btn-outline"}" id="btnQuickRoiInteira">Imagem Inteira</button>
          <button type="button" class="btn btn-xs ${state.reduzir2x ? "btn-primary" : "btn-outline"}" id="btnQuickReduzir2x">${state.reduzir2x ? "Redução 2×: Ligada ✓" : "Redução 2×: Desligada"}</button>
        </div>
      </div>
    `
    : "";

  const tradMap = window.PAVIMENTO_INFO?.PARAM_TRADUCAO || {};

  container.innerHTML = stage1Helper + ordenados
    .map((grp) => {
      const ehAtual = grp.id === currentGroup;
      const isOpen = Boolean(state.openParamGroups[grp.id]);
      return `
        <div class="param-group ${ehAtual ? "current-stage" : ""} ${isOpen ? "" : "collapsed"}" data-grp="${escHtml(grp.id)}">
          <div class="param-group-head">
            <span>${ehAtual ? "★ " : ""}${escHtml(grp.title)}</span>
            <span>${isOpen ? "▾" : "▸"}</span>
          </div>
          <div class="param-group-body">
            ${grp.items
              .map((it) => {
                const val = state.params[it.key] ?? it.min;
                const trad = tradMap[it.key];
                const nomeHtml = trad
                  ? `${escHtml(trad.nomeSimples)} <small style="font-weight:500;color:var(--ink-soft);">(${escHtml(it.label)})</small>`
                  : escHtml(it.label);
                const helpText = trad
                  ? `← ${trad.esquerda} | ${trad.direita} →`
                  : it.help;
                return `
                  <div class="param-row">
                    <div class="param-row-top">
                      <span class="param-name">${nomeHtml}</span>
                      <span class="param-val" id="pval_${escHtml(it.key)}">${escHtml(it.fmt(val))}</span>
                    </div>
                    <input type="range" data-param="${escHtml(it.key)}" min="${it.min}" max="${it.max}" step="${it.step}" value="${val}">
                    <span class="param-help">${escHtml(helpText)}</span>
                  </div>
                `;
              })
              .join("")}
          </div>
        </div>
      `;
    })
    .join("");

  $("#btnQuickRoiCenario")?.addEventListener("click", () => {
    state.roiModo = "cenario";
    atualizarBotoesRoi();
    executarAnalise();
  });
  $("#btnQuickRoiInteira")?.addEventListener("click", () => {
    state.roiModo = "inteira";
    atualizarBotoesRoi();
    executarAnalise();
  });
  $("#btnQuickReduzir2x")?.addEventListener("click", () => {
    state.reduzir2x = !state.reduzir2x;
    state.pinnedPixel = null;
    const chk = $("#chkReduzir2x");
    if (chk) chk.checked = state.reduzir2x;
    executarAnalise();
  });

  container.querySelectorAll(".param-group-head").forEach((head) => {
    head.addEventListener("click", () => {
      const grp = head.parentElement;
      const isCollapsed = grp.classList.toggle("collapsed");
      state.openParamGroups[grp.dataset.grp] = !isCollapsed;
      head.lastElementChild.textContent = isCollapsed ? "▸" : "▾";
    });
  });

  container.querySelectorAll('input[type="range"]').forEach((inp) => {
    const key = inp.dataset.param;
    const item = PARAM_GROUPS.flatMap((g) => g.items).find((x) => x.key === key);
    inp.addEventListener("input", () => {
      const v = Number(inp.value);
      state.params[key] = v;
      const lbl = $(`#pval_${key}`);
      if (lbl && item) lbl.textContent = item.fmt(v);
      agendarAnalise(75);
    });
  });
}

// ==================== PAINEL CONTEXTUAL (ETAPAS 3B, 4 E 5) ====================
function renderPainelContextual() {
  const card = $("#contextualCard");
  if (!state.result || ["1", "2", "3a"].includes(state.step)) {
    card.classList.add("hidden");
    card.innerHTML = "";
    return;
  }

  const est = state.result.estatisticas;
  const p = state.result.params;
  card.classList.remove("hidden");

  if (state.step === "3b") {
    card.innerHTML = `
      <div class="inline-row" style="justify-content:space-between; margin-bottom:8px;">
        <div>
          <div class="stage-eyebrow amber">HISTOGRAMA NA ROI</div>
          <h3 style="margin:2px 0 0">Limiar Robusto de Textura (Mediana + k·MAD)</h3>
        </div>
        <div class="segmented segmented-sm" id="histModeGroup">
          <button class="seg-btn ${state.histMode === "textura" ? "active" : ""}" data-hmode="textura">Textura (s_b)</button>
          <button class="seg-btn ${state.histMode === "intensidade" ? "active" : ""}" data-hmode="intensidade">Intensidade (I)</button>
        </div>
      </div>
      <div class="hist-wrap" id="histWrap"></div>
      <p class="muted" style="margin-top:8px">
        ${
          state.histMode === "textura"
            ? `Na ROI atual: <strong>Mediana = ${est.mediana_textura}</strong>, <strong>MAD = ${est.mad_textura}</strong>, <strong>k = ${(p.buraco_k4 / 4).toFixed(2)}</strong> → <strong>t_tex = ${est.limiar_textura}</strong>.`
            : `Histograma de intensidade I na ROI: <strong>Mediana = ${est.mediana_g}</strong> e <strong>Otsu inteiro = ${est.limiar_otsu}</strong>.`
        }
      </p>
      <div class="hist-guide">
        <div><strong>Abaixo do corte:</strong> ${
          state.histMode === "textura"
            ? "Asfalto uniforme dentro da aspereza típica da pista (ignorado para buracos)."
            : "Pixels escuros da via (trincas e sombras à esquerda do corte)."
        }</div>
        <div><strong>Acima do corte (${state.histMode === "textura" ? `t_tex = ${est.limiar_textura}` : `Otsu = ${est.limiar_otsu}`}):</strong> ${
          state.histMode === "textura"
            ? "Regiões ásperas candidatas a buraco (e faixas antes da Abertura)."
            : "Asfalto claro e sinalização pintada."
        }</div>
      </div>
    `;
    card.querySelectorAll("[data-hmode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.histMode = btn.dataset.hmode;
        renderPainelContextual();
      });
    });
    renderHistogramaSVG();
    return;
  }

  if (state.step === "4") {
    const ees = [
      { nome: "Horizontal (—)", bits: [0, 0, 0, 1, 2, 1, 0, 0, 0] },
      { nome: "Vertical (|)", bits: [0, 1, 0, 0, 2, 0, 0, 1, 0] },
      { nome: "Diagonal (\\)", bits: [1, 0, 0, 0, 2, 0, 0, 0, 1] },
      { nome: "Antidiagonal (/)", bits: [0, 0, 1, 0, 2, 0, 1, 0, 0] },
      { nome: "Quadrado (Buracos)", bits: [1, 1, 1, 1, 2, 1, 1, 1, 1] },
    ];
    card.innerHTML = `
      <div class="stage-eyebrow amber">ELEMENTOS ESTRUTURANTES 3×3 (NUMPY DO ZERO)</div>
      <h3>Moldes Direcionais de Trincas e EE Quadrado de Buracos</h3>
      <div class="ee-grid">
        ${ees
          .map(
            (e) => `
          <div class="ee-card">
            <strong>${escHtml(e.nome)}</strong>
            <div class="ee-matrix">
              ${e.bits.map((b) => `<span class="ee-cell ${b === 2 ? "center" : b === 1 ? "on" : ""}"></span>`).join("")}
            </div>
          </div>
        `
          )
          .join("")}
      </div>
      <div class="ee-legend">
        <span class="ee-legend-item"><span class="ee-cell center"></span> Centro do EE (pixel avaliado)</span>
        <span class="ee-legend-item"><span class="ee-cell on"></span> Vizinho ativo</span>
        <span class="ee-legend-item"><span class="ee-cell"></span> Posição inativa</span>
      </div>
    `;
    return;
  }

  if (state.step === "5") {
    card.innerHTML = `
      <div class="stage-eyebrow">COMPONENTES ROTULADOS E FILTROS GEOMÉTRICOS</div>
      <h3>Regiões Detectadas e Rejeitadas (Clique para Destacar no Canvas)</h3>
      <div class="reg-tabs" id="regTabs">
        <button class="sub-pill ${state.regTab === "trincas" ? "active" : ""}" data-rtab="trincas">Trincas Aceitas (${state.result.regioes_trinca.length})</button>
        <button class="sub-pill ${state.regTab === "buracos" ? "active" : ""}" data-rtab="buracos">Buracos Aceitos (${state.result.regioes_buraco.length})</button>
        <button class="sub-pill ${state.regTab === "rej_trincas" ? "active" : ""}" data-rtab="rej_trincas">Trincas Rejeitadas (${state.result.rejeitados_trinca.length})</button>
        <button class="sub-pill ${state.regTab === "rej_buracos" ? "active" : ""}" data-rtab="rej_buracos">Buracos Rejeitados (${state.result.rejeitados_buraco.length})</button>
      </div>
      <p class="muted" style="font-size:12px; margin-bottom:6px">Passe o mouse ou clique em uma linha para destacar a caixa delimitadora na imagem.</p>
      <div class="table-wrap" id="regTableWrap"></div>
    `;

    card.querySelectorAll("[data-rtab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.regTab = btn.dataset.rtab;
        state.highlightBbox = null;
        state.selectedRegionKey = null;
        renderPainelContextual();
        desenharCanvas();
      });
    });

    renderTabelaRegioes();
  }
}

function renderHistogramaSVG() {
  const wrap = $("#histWrap");
  if (!wrap || !state.result) return;
  const est = state.result.estatisticas;
  const isTex = state.histMode === "textura";
  const bins = isTex ? state.result.histogramas.textura : state.result.histogramas.intensidade;
  const n = bins.length;
  const maxVal = Math.max(1, ...bins);

  const W = 680;
  const H = 170;
  const padL = 36;
  const padR = 14;
  const padT = 22;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const barW = plotW / n;

  const xOf = (val) => padL + (Math.min(n - 1, Math.max(0, val)) + 0.5) * barW;

  const bars = bins
    .map((v, i) => {
      if (v === 0) return "";
      const h = (Math.sqrt(v) / Math.sqrt(maxVal)) * plotH;
      const fill = isTex
        ? (i > est.limiar_textura ? "#F59E0B" : "#0F4C5C")
        : (i <= est.limiar_otsu ? "#B91C1C" : "#64748B");
      return `<rect x="${(padL + i * barW).toFixed(1)}" y="${(padT + plotH - h).toFixed(1)}" width="${Math.max(1, barW - 0.5).toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}" opacity="0.88"><title>Valor ${i}: ${fmtInt(v)} pixels</title></rect>`;
    })
    .join("");

  let markers = "";
  if (isTex) {
    const xMed = xOf(est.mediana_textura);
    const xMad = xOf(est.mediana_textura + est.mad_textura);
    const xTex = xOf(est.limiar_textura);
    const xMadLbl = Math.min(W - 62, xMad + 4);
    markers = `
      <line x1="${xMed}" y1="${padT}" x2="${xMed}" y2="${padT + plotH}" stroke="#0F4C5C" stroke-width="2" stroke-dasharray="4 2"/>
      <text x="${xMed + 4}" y="${padT + 11}" fill="#0F4C5C" font-size="10.5" font-weight="700">Med=${est.mediana_textura}</text>
      <line x1="${xMad}" y1="${padT + 14}" x2="${xMad}" y2="${padT + plotH}" stroke="#64748B" stroke-width="1.5" stroke-dasharray="2 2"/>
      <text x="${xMadLbl}" y="${padT + 26}" fill="#475569" font-size="10" font-weight="700">Med+MAD</text>
      <line x1="${xTex}" y1="${padT - 6}" x2="${xTex}" y2="${padT + plotH}" stroke="#B45309" stroke-width="2.5"/>
      <text x="${Math.min(W - 92, xTex + 4)}" y="${padT + 2}" fill="#B45309" font-size="11" font-weight="800">t_tex = ${est.limiar_textura}</text>
    `;
  } else {
    const xMed = xOf(est.mediana_g);
    const xOtsu = xOf(est.limiar_otsu);
    markers = `
      <line x1="${xMed}" y1="${padT}" x2="${xMed}" y2="${padT + plotH}" stroke="#0F4C5C" stroke-width="2" stroke-dasharray="4 2"/>
      <text x="${xMed + 4}" y="${padT + 11}" fill="#0F4C5C" font-size="10.5" font-weight="700">Mediana=${est.mediana_g}</text>
      <line x1="${xOtsu}" y1="${padT - 4}" x2="${xOtsu}" y2="${padT + plotH}" stroke="#B91C1C" stroke-width="2.5"/>
      <text x="${Math.max(padL + 2, xOtsu - 66)}" y="${padT + 2}" fill="#B91C1C" font-size="11" font-weight="800">Otsu=${est.limiar_otsu}</text>
    `;
  }

  const stepTick = isTex ? Math.max(5, Math.round(n / 8)) : 32;
  let ticks = "";
  for (let i = 0; i < n; i += stepTick) {
    const x = padL + i * barW;
    ticks += `<text x="${x}" y="${H - 6}" fill="#64748B" font-size="10" text-anchor="middle">${i}</text>`;
  }

  wrap.innerHTML = `
    <svg class="hist-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#CBD5E1" stroke-width="1"/>
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#CBD5E1" stroke-width="1"/>
      ${bars}
      ${markers}
      ${ticks}
      <text x="${padL + 4}" y="${padT - 8}" fill="#64748B" font-size="9.5">Freq. (escala √ para evidenciar a cauda de buracos)</text>
    </svg>
  `;
}

function renderTabelaRegioes() {
  const wrap = $("#regTableWrap");
  if (!wrap || !state.result) return;
  let html = "";

  if (state.regTab === "trincas") {
    const rows = state.result.regioes_trinca;
    if (!rows.length) {
      wrap.innerHTML = `<p class="muted" style="padding:12px">Nenhuma trinca aceita nesta imagem.</p>`;
      return;
    }
    html = `<table class="data-table"><thead><tr><th>#</th><th>Tipo (DNIT)</th><th>Comp. (px)</th><th>Área (px)</th><th>Larg. Média</th><th>Bifurcações</th></tr></thead><tbody>
      ${rows
        .map(
          (r) => `<tr data-key="t_${r.id}" data-bbox="${r.bbox.join(",")}">
            <td>${r.id}</td>
            <td><span class="chip chip-${escHtml(r.tipo)}">${escHtml(r.tipo.replace(/_/g, " "))}</span></td>
            <td>${r.comprimento}</td>
            <td>${r.area}</td>
            <td>${r.largura_px.toFixed(2)} px</td>
            <td>${r.bifurcacoes}</td>
          </tr>`
        )
        .join("")}
    </tbody></table>`;
  } else if (state.regTab === "buracos") {
    const rows = state.result.regioes_buraco;
    if (!rows.length) {
      wrap.innerHTML = `<p class="muted" style="padding:12px">Nenhum buraco aceito nesta imagem.</p>`;
      return;
    }
    html = `<table class="data-table"><thead><tr><th>#</th><th>Área (px)</th><th>Perímetro</th><th>Circularidade</th><th>Aspecto</th><th>Escurec. Médio</th></tr></thead><tbody>
      ${rows
        .map(
          (r, i) => `<tr data-key="b_${r.id}" data-bbox="${r.bbox.join(",")}">
            <td>${i + 1}</td>
            <td>${fmtInt(r.area)}</td>
            <td>${r.perimetro}</td>
            <td>${(r.circularidade_1000 / 1000).toFixed(3)}</td>
            <td>${(r.aspecto_1000 / 1000).toFixed(2)}×</td>
            <td>${r.escurecimento_medio}</td>
          </tr>`
        )
        .join("")}
    </tbody></table>`;
  } else if (state.regTab === "rej_trincas") {
    const rows = state.result.rejeitados_trinca;
    if (!rows.length) {
      wrap.innerHTML = `<p class="muted" style="padding:12px">Nenhum componente de trinca foi rejeitado pelos filtros geométricos.</p>`;
      return;
    }
    html = `<table class="data-table"><thead><tr><th>Comp.</th><th>Área</th><th>Esqueleto</th><th>Motivo da Rejeição</th></tr></thead><tbody>
      ${rows
        .map(
          (r) => `<tr data-key="rt_${r.id}" data-bbox="${r.bbox.join(",")}">
            <td>#${r.id}</td>
            <td>${r.area} px</td>
            <td>${r.comprimento} px</td>
            <td><span class="chip chip-rejeitado">${escHtml(r.motivos.join(" · "))}</span></td>
          </tr>`
        )
        .join("")}
    </tbody></table>`;
  } else {
    const rows = state.result.rejeitados_buraco;
    if (!rows.length) {
      wrap.innerHTML = `<p class="muted" style="padding:12px">Nenhum componente de buraco foi rejeitado pelos filtros de forma.</p>`;
      return;
    }
    html = `<table class="data-table"><thead><tr><th>Comp.</th><th>Área</th><th>Circ.</th><th>Aspecto</th><th>Motivo da Rejeição</th></tr></thead><tbody>
      ${rows
        .map(
          (r) => `<tr data-key="rb_${r.id}" data-bbox="${r.bbox.join(",")}">
            <td>#${r.id}</td>
            <td>${fmtInt(r.area)} px</td>
            <td>${(r.circularidade_1000 / 1000).toFixed(3)}</td>
            <td>${(r.aspecto_1000 / 1000).toFixed(2)}×</td>
            <td><span class="chip chip-rejeitado">${escHtml(r.motivos.join(" · "))}</span></td>
          </tr>`
        )
        .join("")}
    </tbody></table>`;
  }

  wrap.innerHTML = html;
  wrap.querySelectorAll("tbody tr").forEach((tr) => {
    const bbox = tr.dataset.bbox.split(",").map(Number);
    const key = tr.dataset.key;
    if (state.selectedRegionKey === key) tr.classList.add("selected-row");
    tr.addEventListener("mouseenter", () => {
      if (!state.selectedRegionKey) {
        state.highlightBbox = bbox;
        desenharCanvas();
      }
    });
    tr.addEventListener("mouseleave", () => {
      if (!state.selectedRegionKey) {
        state.highlightBbox = null;
        desenharCanvas();
      }
    });
    tr.addEventListener("click", () => {
      if (state.selectedRegionKey === key) {
        state.selectedRegionKey = null;
        state.highlightBbox = null;
      } else {
        state.selectedRegionKey = key;
        state.highlightBbox = bbox;
      }
      wrap.querySelectorAll("tbody tr").forEach((r) => r.classList.toggle("selected-row", r.dataset.key === state.selectedRegionKey));
      desenharCanvas();
    });
  });
}

// ==================== VIEW 2: ETAPAS (MOSAICO) ====================
function renderMosaico() {
  if (!state.result) return;
  const est = state.result.estatisticas;
  const q = state.result.quant;
  const tiles = [
    { step: "1", sub: "1_original", num: "01", title: "Original (RGB)", foot: `${state.result.dimensoes.original.join("×")} → ${state.result.dimensoes.processamento.join("×")}` },
    { step: "1", sub: "1_roi", num: "01", title: "Intensidade HSI + ROI", foot: `ROI = ${state.result.dimensoes.roi_pct}% da imagem` },
    { step: "2", sub: "2_media", num: "02", title: "Média Local (m_xy)", foot: `Filtro de caixa ${est.janela_trinca}×${est.janela_trinca} em O(1)` },
    { step: "2", sub: "2_desvio_b", num: "02", title: "Desvio Local (Textura s_b)", foot: `Janela ${est.janela_buraco}×${est.janela_buraco} · Med=${est.mediana_textura}, MAD=${est.mad_textura}` },
    { step: "3a", sub: "3a_sementes", num: "3A", title: "Trincas: Sementes × Fracas", foot: `${fmtInt(est.px_trinca_forte)} fortes · ${fmtInt(est.px_trinca_fraca_bruta)} fracas` },
    { step: "3a", sub: "3a_histerese", num: "3A", title: "Trincas: Pós-Histerese", foot: `${fmtInt(est.px_trinca_pos_histerese)} mantidos · ${fmtInt(est.px_trinca_descartados_histerese)} ruído` },
    { step: "3b", sub: "3b_textura", num: "3B", title: "Buracos: Mediana + MAD", foot: `t_tex = ${est.limiar_textura} · ${fmtInt(est.px_buraco_cand)} px candidatos` },
    { step: "4", sub: "4_abertura", num: "04", title: "Morfologia: Abertura", foot: `−${fmtInt(est.px_buraco_removido_abertura)} px estreitos apagados` },
    { step: "4", sub: "4_fecho_preench", num: "04", title: "Morfologia: Fecho + Preench.", foot: `+${fmtInt(est.px_buraco_adicionado_preench)} px de miolo preenchidos` },
    { step: "5", sub: "5_esqueleto", num: "05", title: "Esqueleto Zhang-Suen", foot: `${q.n_trincas} trincas · ${fmtInt(q.comprimento_trincas_px)} px · ${fmtInt(est.px_bifurcacoes)} bif.` },
    { step: "5", sub: "5_buracos_forma", num: "05", title: "Buracos: Filtros de Forma", foot: `${q.n_buracos} aceitos · ${est.n_comp_buraco_rejeitados} rejeitados` },
    { step: "5", sub: "5_final", num: "05", title: "Resultado Final", foot: `Índice ${(q.indice_dano_1000 / 10).toFixed(1)}/100 (${formatarSeveridade(q.severidade)})` },
  ];
  if (state.result.metricas_gt && state.result.imagens["5_gt_comparar"]) {
    const gt = state.result.metricas_gt;
    tiles.push({
      step: "5",
      sub: "5_gt_comparar",
      num: "GT",
      title: "vs. Gabarito DNIT",
      foot: `Trinca F1(2px)=${gt.trinca_tol2.f1.toFixed(2)} · Buraco IoU=${gt.buraco.iou.toFixed(2)}`,
    });
  }

  const grid = $("#mosaicGrid");
  grid.innerHTML = tiles
    .map(
      (t) => `
      <article class="mosaic-tile" data-step="${escHtml(t.step)}" data-sub="${escHtml(t.sub)}">
        <div class="mosaic-tile-head">
          <span>${escHtml(t.title)}</span>
          <span class="mosaic-tile-badge">ETAPA ${escHtml(t.num)}</span>
        </div>
        <img src="${state.result.imagens[t.sub]}" alt="${escHtml(t.title)}">
        <div class="mosaic-tile-foot">${escHtml(t.foot)}</div>
      </article>
    `
    )
    .join("");

  grid.querySelectorAll(".mosaic-tile").forEach((tile) => {
    tile.addEventListener("click", () => {
      state.step = tile.dataset.step;
      state.substepByStep[state.step] = tile.dataset.sub;
      state.highlightBbox = null;
      trocarView("etapas");
      renderEtapaAtual();
    });
  });
}

// ==================== VIEW 3: REQUISITOS & RESULTADOS ====================
function renderRequisitos() {
  const tbody = $("#tabelaRequisitos");
  if (!tbody) return;
  const reqs = window.PAVIMENTO_INFO?.REQUISITOS_PROFESSOR || [];
  tbody.innerHTML = reqs
    .map(
      (r, idx) => `
      <tr>
        <td><strong>${escHtml(r.item)}</strong></td>
        <td><strong>${escHtml(r.topico)}</strong></td>
        <td>${escHtml(r.comoAtende)}</td>
        <td><span class="req-code">${escHtml(r.codigo)}</span></td>
        <td><span class="req-badge-ok">✓ Atendido</span></td>
        <td><button type="button" class="req-demo-btn" data-req-idx="${idx}">${escHtml(r.labelBotao)} →</button></td>
      </tr>
    `
    )
    .join("");

  tbody.querySelectorAll(".req-demo-btn[data-req-idx]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const r = reqs[Number(btn.dataset.reqIdx)];
      const acao = r?.acao;
      if (!acao) return;

      if (acao.tipo === "step") {
        state.step = acao.step;
        if (acao.substep) state.substepByStep[acao.step] = acao.substep;
        if (acao.compareMode) {
          state.compareMode = acao.compareMode;
          atualizarCompareModeUI();
        }
        state.highlightBbox = null;
        trocarView("etapas");
        renderEtapaAtual();
        window.scrollTo({ top: 0, behavior: "smooth" });
      } else if (acao.tipo === "view") {
        trocarView(acao.view);
        window.scrollTo({ top: 0, behavior: "smooth" });
      } else if (acao.tipo === "dnit") {
        $("#modalDnit")?.classList.remove("hidden");
        carregarListaDnit();
      }
    });
  });
}

function renderResultadosOficiais() {
  const tbody = $("#tabelaMetricasOficiais");
  if (!tbody) return;
  tbody.innerHTML = state.metricasOficiais
    .map(
      (m) => `
      <tr>
        <td><strong>${escHtml(m.metrica)}</strong></td>
        <td style="font-family:var(--font-mono); color:var(--ink-soft)">${escHtml(m.inicial ?? "—")}</td>
        <td style="font-family:var(--font-mono); color:var(--ink-soft)">${escHtml(m.anterior ?? "—")}</td>
        <td style="font-family:var(--font-mono); font-weight:800; color:var(--teal)">${escHtml(m.final)}</td>
        <td>${escHtml(m.destaque)}</td>
      </tr>
    `
    )
    .join("");
}

function renderAnaliseCritica() {
  const grid = $("#analiseCriticaGrid");
  if (!grid) return;
  const crit = window.PAVIMENTO_INFO?.ANALISE_CRITICA;
  if (!crit) {
    grid.innerHTML = "";
    return;
  }
  grid.innerHTML = `
    <article class="crit-card crit-ok">
      <h3>Pontos Fortes</h3>
      <ul>
        ${(crit.pontosFortes || []).map((p) => `<li>${escHtml(p)}</li>`).join("")}
      </ul>
    </article>
    <article class="crit-card crit-warn">
      <h3>Limitações &amp; Trabalhos Futuros</h3>
      <ul>
        ${(crit.limitacoes || []).map((p) => `<li>${escHtml(p)}</li>`).join("")}
      </ul>
    </article>
  `;
}

// ==================== NAVEGAÇÃO E EVENTOS ====================
function trocarView(viewName) {
  state.view = viewName;
  $$(".mode-tab").forEach((b) => b.classList.toggle("active", b.dataset.view === viewName));
  $("#viewEtapas").classList.toggle("active", viewName === "etapas");
  $("#viewMosaico").classList.toggle("active", viewName === "mosaico");
  $("#viewResultados").classList.toggle("active", viewName === "resultados");
  if (viewName === "mosaico") renderMosaico();
  if (viewName === "etapas") setTimeout(desenharCanvas, 20);
}

function selecionarCenario(id) {
  const cen = state.cenarios.find((c) => c.id === id);
  if (!cen) return;
  state.cenarioId = id;
  state.amostraDnit = null;
  state.uploadId = null;
  state.uploadNome = "";
  state.temRoiUpload = false;
  state.roiModo = "cenario";
  state.roiPoligono = cen.roi_poligono.map((pt) => [...pt]);
  state.reduzir2x = cen.reduzir_padrao;
  state.activeExperimentId = null;
  state.experimentBackup = null;
  state.pinnedPixel = null;
  state.highlightBbox = null;
  state.selectedRegionKey = null;
  state.zoom = 1.0;
  state.panX = 0;
  state.panY = 0;

  $("#chkReduzir2x").checked = state.reduzir2x;
  $("#btnUploadRoi").classList.add("hidden");
  atualizarBotoesRoi();
  executarAnalise();
}

function selecionarAmostraDnit(nome) {
  state.amostraDnit = nome;
  state.uploadId = null;
  state.uploadNome = "";
  state.temRoiUpload = false;
  state.roiModo = "cenario";
  state.roiPoligono = [[0.25, 0.0], [0.75, 0.0], [0.98, 1.0], [0.02, 1.0]];
  state.reduzir2x = true;
  state.activeExperimentId = null;
  state.experimentBackup = null;
  state.pinnedPixel = null;
  state.highlightBbox = null;
  state.selectedRegionKey = null;
  state.zoom = 1.0;
  state.panX = 0;
  state.panY = 0;

  const sel = $("#selectCenario");
  const valAlvo = `dnit:${nome}`;
  if (!sel.querySelector(`option[value="${valAlvo}"]`)) {
    let optCustom = sel.querySelector('option[data-dnit-custom="1"]');
    if (!optCustom) {
      optCustom = document.createElement("option");
      optCustom.dataset.dnitCustom = "1";
      sel.prepend(optCustom);
    }
    optCustom.value = valAlvo;
    optCustom.textContent = `DNIT Real · ${nome} (LANE + GT)`;
  }
  sel.value = valAlvo;
  $("#chkReduzir2x").checked = true;
  $("#btnUploadRoi").classList.add("hidden");
  atualizarBotoesRoi();
  executarAnalise();
}

async function carregarListaDnit() {
  const m = state.dnitModal;
  const qs = new URLSearchParams({
    q: m.q,
    particao: m.particao,
    estado: m.estado,
    offset: String(m.offset),
    limit: String(m.limit),
  });
  $("#dnitStatus").textContent = "Buscando amostras...";
  const resp = await fetch(`/api/dataset?${qs.toString()}`);
  const data = await resp.json();
  m.totalFiltrado = data.filtrado;

  $("#dnitStatus").textContent = `Exibindo ${data.amostras.length} de ${fmtInt(data.filtrado)} amostras filtradas (total local: ${fmtInt(data.total)})`;
  const pagAtual = Math.floor(m.offset / m.limit) + 1;
  const pagTotal = Math.max(1, Math.ceil(data.filtrado / m.limit));
  $("#dnitPagInfo").textContent = `Página ${pagAtual} de ${pagTotal}`;
  $("#btnDnitPrev").disabled = m.offset <= 0;
  $("#btnDnitNext").disabled = m.offset + m.limit >= data.filtrado;

  const list = $("#dnitList");
  if (!data.amostras.length) {
    list.innerHTML = `<p class="muted" style="padding:16px">Nenhuma amostra encontrada com esse filtro.</p>`;
    return;
  }
  list.innerHTML = data.amostras
    .map(
      (a) => `
      <button class="dnit-item ${state.amostraDnit === a.nome ? "active" : ""}" data-dnit="${escHtml(a.nome)}">
        <span class="dnit-item-name">${escHtml(a.nome)}</span>
        <span class="dnit-item-meta">
          <span>${escHtml(a.rodovia)}</span>
          <span class="chip ${a.particao === "teste" ? "chip-longitudinal" : "chip-rejeitado"}">${escHtml(a.particao)}</span>
        </span>
      </button>
    `
    )
    .join("");

  list.querySelectorAll(".dnit-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("#modalDnit").classList.add("hidden");
      selecionarAmostraDnit(btn.dataset.dnit);
      showToast(`Amostra DNIT ${btn.dataset.dnit} carregada!`);
    });
  });
}

async function sortearAmostraTesteDnit() {
  const resp = await fetch("/api/dataset?particao=teste&offset=0&limit=200");
  const data = await resp.json();
  if (!data.filtrado) return;
  const randIdx = Math.floor(Math.random() * data.filtrado);
  const r2 = await fetch(`/api/dataset?particao=teste&offset=${randIdx}&limit=1`);
  const d2 = await r2.json();
  if (d2.amostras && d2.amostras[0]) {
    selecionarAmostraDnit(d2.amostras[0].nome);
    showToast(`Sorteada amostra de teste: ${d2.amostras[0].nome}`);
  }
}

function atualizarBotoesRoi() {
  $$("#roiModeGroup .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.roi === state.roiModo));
  $("#roiEditBanner").classList.toggle("hidden", state.roiModo !== "poligono");
}

function lerArquivoDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

async function enviarFotoUsuario(file, roiFile = null) {
  if (!file) return;
  setLoading(true);
  try {
    const imgB64 = await lerArquivoDataURL(file);
    const roiB64 = roiFile ? await lerArquivoDataURL(roiFile) : null;
    const resp = await fetch("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imagem_b64: imgB64, nome: file.name, roi_b64: roiB64 }),
    });
    const info = await resp.json();
    if (!resp.ok) throw new Error(info.erro || "Erro no upload");

    state.uploadId = info.upload_id;
    state.amostraDnit = null;
    state.uploadNome = info.nome;
    state.temRoiUpload = info.tem_roi_upload;
    state.reduzir2x = info.reduzir_padrao;
    state.roiModo = info.tem_roi_upload ? "cenario" : "inteira";
    state.roiPoligono = [[0.22, 0.0], [0.78, 0.0], [0.96, 1.0], [0.04, 1.0]];
    state.activeExperimentId = null;
    state.experimentBackup = null;
    state.pinnedPixel = null;
    state.highlightBbox = null;
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;

    const sel = $("#selectCenario");
    let opt = sel.querySelector('option[value="__upload__"]');
    if (!opt) {
      opt = document.createElement("option");
      opt.value = "__upload__";
      sel.prepend(opt);
    }
    opt.textContent = `Enviada: ${info.nome} (${info.largura}×${info.altura})`;
    sel.value = "__upload__";
    $("#chkReduzir2x").checked = state.reduzir2x;
    $("#btnUploadRoi").classList.remove("hidden");
    atualizarBotoesRoi();

    showToast(`Foto "${info.nome}" carregada!`);
    await executarAnalise();
  } catch (err) {
    showToast(`Erro: ${err.message}`);
    setLoading(false);
  }
}

async function baixarPainelPng() {
  try {
    showToast("Gerando painel oficial em PNG…");
    const resp = await fetch("/api/exportar_painel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(montarPayloadAnalise()),
    });
    if (!resp.ok) throw new Error("Falha ao gerar PNG");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pavimento_${state.amostraDnit || (state.uploadId ? "foto" : state.cenarioId)}_painel.png`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast(`Erro: ${err.message}`);
  }
}

function baixarRelatorioJson() {
  if (!state.result) return;
  const relatorio = {
    origem: state.amostraDnit || (state.uploadId ? state.uploadNome : state.cenarioId),
    dimensoes: state.result.dimensoes,
    tempo_nucleo_ms: state.result.tempo_nucleo_ms,
    params: state.result.params,
    quant: state.result.quant,
    metricas_gt: state.result.metricas_gt,
    regioes_trinca: state.result.regioes_trinca,
    regioes_buraco: state.result.regioes_buraco,
  };
  const blob = new Blob([JSON.stringify(relatorio, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pavimento_${state.amostraDnit || (state.uploadId ? "foto" : state.cenarioId)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==================== INTERAÇÃO COM O CANVAS ====================
function conectarEventosCanvas() {
  const stage = $("#canvasStage");

  function findRoiHandle(mx, my) {
    if (state.roiModo !== "poligono") return -1;
    const lay = calcularLayoutCanvas();
    if (!lay || lay.mode === "lado") return -1;
    for (let i = 0; i < state.roiPoligono.length; i++) {
      const [px, py] = state.roiPoligono[i];
      const hx = lay.x + px * (lay.iw - 1) * lay.scale;
      const hy = lay.y + py * (lay.ih - 1) * lay.scale;
      if (Math.hypot(mx - hx, my - hy) <= 14) return i;
    }
    return -1;
  }

  mainCanvas.addEventListener("mousedown", (e) => {
    const rect = mainCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const handleIdx = findRoiHandle(mx, my);
    if (handleIdx >= 0) {
      state.dragMode = "roi_pt";
      state.dragRoiIndex = handleIdx;
      return;
    }

    state.dragMode = "maybe_pan";
    state.dragStart = { x: e.clientX, y: e.clientY, panX: state.panX, panY: state.panY, moved: false };
  });

  window.addEventListener("mousemove", (e) => {
    if (state.dragMode === "curtain") {
      const lay = calcularLayoutCanvas();
      if (!lay) return;
      const rect = mainCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      state.curtainPos = Math.max(0.02, Math.min(0.98, (mx - lay.x) / Math.max(lay.dw, 1)));
      desenharCanvas();
      return;
    }

    if (state.dragMode === "roi_pt") {
      const lay = calcularLayoutCanvas();
      if (!lay) return;
      const rect = mainCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const nx = Math.max(0, Math.min(1, (mx - lay.x) / Math.max(lay.dw, 1)));
      const ny = Math.max(0, Math.min(1, (my - lay.y) / Math.max(lay.dh, 1)));
      state.roiPoligono[state.dragRoiIndex] = [Number(nx.toFixed(3)), Number(ny.toFixed(3))];
      desenharCanvas();
      return;
    }

    if (state.dragMode === "maybe_pan" && state.dragStart) {
      const dx = e.clientX - state.dragStart.x;
      const dy = e.clientY - state.dragStart.y;
      if (Math.hypot(dx, dy) > 4) state.dragStart.moved = true;
      if (state.dragStart.moved && state.zoom > 1.0) {
        state.panX = state.dragStart.panX + dx;
        state.panY = state.dragStart.panY + dy;
        desenharCanvas();
      }
      return;
    }
  });

  window.addEventListener("mouseup", (e) => {
    if (state.dragMode === "maybe_pan" && state.dragStart && !state.dragStart.moved) {
      const pt = canvasParaPixel(e.clientX, e.clientY);
      if (pt) {
        if (state.pinnedPixel && state.pinnedPixel.x === pt.x && state.pinnedPixel.y === pt.y) {
          state.pinnedPixel = null;
        } else {
          state.pinnedPixel = { x: pt.x, y: pt.y };
        }
        atualizarInspetorPixel(state.pinnedPixel || state.hoverPixel);
        desenharCanvas();
      }
    }
    state.dragMode = null;
    state.dragRoiIndex = -1;
    state.dragStart = null;
  });

  mainCanvas.addEventListener("mousemove", (e) => {
    if (state.dragMode) return;
    const pt = canvasParaPixel(e.clientX, e.clientY);
    state.hoverPixel = pt ? { x: pt.x, y: pt.y } : null;
    if (!state.pinnedPixel) atualizarInspetorPixel(state.hoverPixel);
    desenharLupa(pt);
    desenharCanvas();
  });

  mainCanvas.addEventListener("mouseleave", () => {
    state.hoverPixel = null;
    loupeCanvas.classList.add("hidden");
    if (!state.pinnedPixel) atualizarInspetorPixel(null);
    desenharCanvas();
  });

  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const fator = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    state.zoom = Math.max(1.0, Math.min(8.0, state.zoom * fator));
    if (state.zoom <= 1.01) {
      state.zoom = 1.0;
      state.panX = 0;
      state.panY = 0;
    }
    $("#lblZoom").textContent = `${Math.round(state.zoom * 100)}%`;
    desenharCanvas();
  }, { passive: false });

  $("#curtainHandle").addEventListener("mousedown", (e) => {
    e.stopPropagation();
    state.dragMode = "curtain";
  });

  window.addEventListener("resize", () => desenharCanvas());
}

// ==================== INICIALIZAÇÃO ====================
async function inicializar() {
  const resp = await fetch("/api/init");
  const init = await resp.json();
  state.cenarios = init.cenarios;
  state.dataset = init.dataset || { disponivel: false, total: 0, destaques: [], estados: [] };
  state.presets = init.presets;
  state.metricasOficiais = init.metricas_oficiais;
  state.params = { ...init.presets.final };

  const sel = $("#selectCenario");
  if (state.dataset.disponivel && state.dataset.destaques.length) {
    const grupoDnit = document.createElement("optgroup");
    grupoDnit.label = `Dataset DNIT Real (${fmtInt(state.dataset.total)} imagens · ROI + Gabarito reais)`;
    state.dataset.destaques.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = `dnit:${d.nome}`;
      opt.textContent = `${d.titulo} — ${d.subtitulo}`;
      grupoDnit.appendChild(opt);
    });
    sel.appendChild(grupoDnit);

    const btnExp = $("#btnExplorarDnit");
    btnExp.textContent = `Explorar DNIT (${fmtInt(state.dataset.total)})`;
    btnExp.classList.remove("hidden");
    $("#btnAleatoriaDnit").classList.remove("hidden");
    $("#modalDnitTitulo").textContent = `Explorar ${fmtInt(state.dataset.total)} Amostras Reais do DNIT`;

    const selEst = $("#dnitFiltroEstado");
    state.dataset.estados.forEach((uf) => {
      const opt = document.createElement("option");
      opt.value = uf;
      opt.textContent = `Estado: ${uf}`;
      selEst.appendChild(opt);
    });
  }

  const grupoReal = document.createElement("optgroup");
  grupoReal.label = "Recortes Embutidos da Apresentação";
  const grupoDid = document.createElement("optgroup");
  grupoDid.label = "Cenários Didáticos Controlados (1024×640)";
  state.cenarios.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.titulo} — ${c.subtitulo}`;
    (c.categoria === "real" ? grupoReal : grupoDid).appendChild(opt);
  });
  sel.appendChild(grupoReal);
  sel.appendChild(grupoDid);

  $$(".mode-tab").forEach((btn) => btn.addEventListener("click", () => trocarView(btn.dataset.view)));

  $$(".step-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.step = btn.dataset.step;
      state.highlightBbox = null;
      renderEtapaAtual();
    });
  });

  $("#btnToggleInspector").addEventListener("click", () => {
    state.inspectorOpen = !state.inspectorOpen;
    $("#pixelInspector").classList.toggle("hidden", !state.inspectorOpen);
    const btnInsp = $("#btnToggleInspector");
    btnInsp.classList.toggle("active", state.inspectorOpen);
    btnInsp.setAttribute("aria-expanded", String(state.inspectorOpen));
    if (state.inspectorOpen) {
      atualizarInspetorPixel(state.pinnedPixel || state.hoverPixel);
    }
  });

  $("#btnPrevStep").addEventListener("click", () => {
    const i = STEP_ORDER.indexOf(state.step);
    if (i > 0) {
      state.step = STEP_ORDER[i - 1];
      state.highlightBbox = null;
      renderEtapaAtual();
    }
  });
  $("#btnNextStep").addEventListener("click", () => {
    const i = STEP_ORDER.indexOf(state.step);
    if (i < STEP_ORDER.length - 1) {
      state.step = STEP_ORDER[i + 1];
      state.highlightBbox = null;
      renderEtapaAtual();
    }
  });

  $$(".side-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".side-tab").forEach((b) => b.classList.toggle("active", b === btn));
      $("#paneExplicacao").classList.toggle("active", btn.dataset.sidetab === "explicacao");
      $("#paneParametros").classList.toggle("active", btn.dataset.sidetab === "parametros");
    });
  });

  sel.addEventListener("change", () => {
    if (sel.value.startsWith("dnit:")) {
      selecionarAmostraDnit(sel.value.slice(5));
    } else if (sel.value !== "__upload__") {
      selecionarCenario(sel.value);
    }
  });

  $("#btnExplorarDnit").addEventListener("click", () => {
    $("#modalDnit").classList.remove("hidden");
    carregarListaDnit();
  });
  $("#btnFecharModalDnit").addEventListener("click", () => $("#modalDnit").classList.add("hidden"));
  $("#modalDnit").addEventListener("click", (e) => {
    if (e.target.id === "modalDnit") $("#modalDnit").classList.add("hidden");
  });
  let buscaTimer = null;
  $("#dnitBusca").addEventListener("input", (e) => {
    state.dnitModal.q = e.target.value;
    state.dnitModal.offset = 0;
    clearTimeout(buscaTimer);
    buscaTimer = setTimeout(carregarListaDnit, 180);
  });
  $("#dnitFiltroParticao").addEventListener("change", (e) => {
    state.dnitModal.particao = e.target.value;
    state.dnitModal.offset = 0;
    carregarListaDnit();
  });
  $("#dnitFiltroEstado").addEventListener("change", (e) => {
    state.dnitModal.estado = e.target.value;
    state.dnitModal.offset = 0;
    carregarListaDnit();
  });
  $("#btnDnitPrev").addEventListener("click", () => {
    state.dnitModal.offset = Math.max(0, state.dnitModal.offset - state.dnitModal.limit);
    carregarListaDnit();
  });
  $("#btnDnitNext").addEventListener("click", () => {
    state.dnitModal.offset += state.dnitModal.limit;
    carregarListaDnit();
  });
  $("#btnAleatoriaDnit").addEventListener("click", sortearAmostraTesteDnit);

  $("#btnUploadFoto").addEventListener("click", () => $("#inputFoto").click());
  $("#inputFoto").addEventListener("change", (e) => {
    if (e.target.files[0]) enviarFotoUsuario(e.target.files[0]);
    e.target.value = "";
  });
  $("#btnUploadRoi").addEventListener("click", () => $("#inputRoi").click());
  $("#inputRoi").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f || !state.uploadId) return;
    try {
      const roiB64 = await lerArquivoDataURL(f);
      const resp = await fetch("/api/upload_roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ upload_id: state.uploadId, roi_b64: roiB64 }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.erro || "Erro ao aplicar máscara ROI");
      state.temRoiUpload = true;
      state.roiModo = "cenario";
      atualizarBotoesRoi();
      showToast(`Máscara ROI "${f.name}" aplicada!`);
      await executarAnalise();
    } catch (err) {
      showToast(`Erro: ${err.message}`);
    }
  });

  window.addEventListener("dragover", (e) => {
    e.preventDefault();
    $("#dropOverlay").classList.remove("hidden");
  });
  $("#dropOverlay").addEventListener("dragleave", () => $("#dropOverlay").classList.add("hidden"));
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    $("#dropOverlay").classList.add("hidden");
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) enviarFotoUsuario(file);
  });

  $$("#roiModeGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.roiModo = btn.dataset.roi;
      atualizarBotoesRoi();
      if (state.roiModo === "poligono") {
        if (state.compareMode === "lado") {
          state.compareMode = "unico";
          atualizarCompareModeUI();
        }
        desenharCanvas();
      } else {
        executarAnalise();
      }
    });
  });

  $("#btnAplicarRoi").addEventListener("click", () => executarAnalise());
  $("#btnResetRoi").addEventListener("click", () => {
    const cen = state.cenarios.find((c) => c.id === state.cenarioId);
    state.roiPoligono = cen ? cen.roi_poligono.map((pt) => [...pt]) : [[0.22, 0.0], [0.78, 0.0], [0.96, 1.0], [0.04, 1.0]];
    executarAnalise();
  });

  $("#chkReduzir2x").addEventListener("change", (e) => {
    state.reduzir2x = e.target.checked;
    state.pinnedPixel = null;
    executarAnalise();
  });

  $$("#compareModeGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.compareMode = btn.dataset.cmp;
      atualizarCompareModeUI();
      desenharCanvas();
    });
  });

  $("#btnZoomIn").addEventListener("click", () => {
    state.zoom = Math.min(8.0, state.zoom * 1.35);
    $("#lblZoom").textContent = `${Math.round(state.zoom * 100)}%`;
    desenharCanvas();
  });
  $("#btnZoomOut").addEventListener("click", () => {
    state.zoom = Math.max(1.0, state.zoom / 1.35);
    if (state.zoom <= 1.01) {
      state.zoom = 1.0;
      state.panX = 0;
      state.panY = 0;
    }
    $("#lblZoom").textContent = `${Math.round(state.zoom * 100)}%`;
    desenharCanvas();
  });
  $("#btnZoomFit").addEventListener("click", () => {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    $("#lblZoom").textContent = "100%";
    desenharCanvas();
  });
  $("#btnLoupe").addEventListener("click", (e) => {
    state.loupe = !state.loupe;
    e.currentTarget.classList.toggle("active", state.loupe);
    if (!state.loupe) loupeCanvas.classList.add("hidden");
  });

  $("#btnPresetFinal").addEventListener("click", () => {
    state.params = { ...state.presets.final };
    state.activeExperimentId = null;
    state.experimentBackup = null;
    renderParametrosLaterais();
    showToast("Carregado preset Otimizado (params.json)");
    executarAnalise();
  });
  $("#btnPresetSlides").addEventListener("click", () => {
    state.params = { ...(state.presets.slides_anterior || state.presets.final) };
    state.activeExperimentId = null;
    state.experimentBackup = null;
    renderParametrosLaterais();
    showToast("Carregado preset Slides (r=4)");
    executarAnalise();
  });
  $("#btnPresetPadrao").addEventListener("click", () => {
    state.params = { ...state.presets.padrao };
    state.activeExperimentId = null;
    state.experimentBackup = null;
    renderParametrosLaterais();
    showToast("Carregado preset Padrão (Parametros)");
    executarAnalise();
  });

  $("#btnExportarPng").addEventListener("click", baixarPainelPng);
  $("#btnMosaicoExportar").addEventListener("click", baixarPainelPng);
  $("#btnExportarJson").addEventListener("click", baixarRelatorioJson);

  function toggleApresentacao() {
    document.body.classList.toggle("presentation-mode");
    $("#btnApresentacao").classList.toggle("active", document.body.classList.contains("presentation-mode"));
    setTimeout(desenharCanvas, 40);
  }
  $("#btnApresentacao").addEventListener("click", toggleApresentacao);

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const modalDnit = $("#modalDnit");
      if (modalDnit && !modalDnit.classList.contains("hidden")) {
        modalDnit.classList.add("hidden");
        return;
      }
    }

    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (!$("#modalDnit").classList.contains("hidden")) return;

    if (e.key === "ArrowRight" && state.view === "etapas") {
      const i = STEP_ORDER.indexOf(state.step);
      if (i < STEP_ORDER.length - 1) {
        state.step = STEP_ORDER[i + 1];
        state.highlightBbox = null;
        renderEtapaAtual();
      }
    } else if (e.key === "ArrowLeft" && state.view === "etapas") {
      const i = STEP_ORDER.indexOf(state.step);
      if (i > 0) {
        state.step = STEP_ORDER[i - 1];
        state.highlightBbox = null;
        renderEtapaAtual();
      }
    } else if (e.key >= "1" && e.key <= "6" && state.view === "etapas") {
      state.step = STEP_ORDER[Number(e.key) - 1];
      state.highlightBbox = null;
      renderEtapaAtual();
    } else if (e.key.toLowerCase() === "f") {
      toggleApresentacao();
    }
  });

  atualizarCompareModeUI();
  conectarEventosCanvas();
  renderRequisitos();
  renderResultadosOficiais();
  renderAnaliseCritica();

  if (state.dataset.disponivel && state.dataset.destaques.length) {
    selecionarAmostraDnit(state.dataset.destaques[0].nome);
  } else {
    sel.value = state.cenarioId;
    selecionarCenario(state.cenarioId);
  }
}

inicializar();
