# Referências — de onde veio cada técnica

Toda técnica do pipeline aponta para o material da disciplina (Prof. Lucas Fernando da Silva
Cambuim, PI 2026.2, UFRPE), consultado localmente em `materiais_PI_2026.2/` (não redistribuído). O texto extraído de cada slide
fica apenas no ambiente local em `docs/texto_aulas/` (não redistribuído) (busque por `slide N`). Onde o material **não** cobre a técnica,
isso está marcado como **complemento**, com a fonte externa.

## Pipeline × material da disciplina

| Etapa | O que fazemos | Fonte (arquivo · slides) | Código |
|---|---|---|---|
| Área da estrada | Crescimento de regiões sobre blocos 8×8 (média de I, média de S, desvio de I); sementes na base central; conectividade-4; para quando nenhum vizinho satisfaz o critério | `7_2 - segmentacao.pdf` · 4–14 (crescimento de regiões: sementes, similaridade, conectividade sl.13, regra de parada sl.14); `7_1 - segmentacao.pdf` · 23 (sub-regiões); `6_1` · 61–66 (preenchimento) | `estrada.crescer_estrada` |
| Saturação | S = 1 − 3·min(R,G,B)/(R+G+B) | `7_1 - processamento de imagens em cores.pdf` · 70–71 (conversão RGB→HSI) | `estrada.saturacao_hsi` |
| Representação da cor | Intensidade do modelo HSI: I = (R+G+B)/3 | `7_1 - processamento de imagens em cores.pdf` · 60–73 (modelo HSI); saturação/matiz 26–33 | `nucleo.intensidade_hsi` |
| Resolução | Redução 2× por média de blocos 2×2 | `2_1 - fundamentos da imagem digital.pdf` · 49–70 (resolução espacial), 71–88 (interpolação) | `nucleo.reduzir_2x` |
| Vizinhança, bordas da imagem | Janela recortada nas bordas; padding com zeros na morfologia | `2_2 - fundamentos...pdf` · 2–5 (vizinhança), 60–62 (limites de imagem); `4_3` · 15–17 (padding) | `estatisticas_locais`, `_desloc` |
| Filtragem espacial | Filtro de caixa para média e variância locais | `4_3 - melhorias em imagens e filtragem.pdf` · 31–36 (passa-baixa, filtro de caixa) | `nucleo.estatisticas_locais` |
| Histograma | Mediana e MAD por histograma (limiar robusto) | `4_2 - melhorias...pdf` · 2–9 (histograma; slide 7: medidas estatísticas obtidas do histograma) | `nucleo.mediana_mad` |
| Segmentação de trincas | Limiarização variável: pixel é trinca se I < m_xy − (a·s_xy + b) | `7_1 - segmentacao.pdf` · 96–100 (limiarização variável com média e desvio locais); 27–28 (linhas = estruturas finas) | `analisar` (c) |
| Segmentação de buracos | Textura pelo desvio padrão local + limiar global (mediana + k·MAD) | `7_1 - segmentacao.pdf` · 19–24 (textura segmentada pelo desvio padrão em sub-regiões); 85–88 (limiar global) | `analisar` (b) |
| Morfologia | Erosão, dilatação, abertura (remove trincas finas do mapa de buracos), fechamento (religa trincas) | `6_1 - processamento de imagens morfologicas.pdf` · 22–27, 28–33, 36–39, 40–42 | `erodir`, `dilatar`, `abertura`, `fechamento` |
| Preenchimento | Preenchimento de buracos por dilatação condicional | `6_1` · 61–66 | `preencher_buracos` |
| Componentes conexos | Rotulagem com adjacência-8 | `6_1` · 67–73; `2_2` · 6–16 | `rotular` |
| Perímetro | Borda β(A) = A − (A ⊖ B) | `6_1` · 59–60; `9_2` · 59 | `borda` |
| Descritores de região | Área, perímetro, circularidade 4πA/P², intensidade média | `9_2 - extracao de caracteristicas.pdf` · 55–56, 59, 60–62 | `analisar` (b) |
| Tipo de trinca | Extensão horizontal × vertical (projeções) | `9_2` · 66–68 (projeções horizontal e vertical) | `analisar` (c) |
| Comprimento | Comprimento = nº de pixels do esqueleto | `9_2` · 49 (comprimento de contorno como contagem de pixels) | `afinar` |
| Pipeline geral e entregáveis | Aquisição → pré-processamento → filtragem → segmentação → características → classificação → pós-processamento → visualização | `Descricao de projeto.pdf` · item 11 | README |
| Métricas | Precisão, revocação, F1, IoU, tempo de execução | `Descricao de projeto.pdf` · item 11.5 | `metricas.py`, `avaliar.py` |

## Técnicas novas do núcleo Python e do diagnóstico

Referências conferidas em `docs/texto_aulas/`. Os números abaixo correspondem ao
marcador `slide N` da extração, que pode diferir do rodapé impresso (por exemplo,
o marcador 90 de segmentação tem rodapé 92). **Complemento** identifica a fórmula
ou aplicação específica que não aparece explicitamente nas aulas.

| Técnica | Implementação e limite da referência | Arquivo de origem e slides conferidos |
|---|---|---|
| Histerese por componentes | `nucleo.histerese_componentes`: conserva componentes fracos 8-conexos com sementes fortes. **Complemento:** formulação com dois limiares por componentes; as aulas fornecem crescimento e extração conexa. | `docs/texto_aulas/7_2 - segmentacao.txt`, slides 4–5 (crescimento e sementes); `docs/texto_aulas/6_1 - processamento de imagens morfologicas.txt`, slides 67–69 (componentes e estabilidade). |
| Otsu inteiro | `nucleo.otsu_inteiro`: maximiza a separação entre classes por histograma. **Complemento:** produtos cruzados inteiros e desempate pelo menor limiar são a realização aritmética do projeto. | `docs/texto_aulas/7_1 - segmentacao.txt`, slides 89–95, especialmente 90 (variância entre classes) e 93 (busca do limiar). |
| Textura por desvio local (modo de buracos) | `nucleo.analisar` com `buraco_modo=1`: desvio padrão local numa janela própria (`raio_buraco`), limiar global mediana + k·MAD por histograma, abertura, fechamento e preenchimento, depois filtros de região. **Complemento:** janela de textura separada da janela das trincas e escolha por parâmetro; as aulas fornecem estatísticas locais, limiar por histograma e morfologia. | `docs/texto_aulas/7_1 - segmentacao.txt`, slides 19–24 (textura) e 98–99 (estatísticas locais); `docs/texto_aulas/6_1 - processamento de imagens morfologicas.txt`, slides 36–42 e 61–66. |
| Escurecimento local | `nucleo.analisar`: intensidade e média local comparadas à mediana da pista; margens forte/fraca e fração escura por região. **Complemento:** combinação específica com textura e sementes para buracos. | `docs/texto_aulas/7_1 - segmentacao.txt`, slides 96–100, especialmente 97–99 (vizinhança, média e desvio locais). |
| Solidez | Somente no diagnóstico: área de contorno / área do casco convexo, com OpenCV. **Complemento:** fórmula definida em `scripts/diagnosticar.py`, função `componentes`; não encontrada nos slides. | Contexto: `docs/texto_aulas/9_2 - extracao de caracteristicas.txt`, slides 55–58 (descritores de região e área). |
| Extensão | `nucleo.analisar`: área em pixels / área da caixa delimitadora, em milésimos. **Complemento:** razão de ocupação definida no projeto; difere da solidez pelo casco convexo. | Contexto: `docs/texto_aulas/9_2 - extracao de caracteristicas.txt`, slides 55–58 (área) e 66–68 (projeções). |
| Razão de aspecto | `nucleo.analisar`: maior / menor lado da caixa delimitadora, em milésimos. **Complemento:** essa razão da caixa difere da excentricidade da elipse da aula. | Contexto: `docs/texto_aulas/9_2 - extracao de caracteristicas.txt`, slide 63 (excentricidade relativa à elipse) e slides 66–68 (projeções). |
| EE lineares | `nucleo.morfologia_direcional`: elementos estruturantes horizontais, verticais e diagonais para abertura/fechamento de trincas. **Complemento:** banco de quatro direções e regra de suporte por componente do projeto. | `docs/texto_aulas/6_1 - processamento de imagens morfologicas.txt`, slides 16–21 (definição, origem e exemplos de EE), 36–39 (abertura), 40–42 (fechamento). |
| Dilatação condicional | `nucleo.preencher_buracos`: dilatação e intersecção com domínio permitido até estabilidade. **Complemento:** seleção de componentes com sementes em `histerese_componentes` equivale a crescer sob máscara sem repetir dilatações. | `docs/texto_aulas/6_1 - processamento de imagens morfologicas.txt`, slide 62 (recorrência e estabilidade), slide 66 (nome e intersecção), slide 69 (crescimento no foreground). |

## Pipeline híbrido (classificação de padrões por blocos)

| Etapa | O que fazemos | Fonte |
|---|---|---|
| Sub-regiões | Imagem dividida em blocos 16×16 | `7_1 - segmentacao.pdf` · 19–24 (sub-regiões 8×8 descritas pelo desvio padrão) |
| Descritores do bloco | média, desvio, mín., máx.; fração de candidatos do limiar variável; gradiente e coerência; Laplaciano; esqueleto; contexto dos vizinhos | `4_2` · 7; `7_1 - segmentacao.pdf` · 19–24, 57–68, 96–100; `4_3` · 59–65; `9_2` · 49; `2_2` · 2–5 |
| Classificador (oficial) | **Bayes para classes gaussianas** sobre 4 descritores escolhidos por seleção para frente na validação (fração de pixels muito escuros, fração de candidatos de buraco, mínimo do bloco, gradiente médio da vizinhança); inferência inteira | Gonzalez & Woods, 3ª ed., seção 12.2.2; distância [`2_2` · 20–22] |
| Coerência espacial | filtro de **mediana 3×3** sobre a grade de decisões dos blocos | `4_3 - melhorias em imagens e filtragem.pdf` · 54–58 |
| Comparações (experimentos/) | distância mínima com protótipos por K-means (K-means como no `Cores.ipynb` do professor); gradient boosting de árvores; LBP | Gonzalez & Woods 12.2.1; Friedman 2001; Ojala et al. 2002 — **não usados no pipeline oficial** |
| Correção de sombreamento | subtrair da imagem uma versão muito suavizada (fundo) — mesma ideia do limiar variável por média local | código do professor: `materiais_PI_2026.2/codigos_4_filtragem.zip` → `Shading Correction/chess.py` |
| Contorno dos buracos | interpolação bilinear inteira do escore dos blocos | `2_1 - fundamentos da imagem digital.pdf` · 71–88 (interpolação) |
| Referência de desempenho | blocos + GLCM/LBP/Gabor + SVM/kNN/MLP no mesmo dataset: F = 75,16% | FERNANDES, A. M. R.; CASSANIGA, M. J.; PASSOS, B. T.; COMUNELLO, E.; STEFENON, S. F.; LEITHARDT, V. R. Q. *Detection and classification of cracks and potholes in road images using texture descriptors.* J. Intell. Fuzzy Syst. 44(6), 2023. DOI 10.3233/JIFS-223218 |
| Inspiração | descritores manuais + floresta de árvores para trincas | SHI, Y. et al. *Automatic road crack detection using random structured forests* (CrackForest). IEEE T-ITS 17(12), 2016 |

## Complementos (não estão nos slides)

| Técnica | Por que usamos | Fonte |
|---|---|---|
| Afinamento de Zhang-Suen | Medir o **comprimento** e as bifurcações das trincas (base da quantificação) | ZHANG, T. Y.; SUEN, C. Y. *A fast parallel algorithm for thinning digital patterns.* Communications of the ACM, 27(3), 1984. Também em Gonzalez & Woods, *Processamento Digital de Imagens*, 3ª ed., seção 9.5.7. |
| Preenchimento automático a partir da moldura | Variante do slide 6_1·61–66 que dispensa semente manual | Gonzalez & Woods, 3ª ed., seção 9.6 (reconstrução morfológica) |
| MAD (desvio absoluto mediano) | Limiar robusto a outliers | Estatística robusta clássica (Hampel, 1974) |
| Índice de dano 0–1000 | Combina densidade de trincas e área de buracos; couro de jacaré pesa 2× | Inspirado na lógica densidade × severidade do PCI (ASTM D6433). **Não é o PCI oficial**; os limiares de severidade são calibrados nos rótulos do treino (`scripts/calibrar_severidade.py`). |
| Nomes dos defeitos (trinca longitudinal, transversal, couro de jacaré, panela) | Terminologia brasileira | Norma DNIT 005/2003-TER — Defeitos nos pavimentos flexíveis e semirrígidos (citada pelos autores do dataset) |

## Dataset

PASSOS, B. T.; CASSANIGA, M. J.; FERNANDES, A. M. R.; MEDEIROS, K. B.; COMUNELLO, E.
*Cracks and Potholes in Road Images.* Mendeley Data, v4, 2020. DOI: 10.17632/t576ydh9v8.4.
Licença CC BY 4.0. 2235 imagens de rodovias do ES, RS e DF (2014–2017), cedidas pelo DNIT,
com máscaras de pista (LANE), trinca (CRACK) e buraco (POTHOLE).
https://data.mendeley.com/datasets/t576ydh9v8

RATEKE, T.; VON WANGENHEIM, A. *Road surface detection and differentiation considering surface damages.*
Autonomous Robots, 2021. DOI: 10.1007/s10514-020-09964-3. Dataset RTK (Road Traversing Knowledge),
Mendeley Data, DOI 10.17632/hssswvmjwf.1, CC BY 4.0: 701 imagens 352×288 de câmera de baixo custo
(Santa Catarina), 13 classes (fundo, asfalto, calçamento, terra, faixas, ..., buraco, trinca).
Usado para ajustar/avaliar a detecção automática da estrada.

## Firmware

- PlatformIO + Arduino-ESP32; SD_MMC em modo 1 bit (pinos fixos da ESP32-CAM AI-Thinker:
  CLK=14, CMD=15, D0=2). Documentação: https://docs.espressif.com/projects/arduino-esp32/
