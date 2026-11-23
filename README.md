# Cavalo Mola — Detecção e quantificação de danos em pavimentos

Projeto da disciplina **Processamento de Imagens (PI 2026.2, UFRPE)** — Prof. Lucas Fernando da Silva Cambuim.
Grupo 9: Guilherme Henrique, Caio Santos Bispo e Gabriel Lima.

**Diferencial:** quantificação, alta velocidade e determinismo.

- **Quantificação.** Para cada imagem o sistema mede o comprimento e a largura média das trincas e as classifica em longitudinal, transversal, diagonal ou couro de jacaré. Também mede a área dos buracos em porcentagem da pista e calcula um índice de dano de 0 a 100 com severidade.
- **Velocidade.** Cerca de 40 ms por imagem com NumPy.
- **Determinismo.** Só aritmética inteira: sem float, sem FFT, sem aleatoriedade.

Cada técnica é referenciada ao material do professor em [`docs/REFERENCIAS.md`](docs/REFERENCIAS.md).


## Arquivos locais e materiais externos

Esta versão pública contém código-fonte, parâmetros, modelos em JSON e testes.
O dataset, os resultados gerados, os modelos binários e os PDFs/textos integrais
das aulas não são distribuídos neste repositório. As referências aos slides são
mantidas para atribuição; consulte o material original da disciplina com autorização.
As pastas citadas no restante do README descrevem também o ambiente local de desenvolvimento.
Baixe o dataset na fonte indicada em Referências e gere os resultados pelos scripts.
Alguns experimentos usam artefatos locais adicionais, incluindo o módulo `bench`,
que não está presente na pasta original disponibilizada.

## Estrutura

```
cavalo_mola/
├── src/pavimento/
│   ├── nucleo.py          núcleo determinístico (pipeline completo, só inteiros)
│   ├── parametros.py      parâmetros inteiros (fonte única → params.json)
│   ├── metricas.py        precisão, revocação, F1, IoU
│   ├── dataset.py         leitura do dataset e divisão treino/teste (hash do nome)
│   └── visualizacao.py    painéis e sobreposições (só para relatório)
├── scripts/
│   ├── ajustar.py               ajuste determinístico dos parâmetros (treino)
│   ├── calibrar_severidade.py   limiares de severidade a partir dos rótulos (treino)
│   ├── avaliar.py               métricas no conjunto de teste
│   └── processar_imagem.py      painel com etapas intermediárias de uma imagem
├── params.json                  parâmetros ajustados
├── outputs/                     avaliações, CSVs e painéis de exemplo
├── docs/REFERENCIAS.md          técnica → arquivo/slide do professor
├── docs/texto_aulas/            texto de cada slide (para buscar referências)
├── materiais_PI_2026.2/         material da disciplina
└── experimentos/legado_float/   primeira versão (float, FFT, regressão logística) — histórico
```

## Pipeline

Cada etapa abaixo traz entre colchetes o arquivo e os slides de origem.

1. **Aquisição**: RGB → intensidade HSI, I = (R+G+B)/3 [7_1 Cores · 60–73]. Redução 2× (1024×640 → 512×320) [2_1 · 49–88].
2. **Estatísticas locais**: média m_xy e desvio s_xy numa janela 19×19, com filtro de caixa [4_3 · 31–36] [7_1 Seg · 98–99].
3. **Buracos**: textura dada pelo desvio local [7_1 Seg · 19–24], com limiar global = mediana + k·MAD calculado por histograma [4_2 · 2–9]. Em seguida:
   - abertura, que remove trincas finas [6_1 · 36–39];
   - preenchimento de buracos [6_1 · 61–66];
   - componentes conexos 8-adjacentes [6_1 · 67–73];
   - filtro por área, circularidade e escurecimento [9_2 · 55–62].
4. **Trincas**: limiarização variável, I < m_xy − (a·s_xy + b) [7_1 Seg · 96–100]. Em seguida:
   - fechamento para religar trechos [6_1 · 40–42];
   - componentes conexos;
   - esqueleto Zhang-Suen (complemento), que dá comprimento, largura = área/comprimento e bifurcações;
   - tipo da trinca pelas projeções [9_2 · 66–68].
5. **Quantificação**: densidade de trincas, % da pista com buraco, índice de dano e severidade calibrada nos rótulos.

## Como executar

### 1. Ambiente

```bash
python -m pip install -r requirements.txt
```

O dataset fica em `dataset/Dataset/v1/<amostra>/` com os arquivos `*_RAW.jpg`, `*_CRACK.png`, `*_POTHOLE.png` e `*_LANE.png`.

### 2. Processar uma imagem e gerar o painel de etapas

```bash
python scripts/processar_imagem.py dataset/Dataset/v1/1092981_DF_251_251BDF0052_00911
```

A saída vai para `outputs/exemplos/<nome>_painel.png` (10 etapas) e `<nome>.json` (quantificação e regiões).

### 3. Avaliar, ajustar e calibrar (no PC)

O `outputs/cache/` guarda cada amostra já reduzida (intensidade, pista e máscaras). Os três scripts abaixo trabalham sobre esse cache, usam a mesma divisão treino/teste por hash do nome e **não precisam do dataset bruto** (`CAVALO_CACHE` troca a pasta). Os scripts `avaliar.py`, `ajustar.py` e `calibrar_severidade.py` fazem o mesmo a partir do dataset bruto.

```bash
python scripts/avaliar_cache.py --particao teste                  # métricas (usa params.json)
OBJ=buraco SO=raio_buraco,buraco python scripts/ajustar_cache.py --src . --inicio params.json --saida novo.json   # só treino
python scripts/calibrar_severidade_cache.py . novo.json sev.json    # pesos e limiares da severidade, só treino
python -m unittest discover -s tests                              # 60 testes sintéticos
```

O ajuste foi feito só no treino (785 imagens), em três etapas: busca geral, busca só de buracos e busca só de trincas. Em seguida veio a calibração da severidade contra a referência congelada (limiares 10/214/522, pesos 40/25). O teste (666 imagens) foi avaliado **uma única vez**, ao final.

## Resultados — anterior × depois

Medidos neste repositório sobre `outputs/cache` (`outputs/resultados_pc.json`). A coluna "anterior" é o código e o `params.json` do commit `442b480`, e reproduz exatamente a tabela antiga do README.

| Teste (666 imagens) | Anterior | Depois |
|---|---|---|
| Trincas, detecção por imagem (F1) | 0,927 | **0,928** |
| Trincas, pixel com tolerância de 2 px (F1) | 0,321 | **0,430** |
| Trincas, pixel exato (F1 / IoU) | 0,200 / 0,111 | **0,234** / — |
| Buracos, detecção por imagem (F1) | 0,330 | **0,380** (revocação 0,34 → 0,54) |
| Buracos, pixel (F1 / IoU) | 0,085 / 0,044 | **0,112 / 0,059** |
| Comprimento de trincas (Spearman) | 0,554 | **0,683** |
| Comprimento total previsto / real | 1,113 | **1,050** |
| Severidade, acerto exato | 48,5% | **56,3%** |
| Severidade, erro de no máximo 1 nível | 90,1% | **94,1%** |

O treino mostra o mesmo padrão (buracos pixel F1 0,092 → 0,124; trincas tol 2 px 0,316 → 0,428; severidade 46,4% → 57,8%), então o ganho não vem de sobreajuste.

**Veredito dos critérios pré-definidos (`scripts/_criterios.py`): 5 de 7 aprovados.** Reprovaram o F1 e o IoU de buracos no pixel, que exigiam 2× o anterior; o ganho foi de ~1,3×. O critério de tempo também ficou sem referência oficial (`BASELINE_TESTE` não tem tempo); na comparação medida o código novo é ~11% mais lento.

**O que mudou no núcleo.**
- Trincas: limiarização variável em dois níveis com crescimento a partir de sementes fortes (histerese por componentes), janela menor (`raio=4`) e filtros por comprimento e alongamento.
- Buracos: janela de textura própria (`raio_buraco`) e novo modo por textura (`buraco_modo=1`: desvio local, limiar mediana + k·MAD, abertura, fechamento, preenchimento, filtros de região). O modo antigo, com exigência de escurecimento, ficou pior nos dados reais (F1 pixel 0,03).
- Severidade: pesos do índice e limiares recalibrados no treino contra a referência congelada.

**Análise crítica.** Trincas melhoraram em tudo. Buracos melhoraram, mas continuam o ponto fraco, e o pixel é limitado pelos dados: os buracos ocupam só 0,69% dos pixels da pista, os rótulos são grosseiros, e limiarizar a melhor característica de textura sozinha dá F1 de no máximo ~0,07. A detecção por imagem (F1 0,38) é mais confiável que a máscara. Um classificador sobre descritores de região é o próximo passo (testado em `experimentos/legado_float`).

**Cuidados.** Os testes de geometria de trincas passaram a usar os parâmetros padrão, porque o `params.json` calibrado para imagens reais não é feito para linhas sintéticas ideais; um teste novo cobre o `params.json` real. Os tempos foram medidos com 12 processos em paralelo e só valem para comparação entre si.

## Referências

Veja [`docs/REFERENCIAS.md`](docs/REFERENCIAS.md). Dataset: Passos et al., *Cracks and Potholes in Road Images*, Mendeley Data, 2020, DOI 10.17632/t576ydh9v8.4 (CC BY 4.0).
