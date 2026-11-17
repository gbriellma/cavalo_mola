# Experimento: gradient boosting de árvores (não faz parte do pipeline oficial)

Comparação feita antes de decidirmos manter o projeto 100% em técnicas clássicas da disciplina.
Mesmos descritores por bloco (src/pavimento/blocos.py), classificador = boosting de árvores
implementado do zero (src/pavimento/boosting.py), 2 estágios com auto-contexto.

Resultados no teste (666 imagens do DNIT), pipeline completo:
| Métrica | Clássico + Bayes (oficial) | Boosting (este experimento) |
|---|---|---|
| Trinca F1 tol. 2 px | 0,45 | 0,51 |
| Buraco F1 pixel | 0,16 | 0,42 |
| Severidade (acerto / até 1 nível) | 44% / 87% | 62% / 97% |
| Tempo Python | 64 ms | 218 ms |

Para reproduzir: copie modelos_blocos.json e params_pixel_boosting.json (deste diretório) para a
raiz como modelos_blocos.json / params_pixel.json e rode os scripts com CLASSIFICADOR=boosting.
