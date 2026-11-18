# Roteiro do vídeo de demonstração (rascunho — ajustar com os resultados reais)

Duração sugerida: 4–6 min. Gravar a tela do Colab; nada de slides com resultados prontos no lugar da execução.

| # | Tempo | O que mostrar | Fala (resumo) |
|---|---|---|---|
| 1 | 0:00–0:40 | foto de calçada com barreiras (crédito Mapillary na tela) | O problema: buracos, desníveis e obstáculos impedem a circulação de quem tem mobilidade reduzida. O Rota Livre quer mapeá-los automaticamente. |
| 2 | 0:40–1:30 | `docs/decisoes_tecnicas.md` (tabela) e `configs/train.yaml` | Arquitetura YOLO26s, pesos COCO, por que essa escolha, 3 classes e critérios de anotação. |
| 3 | 1:30–2:10 | `results/figures/distribuicao_classes.png`, `split_report.json` | Bases usadas, divisão por grupo sem vazamento, teste brasileiro separado. |
| 4 | 2:10–3:30 | **execução ao vivo**: célula do `demo_app.py` → enviar foto que não está no treino → resultado | Entrada → inferência real → caixas, classe, confiança e tempo em ms. Repetir com 2–3 fotos (uma com erro, se houver). |
| 5 | 3:30–4:30 | `results/metrics/eval_test.md`, matriz de confusão | Precision, Recall, F1, mAP@50 por classe; tempo de inferência; erros mais frequentes. |
| 6 | 4:30–5:00 | — | Conclusão: o que funciona, limitações (ex.: classe com menos dados, domínio rua × calçada) e próximos passos. |

Dica: rode a célula de demonstração uma vez antes de gravar (o primeiro carregamento é mais lento) e grave a
segunda execução, mostrando a célula sendo executada.
