# Decisões técnicas

## Arquitetura: Ultralytics YOLO26s

| Critério | YOLO26 (Ultralytics) | RF-DETR (Roboflow) | Faster R-CNN (torchvision) |
|---|---|---|---|
| Tipo | detector de 1 estágio (CNN) | transformer (DETR) | detector de 2 estágios |
| Licença | AGPL-3.0 | Apache 2.0 (Nano–Large) | BSD-3 |
| Formato de dados | YOLO .txt | COCO ou YOLO | COCO (código próprio) |
| Métricas prontas | P, R, mAP50, mAP50-95, curva F1, matriz de confusão, tempo | mAP (resto exige código) | nenhuma (tudo manual) |
| Viável na T4 do Colab gratuito | sim | sim (modelos N/S) | lento |

**Escolha:** YOLO26 tamanho *s* (small), pesos iniciais pré-treinados no COCO (`yolo26s.pt`).
- Treina em GPU gratuita dentro do limite de sessão do Colab.
- Produz caixas, classes e confiança e já calcula todas as métricas exigidas, reduzindo risco de erro na avaliação.
- Reprodutível: um YAML de dados + um YAML de hiperparâmetros + um comando.
- Versão: a versão exata do pacote `ultralytics` e do PyTorch fica registrada em
  `results/metrics/train_env.json` e `pip_freeze.txt` na execução do treino.

**Por que não o tamanho *n* ou *m*:** *n* é mais rápido mas perde precisão em objetos pequenos (buracos
distantes); *m* dobra o custo de treino e arrisca estourar o tempo de sessão do Colab gratuito.
Se der falta de memória, a alternativa documentada é `batch 8`; se o tempo de sessão for o problema, `yolo26n.pt`.

### Licença AGPL-3.0, em termos práticos
- Uso acadêmico com repositório público: **permitido**. O repositório deve ser publicado sob AGPL-3.0
  (no GitHub: *Add file → Create new file → LICENSE → Choose a license template → GNU AGPL v3.0*).
- O modelo treinado também fica sob AGPL-3.0.
- Se o Rota Livre virar produto com código fechado, seria preciso comprar a licença Enterprise da
  Ultralytics **ou** retreinar com RF-DETR (Apache 2.0), cujo formato de dados já é compatível com este pipeline.

## Dados
Nenhuma base pública cobre as três classes com caixas em calçadas. Estratégia: combinar bases abertas já
anotadas (Roboflow Universe) com fotos brasileiras do Mapillary anotadas pela equipe. Detalhes e licenças
em `datasets/README.md`.

## Divisão treino/validação/teste (sem vazamento)
1. **Por grupo, não por imagem.** Cópias aumentadas do Roboflow (`foto_jpg.rf.<hash>`), quadros da mesma
   sequência Mapillary e quadros consecutivos do RDD2022 formam um grupo e vão juntos para o mesmo split.
2. **Quase-duplicatas.** dHash de 64 bits; imagens com distância de Hamming ≤ 6 têm seus grupos unidos, mesmo
   vindas de bases diferentes (bases do Roboflow às vezes reaproveitam imagens umas das outras).
3. **Estratificação por fonte e classe.** Cada fonte é dividida separadamente em 70/15/15, equilibrando o
   número de caixas de cada classe entre os splits.
4. **Teste intocado.** O teste não é usado para escolher modelo nem hiperparâmetros; só para a avaliação final.
5. **test_br.** Subconjunto do teste com fotos brasileiras: mede generalização para o contexto real do projeto.
Verificação automática: `split_report.json` registra quantos grupos aparecem em mais de um split (deve ser 0)
e `check_dataset.py` procura imagens idênticas (MD5) entre splits.

## Treinamento (`configs/train.yaml`)
| Parâmetro | Valor | Justificativa |
|---|---|---|
| Pesos iniciais | `yolo26s.pt` (COCO) | transferência de aprendizado; dataset pequeno |
| Épocas | até 100, `patience 20` | early stopping evita sobreajuste e economiza tempo de GPU |
| Tamanho da imagem | 640 px | padrão do modelo; equilíbrio entre detalhe e memória |
| Batch | 16 | cabe na T4 de 16 GB; fallback 8 |
| Otimizador | AdamW, lr0 = 0,001, weight decay 5e-4, cosseno | ajuste fino estável em poucos dados |
| Aumentos | mosaic, flip horizontal, HSV (padrões da Ultralytics), mosaic desligado nas 10 últimas épocas | robustez a iluminação e enquadramento |
| Semente | 42, `deterministic: true` | reprodutibilidade |
| Seleção do modelo final | melhor época na **validação** (`best.pt`, critério *fitness* da Ultralytics, que pondera mAP50 e mAP50-95) | teste não participa da escolha |

## Avaliação
`src/evaluate.py` roda no teste e no `test_br`: Precision, Recall, F1 = 2PR/(P+R), mAP@50, mAP@50-95 por
classe e geral, matriz de confusão (conf 0,25; IoU 0,45), erros mais frequentes e tempo de inferência
(tempos da Ultralytics + latência ponta a ponta em GPU e CPU, batch 1).
