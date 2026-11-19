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
Nenhuma base pública cobre as três classes com caixas em calçadas brasileiras vistas em primeira pessoa.
Estratégia final (19/11/2026):
1. **Fase 1 — bases públicas** (Roboflow Universe): buracos (665 imagens), calçadas quebradas (50) e
   obstáculos de calçada (1.261). Dão volume, mas são de outro domínio (rua, fotos de produto).
2. **Fase 2 — vídeos próprios**: 13 vídeos gravados pela equipe com o celular em primeira pessoa, em
   calçadas reais, com sol e sombra. Extraído 1 quadro por segundo (292 quadros, 720×1280).
   - **Pré-anotação** com um detector de vocabulário aberto (YOLOE-11L, prompts como "trash bag", "pole",
     "raised paving stone") e **revisão manual de todos os quadros usados** (`data/raw/videos_calcada/revisao.yaml`):
     propostas erradas removidas, caixas faltantes adicionadas (degraus, bordas de laje, postes).
   - Regras automáticas antes da revisão: descartar propostas que ocupam > 35% do quadro e buraco/desnível
     com confiança < 0,35 (o modelo de vocabulário aberto confunde "piso" inteiro com "piso quebrado").
   - Resultado: 184 quadros revisados, 72 caixas de desnível, 246 de obstáculo (buraco quase não aparece nos vídeos).
   - O Mapillary foi abandonado: as fotos são de câmera de carro e quase não mostram a calçada.

## Por que treinar em duas fases
O modelo da fase 1 aprende o conceito geral das classes com mais dados; a fase 2 (ajuste fino partindo dos
pesos da fase 1, quadros de vídeo repetidos 4× no treino) adapta ao domínio real: câmera de celular,
perspectiva de pedestre, calçadas brasileiras, sombras fortes. A comparação fase 1 × fase 2 nos mesmos vídeos
de teste mede o ganho da adaptação.

## Divisão treino/validação/teste (sem vazamento)
1. **Por grupo, não por imagem.** Cópias aumentadas do Roboflow (`foto_jpg.rf.<hash>`), quadros da mesma
   sequência Mapillary e quadros consecutivos do RDD2022 formam um grupo e vão juntos para o mesmo split.
2. **Quase-duplicatas.** dHash de 64 bits; imagens com distância de Hamming ≤ 6 têm seus grupos unidos, mesmo
   vindas de bases diferentes (bases do Roboflow às vezes reaproveitam imagens umas das outras).
3. **Estratificação por fonte e classe.** Cada fonte é dividida separadamente em 70/15/15, equilibrando o
   número de caixas de cada classe entre os splits.
4. **Teste intocado.** O teste não é usado para escolher modelo nem hiperparâmetros; só para a avaliação final.
5. **Vídeos de teste.** Três vídeos inteiros (`124457`, `124907`, `125352`, ≈72 s, com sol e sombra) só no teste;
   usados nas métricas `videos_teste` e nas demonstrações de alerta — o modelo nunca viu nenhum quadro deles.
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
