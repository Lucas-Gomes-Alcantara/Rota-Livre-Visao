# Roteiro do vídeo de demonstração (4–6 min)

Gravar a tela executando de verdade. Usar só os vídeos de teste (nunca vistos no treino): `125352`, `124907`, `124457`.

| # | Tempo | O que mostrar | Fala (resumo) |
|---|---|---|---|
| 1 | 0:00–0:40 | trecho de um vídeo bruto da calçada | Problema: buracos, degraus e obstáculos causam quedas de pessoas com mobilidade reduzida e deficiência visual. A ferramenta avisa antes. |
| 2 | 0:40–1:30 | README / `docs/decisoes_tecnicas.md` | YOLO26s; 3 classes; treino em 2 fases (bases públicas + vídeos próprios anotados e revisados); divisão por vídeo sem vazamento. |
| 3 | 1:30–3:00 | **execução ao vivo** no terminal: `python src/alerta.py --source "Videos RUA\20261119_125352.mp4"`; depois abrir o mp4 gerado com áudio | Corredor verde = caminho à frente; caixas coloridas = perigo; cinza = fora do caminho; banner + voz "Atenção, desnível perto". |
| 4 | 3:00–3:40 | (opcional) `--source 0 --live` apontando a câmera para o chão/um objeto | Funciona ao vivo; tempo por quadro na tela. |
| 5 | 3:40–4:40 | `results/metrics/eval_videos_teste.md`, matriz de confusão, comparação com baseline | Precision, Recall, F1, mAP@50 por classe; ganho da fase 2; tempo de inferência. |
| 6 | 4:40–5:20 | — | Limitações (buraco raro nos vídeos, distância qualitativa, sombras) e próximos passos. |
