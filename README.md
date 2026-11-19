# Rota Livre — Visão computacional para alerta de barreiras na calçada

Atividade **Applied Computer Vision (FIAP)** do projeto Rota Livre.

**▶ Vídeo de demonstração:** https://youtu.be/zDZmD35m8LE · **Relatório:** [`docs/Rota_Livre_Relatorio.pdf`](docs/Rota_Livre_Relatorio.pdf)

**Integrantes:** Lucas Gomes Alcântara (RM98766) · Gabriel Gomes Catanzaro (RM93445) · Henrique Canali Cuzato (RM551326)

Ferramenta que analisa vídeo em primeira pessoa da calçada, **detecta buracos, desníveis e obstáculos**
que podem causar queda de pessoas com mobilidade reduzida ou deficiência visual (parcial ou total) e
**avisa por voz e na tela** quando a barreira está no caminho à frente.

| Classe | O que detecta |
|---|---|
| `buraco` | falha ou ausência de piso, abertura na superfície |
| `desnivel` | degrau, placa levantada/afundada, borda de laje, ressalto, piso quebrado em alturas diferentes |
| `obstaculo` | poste, saco/caixa de lixo, lixeira, cone, entulho, vegetação ou objeto que estreita a passagem |

## Como funciona
1. **Detector** — Ultralytics **YOLO26s** treinado em duas fases:
   (a) bases públicas (buracos, calçadas quebradas, obstáculos) e (b) ajuste fino com **quadros dos vídeos
   gravados pela equipe** em calçadas reais (sol e sombra), anotados e revisados.
2. **Sistema de alerta** (`src/alerta.py`) — filtra só o que está no **corredor de caminhada**, estima a
   distância (muito perto / perto / à frente), confirma a detecção em vários quadros seguidos e fala o
   aviso em português. Detalhes: [`docs/sistema_alerta.md`](docs/sistema_alerta.md).

## Resultados (19/11/2026)
Nos **3 vídeos de teste nunca vistos** (72 quadros, 141 objetos), o ajuste fino com os vídeos próprios
triplicou o mAP@50: **0,435** contra 0,144 do modelo treinado só com bases públicas.

| Modelo | Classe | Precision | Recall | F1 | mAP@50 |
|---|---|---|---|---|---|
| Fase 1 (só bases públicas) | geral | 0,214 | 0,174 | 0,192 | 0,144 |
| **Fase 2 (final)** | obstáculo | 0,673 | 0,700 | 0,686 | 0,684 |
| **Fase 2 (final)** | desnível | 0,539 | 0,219 | 0,311 | 0,185 |
| **Fase 2 (final)** | **geral** | 0,606 | 0,459 | 0,523 | **0,435** |

No limiar do alerta (conf 0,35): obstáculos percebidos em 59 de 66 quadros; desnível é o ponto fraco
(recall 0,16, precision 0,71). Inferência: 8,1 ms na RTX 4050 (≈63 FPS ponta a ponta), 102 ms em CPU.

Arquivos em `results/metrics/` (gerados por `src/evaluate.py` e `src/operating_point.py`):
- `eval_videos_teste.md` — **vídeos de teste nunca vistos no treino** (cenário real do projeto)
- `eval_test.md` — teste completo (bases públicas + vídeos)
- `eval_baseline_*.md` — mesmo teste com o modelo treinado só em dados públicos (comparação)

- `operacao_videos_teste.md` — acertos, falsos alarmes e perdidos no limiar real do alerta

Demonstrações (vídeos de teste, com voz) — veja o [vídeo de demonstração](https://youtu.be/zDZmD35m8LE) ou os arquivos: `results/demo/<video>_alerta.mp4` e a linha do tempo dos
avisos em `results/demo/<video>_eventos.csv`. Modelo treinado: `models/final.pt`.

## Reproduzir (Windows/Linux, GPU NVIDIA recomendada)
```bash
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt pyttsx3 imageio-ffmpeg

# 1) dados públicos (chaves em .env: ROBOFLOW_API_KEY) -> ver datasets/README.md
python src/download_data.py roboflow
# 2) quadros dos vídeos próprios (1 por segundo) + pré-anotação + revisão
ffmpeg -i "Videos RUA/<video>.mp4" -vf "fps=1,scale=720:-2" data/raw/videos_calcada/images/<video>_f%03d.jpg
python src/autolabel.py                    # propostas com detector de vocabulário aberto (YOLOE)
python src/review_labels.py sheets         # folhas de revisão em work/review/
python src/review_labels.py apply          # aplica data/raw/videos_calcada/revisao.yaml -> rótulos YOLO
# 3) conversão, divisão sem vazamento, verificação
python src/convert.py && python src/split.py && python src/check_dataset.py && python src/visualize_labels.py
# 4) treino (fase 1: públicos; fase 2: ajuste fino com vídeos)
python src/train.py --epochs 60 --batch 8 --name baseline_publico
python src/train.py --model models/baseline_publico.pt --epochs 40 --batch 8 --name final_videos
# 5) avaliação e demonstração
python src/evaluate.py --weights models/final.pt --data configs/data_test_br.yaml --tag videos_teste
python src/alerta.py --source "Videos RUA/20261119_125352.mp4" --weights models/final.pt
python src/alerta.py --source 0 --live --weights models/final.pt          # câmera ao vivo
```

## Dados e divisão sem vazamento
- Bases públicas (licenças e números): [`datasets/README.md`](datasets/README.md).
- **Vídeos próprios**: 13 vídeos (≈5 min, 1080p, celular em primeira pessoa, sol e sombra).
  **3 vídeos inteiros** (`124457`, `124907`, `125352`) ficam **só no teste**; os demais vão para treino/validação
  com um grupo por vídeo — quadros do mesmo vídeo nunca ficam em splits diferentes.
- Quase-duplicatas entre bases são unidas por hash perceptual antes da divisão.

> Os notebooks em `notebooks/` são da primeira versão (Google Colab, só bases públicas). O fluxo final roda
> localmente com os comandos acima.

## Estrutura
```
configs/   classes.yaml · sources.yaml (bases, mapeamento, vídeos de teste) · train.yaml
src/       download_data · convert · split · check_dataset · visualize_labels
           autolabel · review_labels (anotação dos vídeos) · train · evaluate · operating_point · predict · alerta · demo_app
docs/      sistema_alerta · decisoes_tecnicas · annotation_guide · roteiro_video
tests/     test_pipeline.py (pipeline de dados com dados sintéticos)
results/   metrics/ · figures/ · demo/
```

## Licença
Código sob **AGPL-3.0** (exigência da Ultralytics YOLO). Dados de terceiros não são redistribuídos aqui.
