# Rota Livre — Detecção de barreiras de acessibilidade em calçadas

Atividade **Applied Computer Vision (FIAP)** do projeto Rota Livre, plataforma CitizenTech para mapear
barreiras que dificultam a circulação de pedestres, especialmente pessoas com deficiência, idosos e pessoas
com mobilidade reduzida.

Treinamos um detector de objetos (**Ultralytics YOLO26s**, a partir de pesos COCO) para encontrar três
classes em fotos de calçadas, devolvendo caixa delimitadora, classe e confiança:

| Classe | O que é |
|---|---|
| `buraco` | falha ou ausência de material na superfície |
| `desnivel` | placa levantada, raiz, afundamento, degrau na faixa de circulação |
| `obstaculo` | objeto que reduz a largura livre de passagem |

Critérios completos de anotação: [`docs/annotation_guide.md`](docs/annotation_guide.md).

## Resultados
> Preenchido com os números **reais** depois do treino — gerados em `results/metrics/eval_test.md` e
> `results/metrics/eval_test_br.md`. Nenhuma métrica é escrita à mão.

## Como reproduzir (Google Colab gratuito)
1. Os notebooks já clonam https://github.com/Lucas-Gomes-Alcantara/Rota-Livre-Visao.
2. Crie as chaves gratuitas e salve em *Secrets* do Colab: `ROBOFLOW_API_KEY`, `MAPILLARY_TOKEN`
   (passo a passo em [`datasets/README.md`](datasets/README.md)).
3. Rode na ordem:
   - `notebooks/01_exploracao.ipynb` — baixa e converte as bases; baixa fotos brasileiras para anotar.
   - *(anotação no Roboflow, seguindo o guia)*
   - `notebooks/02_preparacao.ipynb` — divisão sem vazamento, verificação e inspeção visual.
   - `notebooks/03_treinamento.ipynb` — **GPU T4**: treino, avaliação, demonstração e interface ao vivo.

### Linha de comando (qualquer máquina com Python ≥ 3.10)
```bash
pip install -r requirements.txt
python src/download_data.py roboflow          # bases públicas -> data/raw/
python src/convert.py                         # -> data/interim/ + manifest.csv
python src/split.py                           # -> data/processed/ + configs/data.yaml
python src/check_dataset.py --strict          # qualidade e vazamento
python src/visualize_labels.py                # figuras para conferir anotações
python src/train.py                           # treino (GPU recomendada)
python src/evaluate.py --bench-cpu            # métricas no teste
python src/predict.py --source foto.jpg       # demonstração por linha de comando
python src/demo_app.py                        # demonstração com interface web
```

### Teste de funcionamento do pipeline (sem GPU e sem internet)
```bash
python tests/test_pipeline.py
```
Gera dados sintéticos e confere conversão VOC/YOLO, remapeamento de classes, descarte de imagem corrompida,
agrupamento de cópias aumentadas, união de quase-duplicatas entre bases, ausência de vazamento entre splits
e detecção de rótulos inválidos.

## Estrutura
```
configs/      classes.yaml · sources.yaml (bases + mapeamento de classes) · train.yaml (hiperparâmetros)
datasets/     README.md — origem, licença e como obter cada base
docs/         guia de anotação · decisões técnicas · roteiro do vídeo
notebooks/    01_exploracao · 02_preparacao · 03_treinamento (Colab)
src/          download_data · mapillary_fetch · convert · split · check_dataset · visualize_labels
              train · evaluate · predict · demo_app · draw · common
tests/        test_pipeline.py
results/      metrics/ (JSON, CSV, MD) · figures/ · predictions/
```

## Decisões técnicas
Resumo em [`docs/decisoes_tecnicas.md`](docs/decisoes_tecnicas.md): por que YOLO26s, licença, divisão dos
dados por grupo + deduplicação perceptual, hiperparâmetros e critério de seleção do modelo.

## Licença e créditos
- Código sob **AGPL-3.0** (exigência da Ultralytics YOLO para projetos abertos).
- Dados de terceiros **não** são redistribuídos aqui. Créditos e licenças de cada base em
  [`datasets/README.md`](datasets/README.md); fotos do Mapillary sob CC BY-SA 4.0 com autor por imagem.
- Pesos do modelo treinado: publicados como *Release* do GitHub (arquivo `best.pt`).

## Integrantes
_(preencher)_
