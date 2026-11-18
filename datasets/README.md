# Dados — como obter e condições de uso

As imagens **não são publicadas neste repositório** (tamanho e licenças). Os scripts baixam tudo para
`data/raw/` (ignorado pelo Git). A versão exata baixada de cada base fica em `data/raw/<base>/SOURCE.txt`.

| Base | Origem / link | Licença | Classes originais → Rota Livre | Formato | Uso |
|---|---|---|---|---|---|
| Pothole Dataset (Chitholian) | [Roboflow](https://universe.roboflow.com/pothole-skc/pothole-kvso8) (original no Kaggle) | ODbL v1.0 | pothole → `buraco` | YOLO | treino/val/teste |
| Sidewalk Walkability | [Roboflow](https://universe.roboflow.com/sidewalk-detection/sidewalk-walkability) | CC BY 4.0 | broken → `desnivel` (provisório*), obstruction → `obstaculo` | YOLO | treino/val/teste |
| Sidewalk Obstacles (Blindspot) | [Roboflow](https://universe.roboflow.com/blindspot-fmcu0/sidewalk-obstacles) | CC BY 4.0 | pole, ashcan, cone, roadblock, warning_column, fire_hydrant → `obstaculo`; veículos/pessoas descartados | YOLO | treino/val/teste |
| Mapillary (cidades brasileiras) | [mapillary.com](https://www.mapillary.com) via API | CC BY-SA 4.0 (atribuição por foto em `metadata.csv`) | anotadas pela equipe | YOLO (export Roboflow) | treino/val/**test_br** |
| RDD2022 (opcional) | [figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) · [GitHub](https://github.com/sekilab/RoadDamageDetector) | **a confirmar** (`download_data.py rdd2022 --list` mostra o que o figshare informa) | D40 → `buraco`; demais descartadas | PASCAL VOC | reforço de `buraco`, limitado a 800 imagens |
| PEDESTRIAN (opcional) | [Zenodo 10.5281/zenodo.10907945](https://doi.org/10.5281/zenodo.10907945) | **a confirmar** | sem caixas (rótulo por vídeo) | quadros de vídeo | só se a equipe anotar |

\* `broken` é um rótulo genérico de piso quebrado. Ele só entra como `desnivel` depois da inspeção visual em
`results/figures/anotacoes_train_desnivel.jpg`. Se a maior parte for buraco, troque o mapeamento em
`configs/sources.yaml` e documente a decisão.

### Números reais obtidos (download de 18/11/2026, `results/metrics/convert_report.json`)

| Base | Imagens na base | Imagens aproveitadas | Caixas Rota Livre |
|---|---|---|---|
| Pothole (Chitholian) | 665 | 665 | 1.739 `buraco` |
| Sidewalk Walkability | 50 | 50 | 52 `desnivel` · 18 `obstaculo` |
| Sidewalk Obstacles (Blindspot) | 3.730 | 1.261 (só as que têm obstáculo mapeado) | 3.569 `obstaculo` |

Divisão (`split_report.json`): treino 1.384 · validação 296 · teste 296 imagens; 0 grupos em mais de um split;
5 quase-duplicatas uniram grupos. **`desnivel` tem só 35 instâncias no treino** (mínimo recomendado 150): por isso
a anotação das fotos brasileiras do Mapillary é necessária.

Inspeção visual (`results/figures/anotacoes_train_*.jpg`):
- `buraco`: anotações consistentes, mas fotos de **rua** (domínio diferente de calçada).
- `desnivel` (rótulo *broken*): calçadas reais com piso quebrado/ausente — compatível com "irregularidade"; mapeamento mantido.
- `obstaculo`: ruidoso — inclui fotos de produto em fundo branco e postes no meio da via.

### Bases sem versão publicada (Pothole e Sidewalk Obstacles)
O Roboflow só permite baixar versões. Essas duas não têm nenhuma, então é preciso clicar em **Fork Dataset**
na página do Universe (copia para o seu workspace). `download_data.py roboflow` detecta o fork na conta da chave,
gera uma versão sem augmentation (só auto-orient) e baixa. Forks usados neste projeto:
`lucass-workspace-shzef/pothole-kvso8-jwzbd` e `lucass-workspace-shzef/sidewalk-obstacles-7mfgc` (versão 1).

## Passo a passo
1. Chave gratuita do Roboflow: app.roboflow.com → Settings → API Keys. No Colab, salve em *Secrets* como `ROBOFLOW_API_KEY`.
2. Token gratuito do Mapillary: mapillary.com/dashboard/developers → crie um app → *Client Token*. Salve como `MAPILLARY_TOKEN`.
3. Rode o notebook `notebooks/01_exploracao.ipynb` (ou os comandos abaixo).

```bash
python src/download_data.py roboflow
python src/mapillary_fetch.py --city sao_paulo rio_de_janeiro belo_horizonte --n 300 --require-sidewalk
# anotar no Roboflow (docs/annotation_guide.md), depois:
python src/download_data.py roboflow-own --workspace SEU_WS --project SEU_PROJETO --version 1
python src/convert.py
```

## Atribuição exigida
- **CC BY / CC BY-SA:** cite cada base (autor, link, licença) no README e no PDF. Para o Mapillary, a lista
  de autores e links por imagem está em `data/raw/mapillary_br/metadata.csv`; imagens mostradas no PDF/vídeo
  devem ter a atribuição na legenda (ex.: "Foto: <usuário> / Mapillary, CC BY-SA 4.0").
- **ODbL:** citar a base e manter derivados do banco de dados sob ODbL se forem redistribuídos.

## Excluídas
- **SideGuide:** termos restringem a uso de pesquisa não comercial e proíbem redistribuição.
- **SideSeeing (USP):** calçadas brasileiras, mas sem caixas delimitadoras e sem licença explícita dos dados.
- **Google Street View / Project Sidewalk:** as imagens pertencem ao Google e seguem os termos do Google Maps,
  não uma licença aberta; evitadas para não depender de permissão de uso.
