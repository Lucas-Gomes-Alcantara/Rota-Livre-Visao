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

Quantidades de imagens: as páginas do Roboflow não informam o total de todas as bases; os números reais
obtidos ficam em `results/metrics/convert_report.json` depois do download e entram no relatório.

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
