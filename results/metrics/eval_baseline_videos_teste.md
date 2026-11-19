## Avaliação — baseline_videos_teste

Imagens: 72 | Instâncias: 141 | Modelo: `baseline_publico.pt`

| Classe | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| desnivel | 0.000 | 0.000 | 0.000 | 0.009 | 0.003 |
| obstaculo | 0.429 | 0.349 | 0.384 | 0.280 | 0.098 |
| **Geral (média)** | 0.214 | 0.174 | 0.192 | 0.144 | 0.051 |

Tempo por imagem (Ultralytics, GPU): pré 0.7 ms + inferência 8.1 ms + pós 0.8 ms
Latência ponta a ponta (GPU NVIDIA GeForce RTX 4050 Laptop GPU, batch 1, 50 imagens): média 16.0 ms, mediana 15.8 ms, p95 17.0 ms (~62.4 FPS)

### Erros mais frequentes

| Real | Previsto | Qtde | Tipo |
|---|---|---|---|
| fundo | obstaculo | 785 | falso positivo (detectou onde não havia nada) |
| fundo | buraco | 564 | falso positivo (detectou onde não havia nada) |
| fundo | desnivel | 239 | falso positivo (detectou onde não havia nada) |
| obstaculo | fundo | 39 | falso negativo (não detectou) |
| desnivel | fundo | 26 | falso negativo (não detectou) |
| desnivel | buraco | 2 | confusão entre classes |
| obstaculo | buraco | 2 | confusão entre classes |
| obstaculo | desnivel | 2 | confusão entre classes |
