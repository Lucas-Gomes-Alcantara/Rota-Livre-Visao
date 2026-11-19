## Avaliação — videos_teste

Imagens: 72 | Instâncias: 141 | Modelo: `final.pt`

| Classe | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| desnivel | 0.539 | 0.219 | 0.311 | 0.185 | 0.082 |
| obstaculo | 0.673 | 0.700 | 0.686 | 0.684 | 0.391 |
| **Geral (média)** | 0.606 | 0.459 | 0.523 | 0.435 | 0.237 |

Tempo por imagem (Ultralytics, GPU): pré 0.7 ms + inferência 8.1 ms + pós 1.0 ms
Latência ponta a ponta (GPU NVIDIA GeForce RTX 4050 Laptop GPU, batch 1, 50 imagens): média 15.9 ms, mediana 15.8 ms, p95 16.7 ms (~62.9 FPS)

### Erros mais frequentes

| Real | Previsto | Qtde | Tipo |
|---|---|---|---|
| fundo | obstaculo | 629 | falso positivo (detectou onde não havia nada) |
| fundo | desnivel | 185 | falso positivo (detectou onde não havia nada) |
| fundo | buraco | 42 | falso positivo (detectou onde não havia nada) |
| desnivel | fundo | 20 | falso negativo (não detectou) |
| obstaculo | fundo | 14 | falso negativo (não detectou) |
