## Avaliação — test

Imagens: 368 | Instâncias: 950 | Modelo: `final.pt`

| Classe | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| buraco | 0.723 | 0.687 | 0.704 | 0.719 | 0.460 |
| desnivel | 0.358 | 0.150 | 0.211 | 0.138 | 0.063 |
| obstaculo | 0.809 | 0.719 | 0.761 | 0.780 | 0.539 |
| **Geral (média)** | 0.630 | 0.519 | 0.569 | 0.546 | 0.354 |

Tempo por imagem (Ultralytics, GPU): pré 0.8 ms + inferência 11.5 ms + pós 0.7 ms
Latência ponta a ponta (GPU NVIDIA GeForce RTX 4050 Laptop GPU, batch 1, 50 imagens): média 21.7 ms, mediana 20.2 ms, p95 45.7 ms (~46.0 FPS)
Latência ponta a ponta (cpu, batch 1, 20 imagens): média 102.2 ms, mediana 101.7 ms, p95 108.0 ms (~9.8 FPS)

### Erros mais frequentes

| Real | Previsto | Qtde | Tipo |
|---|---|---|---|
| fundo | obstaculo | 4100 | falso positivo (detectou onde não havia nada) |
| fundo | buraco | 2091 | falso positivo (detectou onde não havia nada) |
| fundo | desnivel | 655 | falso positivo (detectou onde não havia nada) |
| obstaculo | fundo | 68 | falso negativo (não detectou) |
| buraco | fundo | 26 | falso negativo (não detectou) |
| desnivel | fundo | 20 | falso negativo (não detectou) |
| obstaculo | desnivel | 1 | confusão entre classes |
