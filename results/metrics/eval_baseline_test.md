## Avaliação — baseline_test

Imagens: 368 | Instâncias: 950 | Modelo: `baseline_publico.pt`

| Classe | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| buraco | 0.676 | 0.729 | 0.702 | 0.769 | 0.490 |
| desnivel | 0.512 | 0.100 | 0.167 | 0.072 | 0.035 |
| obstaculo | 0.823 | 0.648 | 0.725 | 0.732 | 0.499 |
| **Geral (média)** | 0.670 | 0.492 | 0.568 | 0.524 | 0.341 |

Tempo por imagem (Ultralytics, GPU): pré 0.8 ms + inferência 11.6 ms + pós 0.8 ms
Latência ponta a ponta (GPU NVIDIA GeForce RTX 4050 Laptop GPU, batch 1, 50 imagens): média 21.6 ms, mediana 20.1 ms, p95 45.3 ms (~46.2 FPS)

### Erros mais frequentes

| Real | Previsto | Qtde | Tipo |
|---|---|---|---|
| fundo | obstaculo | 5663 | falso positivo (detectou onde não havia nada) |
| fundo | buraco | 3651 | falso positivo (detectou onde não havia nada) |
| fundo | desnivel | 897 | falso positivo (detectou onde não havia nada) |
| obstaculo | fundo | 86 | falso negativo (não detectou) |
| desnivel | fundo | 27 | falso negativo (não detectou) |
| buraco | fundo | 10 | falso negativo (não detectou) |
| obstaculo | desnivel | 3 | confusão entre classes |
| desnivel | buraco | 2 | confusão entre classes |
