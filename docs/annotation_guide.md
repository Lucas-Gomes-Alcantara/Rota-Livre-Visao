# Guia de anotação — Rota Livre

Critérios objetivos para que pessoas diferentes anotem do mesmo jeito. Sem isso, *desnível* se confunde
com *buraco* e o modelo aprende ruído. Referência de acessibilidade: ABNT NBR 9050 (faixa livre de circulação).

## Classes

### `buraco`
**É:** cavidade ou ausência de material na superfície da calçada, com borda visível — piso faltando,
cratera, tampa de caixa de inspeção ausente ou quebrada deixando abertura.
**Não é:** mancha, sujeira, poça sobre piso íntegro, fissura fina sem abertura, grelha/bueiro íntegro.
**Caixa:** envolve toda a cavidade, justa à borda.

### `desnivel`
**É:** descontinuidade vertical na faixa de circulação — placa ou ladrilho levantado, raiz que ergue o piso,
afundamento de placa, degrau isolado no meio da calçada, emenda com ressalto, pedras portuguesas soltas
formando ondulação, piso quebrado com fragmentos em alturas diferentes.
**Não é:** meio-fio (transição esperada calçada → rua), rampa (classe futura), escada de acesso a imóvel
fora da faixa de circulação, fissura plana sem diferença de altura.
**Caixa:** cobre toda a região irregular (pode ser larga e baixa).
**Regra de desempate com `buraco`:** se o problema principal é *falta* de piso → `buraco`;
se é piso *presente em alturas diferentes* → `desnivel`.

### `obstaculo`
**É:** objeto físico sobre a calçada que reduz a largura livre de passagem — poste, lixeira, cone,
entulho, saco de lixo, placa/cavalete, tapume, mesa e cadeira, veículo estacionado sobre a calçada,
vegetação invadindo a passagem, galho caído.
**Não é:** pessoas e animais; veículos na rua; objetos totalmente fora da calçada.
**Caixa:** envolve o objeto inteiro visível.

## Regras gerais
1. Caixas justas ao objeto; sem margem extra.
2. Objeto parcialmente encoberto: anote a parte visível se ≥ 30% do objeto estiver visível.
3. Ignore ocorrências muito pequenas (menos de ~15 px de lado) ou muito distantes — não dá para verificar.
4. **Imagem sem calçada visível → exclua a imagem** (não anote como negativa).
5. **Calçada visível e sem problemas → mantenha sem caixas.** Essas imagens negativas reduzem falsos positivos.
6. Na dúvida entre duas classes, siga as regras de desempate acima; se continuar em dúvida, não anote e
   registre o id da imagem em `docs/duvidas_anotacao.md`.

## Fluxo no Roboflow (gratuito)
1. Crie um projeto **Object Detection** com exatamente as classes `buraco`, `desnivel`, `obstaculo`.
2. Envie as imagens de `mapillary_br_para_anotar.zip` (gerado pelo notebook 01). **Não altere os nomes dos
   arquivos** — o id do Mapillary no nome é usado para agrupar por sequência e evitar vazamento.
3. Anote seguindo este guia. Divida as imagens entre os integrantes; peça para uma segunda pessoa revisar ~10%.
4. Em *Generate*: **sem augmentation** (ela é feita no treino) e sem redimensionar. O split do Roboflow é
   ignorado — nosso `split.py` refaz a divisão por grupo.
5. Exporte em **YOLOv8** e baixe com `python src/download_data.py roboflow-own ...` (notebook 02).

## Meta de quantidade
~250–300 imagens anotadas. Com a divisão 70/15/15, ~40–45 imagens brasileiras ficam no teste (`test_br`)
para medir generalização. Mais imagens = métricas por classe mais confiáveis.
