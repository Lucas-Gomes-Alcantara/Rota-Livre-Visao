# Sistema de alerta (src/alerta.py)

Objetivo: avisar uma pessoa com mobilidade reduzida ou deficiência visual (parcial ou total) sobre barreiras
que podem causar queda **no caminho à frente**, sem avisar sobre tudo o que aparece na imagem.

## Fluxo por quadro
1. **Detecção** — o modelo YOLO26s treinado (`models/final.pt`) devolve caixas de `buraco`, `desnivel` e `obstaculo`.
2. **Corredor de caminhada** — trapézio fixo à frente do usuário (câmera no peito ou na mão, apontada para a frente):
   base ocupando 12%–88% da largura no rodapé do quadro, afunilando até 32%–68% a 40% da altura.
   Uma detecção só é **perigo** se ≥ 25% da metade inferior da caixa (onde o objeto toca o chão) estiver no corredor.
   Detecções fora do corredor aparecem em cinza e não geram aviso.
3. **Distância aproximada** — pela posição da base da caixa (sem calibração de câmera):
   base abaixo de 85% da altura = "muito perto"; 65%–85% = "perto"; acima = "à frente".
4. **Filtro temporal** — a classe precisa aparecer em ≥ 3 dos últimos 6 quadros (0,2 s a 30 fps) para virar alerta.
   Evita alertas piscando por detecções isoladas (falsos positivos de um quadro).
5. **Aviso** — banner colorido na tela + **voz em português** ("Atenção, desnível perto"), gerada pelo
   sintetizador do sistema (pyttsx3/SAPI). Sem voz disponível, usa bipes com tom diferente por classe.
   Um alerta novo é falado na hora; a mesma classe só é repetida após 4 s, e falas não se sobrepõem.

## Saídas
- `results/demo/<video>_alerta.mp4` — vídeo anotado com o áudio dos alertas (demonstração).
- `results/demo/<video>_eventos.csv` — linha do tempo dos alertas (tempo, classe, distância, confiança).
- Modo ao vivo: `python src/alerta.py --source 0 --live` (webcam/câmera do notebook ou celular via DroidCam).

## Parâmetros (linha de comando)
`--conf` (0,35), `--min-overlap` (0,25), `--n`/`--k` (6/3), `--repeat-s` (4,0), `--beep`.

## Limitações conhecidas
- Corredor e distância assumem câmera à frente na altura do peito/mão; com a câmera muito inclinada a distância perde sentido.
- Sem profundidade real (câmera monocular): a distância é qualitativa.
- O alerta é tão bom quanto o detector — ver métricas por classe em `results/metrics/`.
