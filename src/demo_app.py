"""Interface web simples (Gradio) para a demonstração em vídeo.

Envie uma foto de calçada -> o modelo roda na hora -> aparecem as caixas, classes, confianças e o tempo.

Uso (Colab ou local):
  python src/demo_app.py            # no Colab, use --share para gerar um link público temporário
"""
from __future__ import annotations

import argparse
import time

from PIL import Image

from common import ROOT
from draw import DISPLAY, draw_boxes
from predict import detections_from_result


def build(weights: str):
    import gradio as gr
    from ultralytics import YOLO

    model = YOLO(weights)

    def infer(image: Image.Image, conf: float):
        if image is None:
            return None, [], "Envie uma imagem."
        t = time.perf_counter()
        r = model.predict(image, conf=conf, verbose=False)[0]
        total = (time.perf_counter() - t) * 1000
        dets = detections_from_result(r)
        table = [[DISPLAY.get(d["cls"], d["cls"]), round(d["conf"], 3),
                  "(" + ", ".join(str(round(v)) for v in d["xyxy"]) + ")"] for d in dets]
        info = (f"{len(dets)} detecção(ões) | inferência {r.speed['inference']:.1f} ms | "
                f"total {total:.0f} ms")
        return draw_boxes(image, dets), table, info

    with gr.Blocks(title="Rota Livre — detecção de barreiras em calçadas") as demo:
        gr.Markdown("## Rota Livre — detecção de barreiras em calçadas\n"
                    "Classes: **Buraco**, **Desnível**, **Obstáculo**. Modelo: YOLO26 treinado pela equipe.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Foto da calçada")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confiança mínima")
                btn = gr.Button("Detectar", variant="primary")
            with gr.Column():
                out = gr.Image(type="pil", label="Resultado")
                info = gr.Markdown()
                tbl = gr.Dataframe(headers=["classe", "confiança", "caixa (x1,y1,x2,y2)"], label="Detecções")
        btn.click(infer, [inp, conf], [out, tbl, info])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build(args.weights).launch(share=args.share)


if __name__ == "__main__":
    main()
