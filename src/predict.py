"""Demonstração: recebe imagem(ns) de calçada, executa o modelo e salva a imagem com caixas, classe e confiança.

Uso:
  python src/predict.py --source caminho/foto.jpg
  python src/predict.py --source pasta_com_fotos/ --conf 0.3
  python src/predict.py --source https://exemplo.com/calcada.jpg

Saída: results/predictions/<nome>_pred.jpg e results/predictions/<nome>_pred.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from common import IMG_EXTS, ROOT
from draw import DISPLAY, draw_boxes


def detections_from_result(r) -> list[dict]:
    names = r.names
    out = []
    for xyxy, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        out.append({"cls": names[int(c)], "conf": float(conf), "xyxy": tuple(round(v, 1) for v in xyxy)})
    return out


def run(model, source: str, conf: float, iou: float, imgsz: int, out_dir: Path, device=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(source, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, stream=True)
    saved = []
    for r in results:
        dets = detections_from_result(r)
        img = Image.fromarray(r.orig_img[..., ::-1])  # BGR -> RGB
        annotated = draw_boxes(img, dets)
        stem = Path(r.path).stem or "imagem"
        dst = out_dir / f"{stem}_pred.jpg"
        annotated.save(dst, quality=92)
        payload = {"imagem": r.path, "tempo_ms": {k: round(v, 1) for k, v in r.speed.items()},
                   "deteccoes": [{**d, "classe": DISPLAY.get(d["cls"], d["cls"])} for d in dets]}
        dst.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{r.path}  ({r.speed['inference']:.1f} ms de inferência)")
        if not dets:
            print("  nenhuma barreira detectada acima do limiar")
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            print(f"  {DISPLAY.get(d['cls'], d['cls']):10s} conf={d['conf']:.2f}  caixa=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        print(f"  -> {dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst}")
        saved.append(dst)
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="arquivo, pasta ou URL de imagem")
    ap.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima para mostrar uma detecção")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "results/predictions"))
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_dir() and not any(p.suffix.lower() in IMG_EXTS for p in src.iterdir()):
        raise SystemExit(f"Nenhuma imagem em {src}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    run(model, args.source, args.conf, args.iou, args.imgsz, Path(args.out), args.device)


if __name__ == "__main__":
    main()
