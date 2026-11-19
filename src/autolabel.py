"""Pré-anotação automática dos quadros dos vídeos com um detector de vocabulário aberto (YOLOE / YOLO-World).

As caixas propostas NÃO são usadas diretamente: passam por revisão (src/review_labels.py + correções
manuais em data/raw/videos_calcada/revisao.yaml). Isso é documentado no relatório como pseudo-rotulagem revisada.

Saída: data/raw/videos_calcada/proposals.json  {imagem: [{prompt, classe, conf, xyxy_norm}]}
Uso:   python src/autolabel.py --conf 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

# prompt (inglês, vocabulário do modelo) -> classe Rota Livre
PROMPTS = {
    "trash bag": "obstaculo", "garbage bag": "obstaculo", "plastic bag": "obstaculo",
    "trash can": "obstaculo", "utility pole": "obstaculo", "pole": "obstaculo",
    "traffic cone": "obstaculo", "bollard": "obstaculo", "fire hydrant": "obstaculo",
    "tree trunk": "obstaculo", "cardboard box": "obstaculo", "rubble": "obstaculo",
    "construction barrier": "obstaculo", "sign post": "obstaculo", "bicycle": "obstaculo",
    "motorcycle": "obstaculo", "plant pot": "obstaculo",
    "pothole": "buraco", "hole in the pavement": "buraco",
    "broken pavement": "desnivel", "step": "desnivel", "uneven sidewalk": "desnivel",
    "raised paving stone": "desnivel",
}


def load_model():
    from ultralytics import YOLOE, YOLO
    names = list(PROMPTS)
    for w in ("yoloe-11l-seg.pt", "yoloe-11m-seg.pt"):
        try:
            m = YOLOE(w)
            m.set_classes(names, m.get_text_pe(names))
            return m, w
        except Exception as e:
            print(f"falha com {w}: {e}")
    m = YOLO("yolov8x-worldv2.pt")
    m.set_classes(names)
    return m, "yolov8x-worldv2.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data/raw/videos_calcada/images"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "data/raw/videos_calcada/proposals.json"))
    args = ap.parse_args()
    model, wname = load_model()
    print("modelo de pré-anotação:", wname)
    names = list(PROMPTS)
    out = {"modelo": wname, "conf": args.conf, "imagens": {}}
    imgs = sorted(Path(args.images).glob("*.jpg"))
    for k, p in enumerate(imgs):
        r = model.predict(str(p), conf=args.conf, imgsz=960, verbose=False)[0]
        dets = []
        for xyxyn, c, cf in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            prompt = names[int(c)]
            dets.append({"prompt": prompt, "classe": PROMPTS[prompt], "conf": round(cf, 3),
                         "xyxy": [round(v, 4) for v in xyxyn]})
        out["imagens"][p.name] = dets
        if k % 25 == 0:
            print(f"{k}/{len(imgs)} {p.name}: {len(dets)} propostas", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
