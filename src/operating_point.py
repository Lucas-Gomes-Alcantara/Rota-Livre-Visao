"""Mede o detector no LIMIAR DE OPERAÇÃO do sistema de alerta (conf 0,35), não só nas curvas do mAP.

Casa previsões e rótulos por classe (IoU >= 0,5, guloso por confiança) e reporta TP, FP, FN, Precision,
Recall e F1 por classe, além de quadros com pelo menos um falso alarme. Também conta quadros: um quadro
"com barreira" (rótulo presente) em que o modelo detectou a mesma classe.

Uso: python src/operating_point.py --weights models/final.pt --list data/processed/test_br.txt --tag videos_teste
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import ROOT, load_classes, read_yolo_labels, save_json


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/final.pt"))
    ap.add_argument("--list", default=str(ROOT / "data/processed/test_br.txt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tag", default="videos_teste")
    args = ap.parse_args()
    from ultralytics import YOLO
    classes = load_classes()
    model = YOLO(args.weights)
    imgs = [Path(l) for l in Path(args.list).read_text(encoding="utf-8").split() if l]
    tp, fp, fn = Counter(), Counter(), Counter()
    frames_fa = 0
    frame_hit, frame_has = Counter(), Counter()
    for p in imgs:
        gts = [(classes[c], (x - w / 2, y - h / 2, x + w / 2, y + h / 2))
               for c, x, y, w, h in read_yolo_labels(p.parent.parent / "labels" / f"{p.stem}.txt")]
        r = model.predict(str(p), conf=args.conf, verbose=False)[0]
        preds = sorted([(r.names[int(c)], tuple(b), cf) for b, c, cf in
                        zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())], key=lambda t: -t[2])
        used = set()
        any_fa = False
        for cls, box, _ in preds:
            best, bi = 0, None
            for i, (gc, gb) in enumerate(gts):
                if gc == cls and i not in used and iou(box, gb) > best:
                    best, bi = iou(box, gb), i
            if bi is not None and best >= args.iou:
                used.add(bi)
                tp[cls] += 1
            else:
                fp[cls] += 1
                any_fa = True
        for i, (gc, _) in enumerate(gts):
            if i not in used:
                fn[gc] += 1
        frames_fa += any_fa
        for c in classes:
            if any(g[0] == c for g in gts):
                frame_has[c] += 1
                if any(pc == c for pc, _, _ in preds):
                    frame_hit[c] += 1
    rows = []
    for c in classes:
        if tp[c] + fn[c] == 0 and fp[c] == 0:
            continue
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        R = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        rows.append({"classe": c, "TP": tp[c], "FP": fp[c], "FN": fn[c], "precision": round(P, 3), "recall": round(R, 3),
                     "f1": round(F, 3), "quadros_com_a_classe": frame_has[c], "quadros_em_que_foi_detectada": frame_hit[c]})
    res = {"modelo": Path(args.weights).name, "conf": args.conf, "iou": args.iou, "imagens": len(imgs),
           "quadros_com_algum_falso_alarme": frames_fa, "por_classe": rows}
    save_json(res, ROOT / f"results/metrics/operacao_{args.tag}.json")
    L = [f"## Limiar de operação (conf {args.conf}, IoU {args.iou}) — {args.tag} — `{res['modelo']}`", "",
         "| Classe | TP | FP | FN | Precision | Recall | F1 | Quadros com a classe | Quadros detectados |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r_ in rows:
        L.append(f"| {r_['classe']} | {r_['TP']} | {r_['FP']} | {r_['FN']} | {r_['precision']:.3f} | {r_['recall']:.3f} | "
                 f"{r_['f1']:.3f} | {r_['quadros_com_a_classe']} | {r_['quadros_em_que_foi_detectada']} |")
    L.append("")
    L.append(f"Quadros com pelo menos um falso alarme: {frames_fa} de {len(imgs)}")
    md = "\n".join(L) + "\n"
    (ROOT / f"results/metrics/operacao_{args.tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
