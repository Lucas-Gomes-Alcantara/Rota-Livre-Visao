"""Gera figuras para conferir a qualidade das anotações e a distribuição das classes.

Saída:
  results/figures/anotacoes_<split>_<classe>.jpg   (grade de exemplos com as caixas desenhadas)
  results/figures/distribuicao_classes.png         (instâncias por classe e split)

Uso: python src/visualize_labels.py --per-class 12
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import IMG_EXTS, ROOT, load_classes, read_yolo_labels
from draw import DISPLAY, draw_boxes, yolo_to_xyxy

SPLITS = ("train", "val", "test")


def collect(data_root: Path, split: str, classes):
    items = []
    for img in sorted((data_root / split / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        boxes = read_yolo_labels(data_root / split / "labels" / f"{img.stem}.txt")
        items.append((img, boxes))
    return items


def grid(items, classes, tile=320, cols=4):
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, max(1, rows) * tile), (245, 245, 245))
    for k, (img_path, boxes) in enumerate(items):
        with Image.open(img_path) as im:
            W, H = im.size
            dets = [{"cls": classes[c], "conf": None, "xyxy": yolo_to_xyxy(*b, W, H)} for c, *b in boxes]
            out = draw_boxes(im, dets)
        out.thumbnail((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * tile
        canvas.paste(out, (x + (tile - out.width) // 2, y + (tile - out.height) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/processed"))
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = load_classes()
    data_root = Path(args.data)
    out_dir = ROOT / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    counts = {s: [0] * len(classes) for s in SPLITS}
    for s in SPLITS:
        if not (data_root / s / "images").exists():
            continue
        items = collect(data_root, s, classes)
        for _, boxes in items:
            for c, *_ in boxes:
                counts[s][c] += 1
        if s not in args.splits:
            continue
        for ci, cname in enumerate(classes):
            with_c = [it for it in items if any(b[0] == ci for b in it[1])]
            if not with_c:
                print(f"{s}/{cname}: nenhuma imagem")
                continue
            sample = rng.sample(with_c, min(args.per_class, len(with_c)))
            path = out_dir / f"anotacoes_{s}_{cname}.jpg"
            grid(sample, classes).save(path, quality=90)
            print(f"salvo {path.relative_to(ROOT)} ({len(sample)} imagens)")

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(SPLITS)
    for i, s in enumerate(SPLITS):
        xs = [k + i * width for k in range(len(classes))]
        bars = ax.bar(xs, counts[s], width=width, label=s,
                      color=["#94a3b8", "#64748b", "#334155"][i])
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks([k + width for k in range(len(classes))])
    ax.set_xticklabels([DISPLAY.get(c, c) for c in classes])
    ax.set_ylabel("instâncias (caixas)")
    ax.set_title("Distribuição das classes por split")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "distribuicao_classes.png", dpi=150)
    print(f"salvo results/figures/distribuicao_classes.png")


if __name__ == "__main__":
    main()
