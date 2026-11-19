"""Revisão das pré-anotações dos quadros dos vídeos e geração dos rótulos finais (YOLO).

1) `python src/review_labels.py sheets`  -> work/review/sheet_XX.jpg: quadros com as propostas numeradas
   e uma malha 10x10 (rótulos 0-9 nas bordas) para localizar caixas novas.
2) As decisões ficam em data/raw/videos_calcada/revisao.yaml:
     imagens:
       <arquivo.jpg>:
         remover: [0, 3]                 # índices das propostas (após deduplicação)
         trocar: {1: desnivel}           # muda a classe da proposta
         adicionar: [[desnivel, x1, y1, x2, y2]]   # coordenadas normalizadas 0-1
         descartar: true                 # tira o quadro do dataset (ex.: borrado)
         manter: [2]                     # mantém proposta que as regras automáticas removeriam
   Regras automáticas: remove propostas com área > --max-area e buraco/desnível com conf < --min-conf-terreno.
3) `python src/review_labels.py apply`  -> data/raw/videos_calcada/labels/*.txt + data.yaml

Propostas com confiança abaixo de --min-conf são ignoradas; caixas sobrepostas de prompts diferentes
(ex.: "trash bag" e "garbage bag") são unidas por NMS independente de classe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from common import ROOT
from draw import COLORS, get_font

BASE = ROOT / "data/raw/videos_calcada"
CLASSES = ["buraco", "desnivel", "obstaculo"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def dedup(dets, min_conf, thr=0.5):
    dets = sorted([d for d in dets if d["conf"] >= min_conf], key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        if all(iou(d["xyxy"], k["xyxy"]) < thr for k in keep):
            keep.append(d)
    return keep


def selected_images(all_names, every, test_videos):
    out = []
    for n in sorted(all_names):
        vid, fr = n.rsplit("_f", 1)
        k = int(fr.split(".")[0])
        if vid in test_videos or (k - 1) % every == 0:
            out.append(n)
    return out


def cmd_sheets(args, props):
    names = selected_images(props["imagens"], args.every, args.test_videos)
    out = ROOT / "work/review"
    out.mkdir(parents=True, exist_ok=True)
    tw, th, cols = 405, 720, 3
    font, small = get_font(22), get_font(15)
    sheets = []
    for s in range(0, len(names), cols * 2):
        chunk = names[s:s + cols * 2]
        sheet = Image.new("RGB", (cols * tw, 2 * (th + 28)), "white")
        for i, n in enumerate(chunk):
            im = Image.open(BASE / "images" / n).convert("RGB").resize((tw, th))
            d = ImageDraw.Draw(im)
            for g in range(1, 10):
                d.line([(g * tw / 10, 0), (g * tw / 10, th)], fill=(255, 255, 255), width=1)
                d.line([(0, g * th / 10), (tw, g * th / 10)], fill=(255, 255, 255), width=1)
                d.text((g * tw / 10 + 2, 2), str(g), fill=(255, 255, 0), font=small)
                d.text((2, g * th / 10 + 2), str(g), fill=(255, 255, 0), font=small)
            for j, p in enumerate(dedup(props["imagens"][n], args.min_conf)):
                x1, y1, x2, y2 = p["xyxy"]
                c = COLORS[p["classe"]]
                d.rectangle([x1 * tw, y1 * th, x2 * tw, y2 * th], outline=c, width=3)
                d.text((x1 * tw + 3, y1 * th + 2), f"{j}:{p['prompt'][:10]} {p['conf']:.2f}", fill=c, font=small,
                       stroke_width=2, stroke_fill=(0, 0, 0))
            x, y = (i % cols) * tw, (i // cols) * (th + 28)
            sheet.paste(im, (x, y + 28))
            ImageDraw.Draw(sheet).text((x + 4, y + 2), n.replace(".jpg", ""), fill=(0, 0, 0), font=font)
        p = out / f"sheet_{s // (cols * 2):02d}.jpg"
        sheet.save(p, quality=88)
        sheets.append(p.name)
    print(f"{len(names)} quadros em {len(sheets)} folhas: {out}")


def cmd_apply(args, props):
    rev_path = BASE / "revisao.yaml"
    rev = (yaml.safe_load(rev_path.read_text(encoding="utf-8")) or {}).get("imagens", {}) if rev_path.exists() else {}
    names = selected_images(props["imagens"], args.every, args.test_videos)
    lbl_dir = BASE / "labels"
    lbl_dir.mkdir(exist_ok=True)
    for old in lbl_dir.glob("*.txt"):
        old.unlink()
    stats = {"quadros": 0, "descartados": 0, "auto_aceitas": 0, "removidas": 0, "trocadas": 0, "adicionadas": 0}
    for n in names:
        r = rev.get(n) or {}
        if r.get("descartar"):
            stats["descartados"] += 1
            continue
        boxes = []
        rem = set(r.get("remover") or [])
        trocar = {int(k): v for k, v in (r.get("trocar") or {}).items()}
        manter = set(r.get("manter") or [])
        for j, p in enumerate(dedup(props["imagens"][n], args.min_conf)):
            if j in rem:
                stats["removidas"] += 1
                continue
            x1, y1, x2, y2 = p["xyxy"]
            area = (x2 - x1) * (y2 - y1)
            regra = area > args.max_area or (p["classe"] != "obstaculo" and p["conf"] < args.min_conf_terreno)
            if regra and j not in manter and j not in trocar:
                stats["removidas_por_regra"] = stats.get("removidas_por_regra", 0) + 1
                continue
            cls = trocar.get(j, p["classe"])
            stats["trocadas" if j in trocar else "auto_aceitas"] += 1
            boxes.append((cls, *p["xyxy"]))
        for a in r.get("adicionar") or []:
            boxes.append(tuple(a))
            stats["adicionadas"] += 1
        with open(lbl_dir / n.replace(".jpg", ".txt"), "w") as f:
            for cls, x1, y1, x2, y2 in boxes:
                f.write(f"{CLASSES.index(cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
        stats["quadros"] += 1
    (BASE / "data.yaml").write_text("names: [buraco, desnivel, obstaculo]\n", encoding="utf-8")
    # imagens não selecionadas ficam fora: move para images_nao_usadas não é possível sem apagar -> lista
    (BASE / "selecionadas.txt").write_text("\n".join(n for n in names if not (rev.get(n) or {}).get("descartar")),
                                          encoding="utf-8")
    print(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sheets", "apply"])
    ap.add_argument("--min-conf", type=float, default=0.2)
    ap.add_argument("--max-area", type=float, default=0.35, help="propostas maiores que isso (fração do quadro) são descartadas")
    ap.add_argument("--min-conf-terreno", type=float, default=0.35, help="confiança mínima para buraco/desnível propostos")
    ap.add_argument("--every", type=int, default=2, help="usa 1 a cada N quadros dos vídeos de treino")
    ap.add_argument("--test-videos", nargs="*", default=["20261119_124457", "20261119_125352", "20261119_124907"])
    args = ap.parse_args()
    props = json.loads((BASE / "proposals.json").read_text(encoding="utf-8"))
    {"sheets": cmd_sheets, "apply": cmd_apply}[args.cmd](args, props)


if __name__ == "__main__":
    main()
