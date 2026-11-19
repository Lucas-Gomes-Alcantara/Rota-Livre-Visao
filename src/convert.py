"""Converte cada fonte de dados (YOLO, PASCAL VOC ou COCO) para o formato YOLO com as classes do Rota Livre.

Saída:
  data/interim/<fonte>/images/*.jpg
  data/interim/<fonte>/labels/*.txt
  data/interim/manifest.csv              (uma linha por imagem: fonte, grupo, licença, contagem por classe)
  results/metrics/convert_report.json    (o que foi aproveitado/descartado em cada fonte)

Uso:
  python src/convert.py                       # todas as fontes com enabled: true
  python src/convert.py --only pothole_chitholian
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from common import (ROOT, list_images, load_classes, load_yaml, read_yolo_labels,
                    roboflow_original_stem, save_json, write_manifest, write_yolo_labels)

SPLIT_DIR_NAMES = {"images", "labels", "train", "valid", "val", "test"}


# ----------------------------------------------------------------------------- leitura das fontes
def _yolo_class_names(root: Path) -> list[str]:
    for yml in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        try:
            cfg = load_yaml(yml)
        except Exception:
            continue
        if isinstance(cfg, dict) and "names" in cfg:
            names = cfg["names"]
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    classes_txt = next(iter(root.rglob("classes.txt")), None)
    if classes_txt:
        return [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise FileNotFoundError(f"Nenhum data.yaml/classes.txt com 'names' encontrado em {root}")


def _yolo_label_path(img: Path) -> Path:
    if img.parent.name == "images":
        return img.parent.parent / "labels" / f"{img.stem}.txt"
    return img.with_suffix(".txt")


def iter_yolo(root: Path):
    """Gera (imagem, [(nome_classe, xc, yc, w, h)]) para um dataset YOLO (ex.: export do Roboflow)."""
    names = _yolo_class_names(root)
    for img in list_images(root):
        boxes = []
        for cls, xc, yc, w, h in read_yolo_labels(_yolo_label_path(img)):
            boxes.append((names[cls], xc, yc, w, h))
        yield img, boxes


def iter_voc(root: Path):
    """PASCAL VOC (ex.: RDD2022). Associa cada XML à imagem de mesmo nome em qualquer subpasta."""
    by_stem = {p.stem: p for p in list_images(root)}
    for xml_path in sorted(root.rglob("*.xml")):
        tree = ET.parse(xml_path).getroot()
        fname = tree.findtext("filename") or xml_path.stem
        img = by_stem.get(Path(fname).stem) or by_stem.get(xml_path.stem)
        if img is None:
            continue
        W = float(tree.findtext("size/width") or 0)
        H = float(tree.findtext("size/height") or 0)
        if W <= 0 or H <= 0:
            with Image.open(img) as im:
                W, H = im.size
        boxes = []
        for obj in tree.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((name, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        yield img, boxes


def iter_coco(root: Path):
    for jpath in sorted(root.rglob("*.json")):
        data = json.loads(jpath.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "annotations" not in data:
            continue
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        anns: dict[int, list] = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            img = jpath.parent / im["file_name"]
            if not img.exists():
                continue
            W, H = float(im["width"]), float(im["height"])
            boxes = []
            for a in anns.get(im["id"], []):
                x, y, w, h = a["bbox"]
                boxes.append((cats[a["category_id"]], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            yield img, boxes


READERS = {"yolo": iter_yolo, "voc": iter_voc, "coco": iter_coco}


# ----------------------------------------------------------------------------- agrupamento
def load_mapillary_sequences(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {r["id"]: r["sequence"] for r in csv.DictReader(f)}


def group_key(img: Path, root: Path, cfg: dict, seq_lookup: dict[str, str]) -> str:
    strategy = cfg.get("group", "roboflow_stem")
    stem = roboflow_original_stem(img.stem)
    if strategy == "roboflow_stem":
        return stem
    if strategy == "regex":
        m = re.match(cfg["group_regex"], stem)
        if not m:
            return stem
        if cfg.get("group_bucket") and m.lastindex and m.lastindex >= 2:
            return f"{m.group(1)}_{int(m.group(2)) // int(cfg['group_bucket'])}"
        return m.group(1)
    if strategy == "parent_dir":
        p = img.parent
        while p.name.lower() in SPLIT_DIR_NAMES and p != root:
            p = p.parent
        return p.name
    if strategy == "mapillary":
        return "seq_" + seq_lookup.get(stem, stem)
    raise ValueError(f"Estratégia de grupo desconhecida: {strategy}")


# ----------------------------------------------------------------------------- conversão
def clip_box(xc, yc, w, h):
    x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def convert_source(name: str, cfg: dict, classes: list[str], out_root: Path, seed: int) -> tuple[list[dict], dict]:
    src_root = ROOT / cfg["path"]
    if not src_root.exists():
        raise FileNotFoundError(f"[{name}] pasta não encontrada: {src_root} (veja datasets/README.md)")
    class_map: dict = cfg.get("class_map", {})
    has_default = "default_map" in cfg
    default = cfg.get("default_map")
    cls_index = {c: i for i, c in enumerate(classes)}
    seq_lookup = load_mapillary_sequences(ROOT / cfg["metadata_csv"]) if cfg.get("metadata_csv") else {}

    stats = Counter()
    orig_counts, unmapped = Counter(), Counter()
    with_boxes, empty = [], []
    for img, boxes in READERS[cfg["format"]](src_root):
        stats["imagens_lidas"] += 1
        mapped = []
        for orig, xc, yc, w, h in boxes:
            orig_counts[orig] += 1
            if orig in class_map:
                target = class_map[orig]
            elif has_default:
                target = default
            else:
                unmapped[orig] += 1
                target = None
            if target is None:
                stats["caixas_descartadas_por_classe"] += 1
                continue
            if target not in cls_index:
                raise ValueError(f"[{name}] class_map aponta para classe inexistente: {target}")
            clipped = clip_box(xc, yc, w, h)
            if clipped is None:
                stats["caixas_invalidas_descartadas"] += 1
                continue
            mapped.append((cls_index[target], *clipped))
        (with_boxes if mapped else empty).append((img, mapped))

    rng = random.Random(seed)
    rng.shuffle(with_boxes)
    if cfg.get("max_images"):
        with_boxes = with_boxes[: int(cfg["max_images"])]
    n_empty = min(len(empty), round(float(cfg.get("keep_empty", 0)) * len(with_boxes)))
    rng.shuffle(empty)
    selected = with_boxes + empty[:n_empty]

    out_img, out_lbl = out_root / name / "images", out_root / name / "labels"
    if (out_root / name).exists():
        shutil.rmtree(out_root / name)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    rows, used_names = [], set()
    tags = ",".join(cfg.get("tags", []))
    for img, boxes in selected:
        try:
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            stats["imagens_corrompidas_ignoradas"] += 1
            continue
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", img.stem)
        new_stem, k = f"{name}__{base}", 1
        while new_stem in used_names:
            k += 1
            new_stem = f"{name}__{base}_{k}"
        used_names.add(new_stem)
        dst_img = out_img / f"{new_stem}{img.suffix.lower()}"
        shutil.copy2(img, dst_img)
        dst_lbl = out_lbl / f"{new_stem}.txt"
        write_yolo_labels(dst_lbl, boxes)
        per_cls = Counter(b[0] for b in boxes)
        row = {
            "image": dst_img.relative_to(ROOT).as_posix(),
            "label": dst_lbl.relative_to(ROOT).as_posix(),
            "source": name,
            "group": f"{name}::{group_key(img, src_root, cfg, seq_lookup)}",
            "tags": tags,
            "license": cfg.get("license", ""),
            "width": W,
            "height": H,
            "orig_path": str(img),
        }
        gk = group_key(img, src_root, cfg, seq_lookup)
        forced = ""
        for sp, gs in (cfg.get("split_groups") or {}).items():
            if gk in set(gs):
                forced = sp
        if gk in set(cfg.get("test_groups", [])):
            forced = "test"
        row["forced_split"] = forced or cfg.get("default_split", "")
        row["oversample"] = int(cfg.get("oversample", 1))
        for i, c in enumerate(classes):
            row[f"n_{c}"] = per_cls.get(i, 0)
        row["n_total"] = len(boxes)
        rows.append(row)
    stats["imagens_aproveitadas"] = len(rows)
    stats["imagens_vazias_mantidas"] = sum(1 for r in rows if r["n_total"] == 0)
    report = {
        "estatisticas": dict(stats),
        "caixas_por_classe_original": dict(orig_counts),
        "classes_sem_mapeamento": dict(unmapped),
        "caixas_por_classe_rota_livre": {c: sum(r[f"n_{c}"] for r in rows) for c in classes},
        "grupos": len({r["group"] for r in rows}),
    }
    if unmapped:
        print(f"  ATENÇÃO [{name}]: classes sem mapeamento (descartadas): {dict(unmapped)}")
    return rows, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(ROOT / "configs/sources.yaml"))
    ap.add_argument("--classes", default=str(ROOT / "configs/classes.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data/interim"))
    ap.add_argument("--only", nargs="*", help="converter apenas estas fontes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = load_classes(args.classes)
    sources = load_yaml(args.sources)["sources"]
    out_root = Path(args.out)
    manifest_path = out_root / "manifest.csv"
    report_path = ROOT / "results/metrics/convert_report.json"

    # Mantém linhas de fontes já convertidas quando --only é usado.
    rows: list[dict] = []
    report: dict = {}
    if args.only and manifest_path.exists():
        from common import read_manifest
        rows = [r for r in read_manifest(manifest_path) if r["source"] not in args.only]
        if report_path.exists():
            report = {k: v for k, v in json.loads(report_path.read_text()).items() if k not in args.only}

    for name, cfg in sources.items():
        if args.only and name not in args.only:
            continue
        if not cfg.get("enabled", True):
            print(f"- {name}: desligada (enabled: false)")
            continue
        print(f"- convertendo {name} ...")
        src_rows, src_report = convert_source(name, cfg, classes, out_root, args.seed)
        rows += src_rows
        report[name] = src_report
        print(f"  {src_report['estatisticas'].get('imagens_aproveitadas', 0)} imagens, "
              f"caixas: {src_report['caixas_por_classe_rota_livre']}")

    write_manifest(manifest_path, rows)
    save_json(report, report_path)
    print(f"\nManifesto: {manifest_path} ({len(rows)} imagens)\nRelatório: {report_path}")


if __name__ == "__main__":
    main()
