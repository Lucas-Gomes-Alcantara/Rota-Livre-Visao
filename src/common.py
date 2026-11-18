"""Funções compartilhadas pelos scripts do Rota Livre."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow renomeia arquivos como "foto_jpg.rf.3f2a...e9.jpg"; o trecho antes de "_jpg.rf." é o nome original.
_RF_RE = re.compile(r"^(?P<stem>.+?)_(?:jpe?g|png|bmp|webp)\.rf\.[0-9a-fA-F]+$", re.IGNORECASE)


def _load_dotenv(path=ROOT / ".env"):
    """Carrega chaves do arquivo .env local (se existir) sem sobrescrever variáveis já definidas."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_classes(path: str | Path = ROOT / "configs/classes.yaml") -> list[str]:
    cfg = load_yaml(path)
    names = cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def roboflow_original_stem(stem: str) -> str:
    m = _RF_RE.match(stem)
    return m.group("stem") if m else stem


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Lê um .txt YOLO. Linhas de polígono (segmentação) são convertidas para caixa envolvente."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            boxes.append((cls, *vals))
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # polígono -> bbox
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        else:
            raise ValueError(f"Linha YOLO inválida em {path}: {line!r}")
    return boxes


def write_yolo_labels(path: Path, boxes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
