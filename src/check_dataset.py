"""Verifica a qualidade do dataset dividido (data/processed).

Checa:
  - imagens corrompidas ou ilegíveis e imagens muito pequenas;
  - rótulos inválidos (colunas, classe fora do intervalo, coordenadas fora de [0,1], caixa de tamanho zero);
  - caixas minúsculas (< 4 px em algum lado), que costumam ser erro de anotação;
  - imagens idênticas (MD5) em splits diferentes = vazamento;
  - classes com menos exemplos que o mínimo definido em configs/classes.yaml.

Saída: results/metrics/dataset_check.json e resumo no terminal.
Uso:   python src/check_dataset.py [--strict]
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import IMG_EXTS, ROOT, load_classes, load_yaml, save_json

SPLITS = ("train", "val", "test")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(data_root: Path, classes: list[str], min_inst: dict, min_side: int = 64) -> dict:
    nc = len(classes)
    errors, warnings = [], []
    counts = {s: Counter() for s in SPLITS}
    images_per_split, empty_per_split = Counter(), Counter()
    hashes = defaultdict(set)
    tiny_boxes = 0

    for s in SPLITS:
        img_dir = data_root / s / "images"
        if not img_dir.exists():
            errors.append(f"split ausente: {img_dir}")
            continue
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            images_per_split[s] += 1
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    im.load()
                    W, H = im.size
            except Exception as e:
                errors.append(f"imagem ilegível: {img.name} ({e.__class__.__name__})")
                continue
            if min(W, H) < min_side:
                warnings.append(f"imagem pequena ({W}x{H}): {img.name}")
            hashes[md5(img)].add(s)

            lbl = data_root / s / "labels" / f"{img.stem}.txt"
            lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
            lines = [l for l in lines if l.strip()]
            if not lines:
                empty_per_split[s] += 1
            for ln, line in enumerate(lines, 1):
                parts = line.split()
                where = f"{lbl.name}:{ln}"
                if len(parts) != 5:
                    errors.append(f"{where} deveria ter 5 colunas, tem {len(parts)}")
                    continue
                try:
                    c = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f"{where} valor não numérico")
                    continue
                if not 0 <= c < nc:
                    errors.append(f"{where} classe {c} fora de 0..{nc - 1}")
                    continue
                if not all(0 <= v <= 1 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
                    errors.append(f"{where} coordenadas fora de [0,1] ou caixa vazia")
                    continue
                if w * W < 4 or h * H < 4:
                    tiny_boxes += 1
                counts[s][classes[c]] += 1

    leaked = [h for h, ss in hashes.items() if len(ss) > 1]
    if leaked:
        errors.append(f"{len(leaked)} imagens idênticas aparecem em mais de um split (vazamento)")
    if tiny_boxes:
        warnings.append(f"{tiny_boxes} caixas com menos de 4 px de lado (revise a anotação)")

    insufficient = []
    for s in SPLITS:
        for c in classes:
            if counts[s][c] < min_inst.get(s, 0):
                insufficient.append({"split": s, "classe": c, "instancias": counts[s][c], "minimo": min_inst[s]})
    for it in insufficient:
        warnings.append(f"classe '{it['classe']}' tem {it['instancias']} instâncias em {it['split']} "
                        f"(mínimo recomendado {it['minimo']})")

    return {
        "imagens_por_split": dict(images_per_split),
        "imagens_sem_caixa_por_split": dict(empty_per_split),
        "instancias_por_split": {s: {c: counts[s][c] for c in classes} for s in SPLITS},
        "classes_insuficientes": insufficient,
        "erros": errors,
        "avisos": warnings,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/processed"))
    ap.add_argument("--strict", action="store_true", help="termina com erro se houver qualquer erro")
    args = ap.parse_args()

    classes = load_classes()
    min_inst = load_yaml(ROOT / "configs/classes.yaml").get("min_instances", {})
    rep = check(Path(args.data), classes, min_inst)
    save_json(rep, ROOT / "results/metrics/dataset_check.json")

    print("Instâncias por split:")
    for s, d in rep["instancias_por_split"].items():
        print(f"  {s:5s} imagens={rep['imagens_por_split'].get(s, 0):5d} "
              f"sem_caixa={rep['imagens_sem_caixa_por_split'].get(s, 0):4d}  {d}")
    print(f"\nErros: {len(rep['erros'])}")
    for e in rep["erros"][:20]:
        print("  ✗", e)
    print(f"Avisos: {len(rep['avisos'])}")
    for w in rep["avisos"][:20]:
        print("  !", w)
    if args.strict and rep["erros"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
