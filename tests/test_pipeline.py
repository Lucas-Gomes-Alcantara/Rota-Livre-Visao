"""Teste de funcionamento do pipeline de dados com dados SINTÉTICOS (não precisa de GPU nem de internet).

Cria uma cópia temporária do repositório com três fontes falsas:
  - "rf_yolo":  export estilo Roboflow (YOLO) com cópias aumentadas "foto_jpg.rf.<hash>.jpg",
                uma imagem corrompida e classes que devem ser descartadas;
  - "rdd_voc":  PASCAL VOC estilo RDD2022 (quadros "Japan_000123");
  - "dup_br":   fonte com tag "br" contendo uma QUASE-DUPLICATA de uma imagem de rf_yolo.
Depois roda convert -> split -> check_dataset -> visualize_labels e confere:
  mapeamento de classes, descarte de corrompida, grupos inteiros no mesmo split,
  quase-duplicata no mesmo split da original, test_br gerado, zero erros na verificação.

Uso: python tests/test_pipeline.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

REPO = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def rand_img(path: Path, size=(320, 240)):
    path.parent.mkdir(parents=True, exist_ok=True)
    # padrão com estrutura (gradientes + blocos) para o dHash ser estável
    w, h = size
    base = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.Resampling.BILINEAR)
    img.save(path, quality=95)
    return img


def make_sources(root: Path):
    raw = root / "data/raw"
    # ---- fonte YOLO estilo Roboflow
    y = raw / "rf_yolo"
    (y / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (y / "data.yaml").write_text("names: ['pothole', 'person', 'broken']\nnc: 3\n")
    for k in range(40):
        stem = f"foto{k:03d}"
        for aug in range(2):  # 2 cópias "aumentadas" da mesma foto -> mesmo grupo
            split = "train" if k < 30 else "valid"
            name = f"{stem}_jpg.rf.{(k * 7 + aug):032x}"
            im = rand_img(y / split / "images" / f"{name}.jpg")
            if aug == 1:
                im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(y / split / "images" / f"{name}.jpg")
            lbl = y / split / "labels" / f"{name}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lines = ["0 0.5 0.5 0.2 0.2"]                 # pothole -> buraco
            if k % 3 == 0:
                lines.append("1 0.3 0.3 0.1 0.3")          # person -> descartada
            if k % 2 == 0:
                lines.append("2 0.7 0.7 0.2 0.1")          # broken -> desnivel
            if k == 5:
                lines.append("0 0.99 0.99 0.2 0.2")        # sai da imagem -> recortada
            lbl.write_text("\n".join(lines) + "\n")
    bad = y / "train/images/corrompida_jpg.rf.ffff.jpg"
    bad.write_bytes(b"isto nao e um jpeg")
    (y / "train/labels/corrompida_jpg.rf.ffff.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    # ---- fonte VOC estilo RDD2022
    v = raw / "rdd_voc"
    for k in range(60):
        stem = f"Japan_{k:06d}"
        rand_img(v / "train/images" / f"{stem}.jpg", (600, 600))
        objs = "".join(
            f"<object><name>{n}</name><bndbox><xmin>{x}</xmin><ymin>{x}</ymin><xmax>{x + 80}</xmax>"
            f"<ymax>{x + 60}</ymax></bndbox></object>"
            for n, x in (("D40", 100), ("D00", 300)))
        xml = (f"<annotation><filename>{stem}.jpg</filename><size><width>600</width><height>600</height>"
               f"</size>{objs}</annotation>")
        (v / "train/annotations/xmls").mkdir(parents=True, exist_ok=True)
        (v / "train/annotations/xmls" / f"{stem}.xml").write_text(xml)

    # ---- fonte "br" com quase-duplicata de rf_yolo/foto000 + imagens próprias
    d = raw / "dup_br"
    (d / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("names: ['buraco', 'desnivel', 'obstaculo']\n")
    src = next((y / "train/images").glob("foto000_jpg.rf.*0.jpg"))
    near = ImageEnhance.Brightness(Image.open(src)).enhance(1.05)
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    near.save(d / "images/quase_dup.jpg", quality=90)
    (d / "labels/quase_dup.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    for k in range(30):
        rand_img(d / "images" / f"br{k:03d}.jpg")
        (d / "labels" / f"br{k:03d}.txt").write_text(f"{k % 3} 0.5 0.5 0.3 0.3\n")

    (root / "configs/sources_test.yaml").write_text("""
sources:
  rf_yolo:
    path: data/raw/rf_yolo
    format: yolo
    license: teste
    class_map: {pothole: buraco, broken: desnivel, person: null}
    group: roboflow_stem
  rdd_voc:
    path: data/raw/rdd_voc
    format: voc
    license: teste
    class_map: {D40: buraco}
    default_map: null
    group: regex
    group_regex: '^([A-Za-z_]+?)_(\\d+)$'
    group_bucket: 10
    max_images: 50
  dup_br:
    path: data/raw/dup_br
    format: yolo
    license: teste
    class_map: {buraco: buraco, desnivel: desnivel, obstaculo: obstaculo}
    group: roboflow_stem
    tags: [br]
""")
    # mínimos baixos para o teste
    (root / "configs/classes.yaml").write_text(
        "names: {0: buraco, 1: desnivel, 2: obstaculo}\nmin_instances: {train: 5, val: 2, test: 2}\n")


def run(root: Path, *args):
    print("$", " ".join(args))
    r = subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"falhou: {args}")
    return r.stdout


def main():
    tmp = Path(tempfile.mkdtemp(prefix="rotalivre_test_"))
    root = tmp / "repo"
    shutil.copytree(REPO, root, ignore=shutil.ignore_patterns("data", "results", "runs", "models", "*.zip"))
    for d in ("results/metrics", "results/figures", "data"):
        (root / d).mkdir(parents=True, exist_ok=True)
    make_sources(root)

    run(root, "src/convert.py", "--sources", "configs/sources_test.yaml")
    rep = json.loads((root / "results/metrics/convert_report.json").read_text())
    assert rep["rf_yolo"]["estatisticas"]["imagens_corrompidas_ignoradas"] == 1, "corrompida não foi ignorada"
    # 80 caixas + 2 da foto005 (caixa que sai da imagem é recortada, não descartada; 2 cópias)
    assert rep["rf_yolo"]["caixas_por_classe_rota_livre"]["buraco"] == 82
    assert rep["rf_yolo"]["caixas_por_classe_rota_livre"]["desnivel"] == 40
    assert rep["rf_yolo"]["grupos"] == 40, "cópias aumentadas deveriam formar 40 grupos"
    assert rep["rdd_voc"]["estatisticas"]["imagens_aproveitadas"] == 50, "max_images não respeitado"
    assert rep["rdd_voc"]["caixas_por_classe_rota_livre"] == {"buraco": 50, "desnivel": 0, "obstaculo": 0}

    run(root, "src/split.py", "--hash-threshold", "6")
    rows = list(csv.DictReader(open(root / "data/processed/split_manifest.csv", encoding="utf-8")))
    by_group = defaultdict(set)
    for r in rows:
        by_group[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in by_group.values()), "grupo original dividido entre splits!"
    split_of = {Path(r["orig_path"]).name: r["split"] for r in rows}
    orig = [n for n in split_of if n.startswith("foto000_jpg.rf.")]
    assert len({split_of[n] for n in orig} | {split_of["quase_dup.jpg"]}) == 1, \
        "quase-duplicata ficou em split diferente da original"
    srep = json.loads((root / "results/metrics/split_report.json").read_text())
    assert srep["grupos_em_mais_de_um_split"] == 0
    assert srep["uniões_por_quase_duplicata"] >= 1
    for s in ("train", "val", "test"):
        assert srep["splits"][s]["imagens"] > 0, f"split {s} vazio"
    assert (root / "configs/data_test_br.yaml").exists() and srep["imagens_test_br"] > 0
    frac = {s: srep["splits"][s]["imagens"] / len(rows) for s in ("train", "val", "test")}
    print("proporções obtidas:", {k: round(v, 2) for k, v in frac.items()})
    assert 0.6 < frac["train"] < 0.8

    run(root, "src/check_dataset.py", "--strict")
    chk = json.loads((root / "results/metrics/dataset_check.json").read_text())
    assert not chk["erros"], chk["erros"]

    run(root, "src/visualize_labels.py", "--per-class", "4")
    assert (root / "results/figures/distribuicao_classes.png").exists()
    assert list((root / "results/figures").glob("anotacoes_train_*.jpg"))

    # rótulo inválido deve ser pego pela verificação
    some = next((root / "data/processed/train/labels").glob("*.txt"))
    some.unlink()
    some.write_text("7 0.5 0.5 1.4 0.2\n")
    r = subprocess.run([sys.executable, "src/check_dataset.py", "--strict"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 1 and "fora de 0..2" in r.stdout, "verificação não pegou rótulo inválido"

    print(f"\nTODOS OS TESTES PASSARAM  (pasta temporária: {tmp})")
    return tmp


if __name__ == "__main__":
    main()
