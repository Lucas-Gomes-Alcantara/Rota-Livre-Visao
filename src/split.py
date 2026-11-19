"""Divide o dataset convertido em treino/validação/teste SEM vazamento de dados.

Critérios:
 1. Grupos: imagens do mesmo grupo (mesma foto original com cópias aumentadas do Roboflow, mesma
    sequência Mapillary, mesmo vídeo, quadros consecutivos do RDD) ficam sempre no mesmo split.
 2. Quase-duplicatas: imagens visualmente quase idênticas (dHash com distância de Hamming <= limite),
    mesmo vindas de fontes diferentes, têm seus grupos unidos antes da divisão.
 3. Estratificação: a divisão é feita por fonte e equilibra a quantidade de caixas de cada classe
    entre os splits (alvo padrão 70/15/15).
 4. test_br: subconjunto do teste com imagens brasileiras (tag "br"), usado para medir generalização.

Saída:
  data/processed/{train,val,test}/{images,labels}/
  data/processed/test_br.txt, data/processed/split_manifest.csv
  configs/data.yaml, configs/data_test_br.yaml
  results/metrics/split_report.json

Uso:
  python src/split.py --ratios 0.7 0.15 0.15 --hash-threshold 6
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common import ROOT, load_classes, read_manifest, save_json, write_manifest

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- quase-duplicatas
def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, chunk: int = 512):
    n = len(hashes)
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        dist = _popcount(block[:, None] ^ hashes[None, :])
        ii, jj = np.nonzero(dist <= threshold)
        for i, j in zip(ii + start, jj):
            if i < j:
                yield int(i), int(j)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ----------------------------------------------------------------------------- divisão
def stratified_group_split(groups: dict[str, np.ndarray], ratios, seed: int) -> dict[str, str]:
    """groups: id -> vetor [n_imagens, n_caixas_classe0, ...]. Retorna id -> split."""
    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -groups[k][1:].sum())  # grupos grandes primeiro (estável após shuffle)
    total = sum(groups.values()) if groups else np.zeros(1)
    targets = {s: total * r for s, r in zip(SPLITS, ratios)}
    current = {s: np.zeros_like(total, dtype=float) for s in SPLITS}
    assign = {}
    for k in keys:
        g = groups[k]
        best, best_score = None, -np.inf
        for s in SPLITS:
            if targets[s].sum() == 0:
                continue
            deficit = (targets[s] - current[s]) / np.maximum(targets[s], 1)
            score = float((g * deficit).sum() / max(g.sum(), 1))
            if score > best_score:
                best, best_score = s, score
        assign[k] = best
        current[best] += g
    return assign


def place(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data/interim/manifest.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--hash-threshold", type=int, default=6,
                    help="distância de Hamming máxima (0-64) para considerar quase-duplicata; -1 desliga")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    args = ap.parse_args()
    assert abs(sum(args.ratios) - 1) < 1e-6, "as proporções precisam somar 1"

    classes = load_classes()
    rows = read_manifest(Path(args.manifest))
    if not rows:
        raise SystemExit("Manifesto vazio. Rode src/convert.py antes.")
    print(f"{len(rows)} imagens no manifesto")

    # 1) une grupos com quase-duplicatas
    uf = UnionFind()
    for r in rows:
        uf.find(r["group"])
    merges = 0
    if args.hash_threshold >= 0:
        print("calculando dHash para detectar quase-duplicatas ...")
        hashes = np.array([dhash(ROOT / r["image"]) for r in rows], dtype=np.uint64)
        for i, j in near_duplicate_pairs(hashes, args.hash_threshold):
            if uf.find(rows[i]["group"]) != uf.find(rows[j]["group"]):
                merges += 1
                uf.union(rows[i]["group"], rows[j]["group"])
    print(f"quase-duplicatas que uniram grupos distintos: {merges}")

    # 2) vetores por grupo final, estratificados pela fonte dominante do grupo
    final_group = [uf.find(r["group"]) for r in rows]
    vec = defaultdict(lambda: np.zeros(1 + len(classes)))
    source_of = {}
    for r, g in zip(rows, final_group):
        vec[g] += np.array([1] + [int(r[f"n_{c}"]) for c in classes], dtype=float)
        source_of.setdefault(g, r["source"])
    forced = {}
    for r, g in zip(rows, final_group):
        if r.get("forced_split"):
            forced[g] = r["forced_split"]
    by_source = defaultdict(dict)
    for g, v in vec.items():
        if g not in forced:
            by_source[source_of[g]][g] = v
    assign = dict(forced)
    for src, groups in by_source.items():
        assign.update(stratified_group_split(groups, args.ratios, args.seed))

    # 3) materializa os splits
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    test_br = []
    for r, g in zip(rows, final_group):
        s = assign[g]
        r["split"], r["final_group"] = s, g
        img, lbl = ROOT / r["image"], ROOT / r["label"]
        dst_img = out / s / "images" / img.name
        place(img, dst_img, args.mode)
        place(lbl, out / s / "labels" / lbl.name, args.mode)
        # superamostragem (só no treino): cópias com outro nome, mesmo split -> sem vazamento
        if s == "train":
            for k in range(2, int(r.get("oversample") or 1) + 1):
                place(img, out / s / "images" / f"{img.stem}_rep{k}{img.suffix}", args.mode)
                place(lbl, out / s / "labels" / f"{lbl.stem}_rep{k}.txt", args.mode)
        if s == "test" and "br" in r.get("tags", "").split(","):
            test_br.append(str(dst_img.resolve()))
    write_manifest(out / "split_manifest.csv", rows)
    (out / "test_br.txt").write_text("\n".join(test_br) + ("\n" if test_br else ""), encoding="utf-8")

    data_cfg = {"path": str(out.resolve()), "train": "train/images", "val": "val/images",
                "test": "test/images", "names": {i: c for i, c in enumerate(classes)}}
    with open(ROOT / "configs/data.yaml", "w", encoding="utf-8") as f:
        f.write("# GERADO por src/split.py — não edite à mão\n")
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)
    if test_br:
        with open(ROOT / "configs/data_test_br.yaml", "w", encoding="utf-8") as f:
            f.write("# GERADO por src/split.py — teste apenas com imagens brasileiras\n")
            yaml.safe_dump({**data_cfg, "test": "test_br.txt"}, f, allow_unicode=True, sort_keys=False)

    # 4) relatório
    rep = {"proporcoes_alvo": dict(zip(SPLITS, args.ratios)), "limiar_dhash": args.hash_threshold,
           "uniões_por_quase_duplicata": merges, "imagens_test_br": len(test_br), "splits": {}}
    for s in SPLITS:
        sr = [r for r in rows if r["split"] == s]
        rep["splits"][s] = {
            "imagens": len(sr),
            "grupos": len({r["final_group"] for r in sr}),
            "caixas_por_classe": {c: sum(int(r[f"n_{c}"]) for r in sr) for c in classes},
            "imagens_por_fonte": dict(Counter(r["source"] for r in sr)),
        }
    # verificação de vazamento: nenhum grupo pode aparecer em dois splits
    seen = defaultdict(set)
    for r in rows:
        seen[r["final_group"]].add(r["split"])
    leaks = [g for g, ss in seen.items() if len(ss) > 1]
    rep["grupos_em_mais_de_um_split"] = len(leaks)
    sizes = Counter(final_group)
    biggest, n_big = sizes.most_common(1)[0]
    rep["maior_grupo"] = {"id": biggest, "imagens": n_big}
    if n_big > 0.05 * len(rows):
        print(f"AVISO: o maior grupo tem {n_big} imagens ({n_big / len(rows):.0%} do total). "
              f"Isso costuma ser o dHash unindo imagens parecidas em cadeia; tente --hash-threshold 4.")
    save_json(rep, ROOT / "results/metrics/split_report.json")

    for s in SPLITS:
        info = rep["splits"][s]
        print(f"{s:5s}: {info['imagens']:5d} imagens, {info['grupos']:5d} grupos, caixas {info['caixas_por_classe']}")
    print(f"test_br: {len(test_br)} imagens | vazamentos de grupo: {len(leaks)}")
    if not test_br:
        print("AVISO: nenhuma imagem brasileira no teste (fonte com tag 'br' desligada ou vazia).")


if __name__ == "__main__":
    main()
