"""Baixa fotos de rua de cidades brasileiras do Mapillary (licença CC BY-SA 4.0) para anotar.

Substitui as fotos próprias da equipe: as imagens servem para (a) reforçar a classe "desnivel" e
(b) montar o conjunto de teste brasileiro (test_br) que demonstra generalização.

- Token gratuito: https://www.mapillary.com/dashboard/developers  (crie um app e copie o "Client Token")
- A API exige bbox pequeno (< 0,01 grau²); o script percorre células de 0,005° × 0,005°.
- Mantém no máximo --per-sequence imagens por sequência (evita quadros quase iguais).
- --require-sidewalk usa as detecções automáticas do Mapillary para manter só imagens com calçada visível.
- metadata.csv guarda autor e link de cada foto: a CC BY-SA exige atribuição.

Uso:
  export MAPILLARY_TOKEN='MLY|...'
  python src/mapillary_fetch.py --city sao_paulo --n 300 --require-sidewalk
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from common import ROOT

API = "https://graph.mapillary.com"
# (oeste, sul, leste, norte) — áreas centrais com boa cobertura de calçadas
CITIES = {
    "sao_paulo": (-46.700, -23.600, -46.600, -23.520),
    "rio_de_janeiro": (-43.250, -22.990, -43.170, -22.900),
    "belo_horizonte": (-43.980, -19.960, -43.900, -19.900),
    "curitiba": (-49.300, -25.460, -49.240, -25.410),
    "porto_alegre": (-51.240, -30.060, -51.180, -30.010),
    "recife": (-34.920, -8.090, -34.870, -8.040),
}
CELL = 0.005
FIELDS = "id,thumb_2048_url,sequence,creator,captured_at,is_pano,computed_geometry"


def get(url, params, token, retries=3):
    for k in range(retries):
        r = requests.get(url, params={**params, "access_token": token}, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (k + 1))
    r.raise_for_status()


def has_sidewalk(image_id: str, token: str) -> bool:
    data = get(f"{API}/{image_id}/detections", {"fields": "value"}, token).get("data", [])
    return any(d.get("value", "").startswith("construction--flat--sidewalk") for d in data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", default=["sao_paulo"], choices=list(CITIES))
    ap.add_argument("--n", type=int, default=300, help="total de imagens a baixar (somando as cidades)")
    ap.add_argument("--per-sequence", type=int, default=2)
    ap.add_argument("--require-sidewalk", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/raw/mapillary_br"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        sys.exit("Defina MAPILLARY_TOKEN (gratuito em mapillary.com/dashboard/developers).")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    existing = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            existing = {r["id"] for r in csv.DictReader(f)}

    rng = random.Random(args.seed)
    per_city = max(1, args.n // len(args.city))
    rows, per_seq = [], defaultdict(int)
    stats = defaultdict(int)
    for city in args.city:
        w, s, e, n = CITIES[city]
        cells = [(x, y) for x in _frange(w, e) for y in _frange(s, n)]
        rng.shuffle(cells)
        got = 0
        for x, y in cells:
            if got >= per_city:
                break
            bbox = f"{x:.4f},{y:.4f},{x + CELL:.4f},{y + CELL:.4f}"
            data = get(f"{API}/images", {"fields": FIELDS, "bbox": bbox, "limit": 200}, token).get("data", [])
            rng.shuffle(data)
            for im in data:
                stats["candidatas"] += 1
                seq = im.get("sequence", "")
                if im.get("is_pano") or im["id"] in existing or per_seq[seq] >= args.per_sequence:
                    continue
                if not im.get("thumb_2048_url"):
                    continue
                if args.require_sidewalk and not has_sidewalk(im["id"], token):
                    stats["sem_calcada"] += 1
                    continue
                dst = out / "images" / f"{im['id']}.jpg"
                r = requests.get(im["thumb_2048_url"], timeout=60)
                if r.status_code != 200:
                    continue
                dst.write_bytes(r.content)
                lon, lat = (im.get("computed_geometry") or {}).get("coordinates", [None, None])
                rows.append({
                    "id": im["id"], "sequence": seq, "city": city,
                    "creator": (im.get("creator") or {}).get("username", ""),
                    "captured_at": im.get("captured_at", ""), "lon": lon, "lat": lat,
                    "url": f"https://www.mapillary.com/app/?pKey={im['id']}",
                    "license": "CC BY-SA 4.0",
                })
                per_seq[seq] += 1
                existing.add(im["id"])
                got += 1
                print(f"\r{city}: {got}/{per_city}", end="", flush=True)
                if got >= per_city:
                    break
        print()

    write_header = not meta_path.exists()
    with open(meta_path, "a", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                            ["id", "sequence", "city", "creator", "captured_at", "lon", "lat", "url", "license"])
        if write_header:
            wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} imagens novas em {out/'images'} | sequências distintas: {len(per_seq)} | {dict(stats)}")
    print(f"Atribuição (CC BY-SA): {meta_path}")


def _frange(a, b):
    v = a
    while v < b:
        yield round(v, 4)
        v += CELL


if __name__ == "__main__":
    main()
