"""Baixa as bases públicas para data/raw/.

Subcomandos:
  roboflow      bases do Roboflow Universe (precisa de chave gratuita: https://app.roboflow.com -> Settings -> API Keys)
  roboflow-own  projeto anotado pela própria equipe (ex.: fotos Mapillary)
  rdd2022    RDD2022 pelo figshare (opcional; arquivos grandes, liste antes com --list)
  pedestrian PEDESTRIAN pelo Zenodo (opcional; 5,9 GB; sem caixas — só serve se a equipe anotar)

Exemplos:
  export ROBOFLOW_API_KEY=xxxx
  python src/download_data.py roboflow
  python src/download_data.py rdd2022 --list
  python src/download_data.py rdd2022 --pattern Japan
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

from common import ROOT

RAW = ROOT / "data/raw"

# nome local -> (workspace, projeto) no Roboflow Universe
ROBOFLOW_SOURCES = {
    "pothole_chitholian": ("pothole-skc", "pothole-kvso8"),
    "sidewalk_walkability": ("sidewalk-detection", "sidewalk-walkability"),
    "sidewalk_obstacles_blindspot": ("blindspot-fmcu0", "sidewalk-obstacles"),
}
FIGSHARE_RDD2022 = "https://api.figshare.com/v2/articles/21431547"
ZENODO_PEDESTRIAN = "https://zenodo.org/api/records/10907945"


def _stream(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dst.name}: {done / 1e6:,.0f}/{total / 1e6:,.0f} MB", end="", flush=True)
    print()


def _unzip(path: Path, dst: Path):
    print(f"  extraindo {path.name} ...")
    with zipfile.ZipFile(path) as z:
        z.extractall(dst)


def latest_version(project) -> int:
    """Descobre a versão mais recente publicada de um projeto do Roboflow Universe."""
    candidates = []
    try:
        for info in project.get_version_information():
            candidates.append(int(str(info.get("id", "")).split("/")[-1]))
    except Exception as e:
        print(f"  (get_version_information falhou: {e})")
    if not candidates:
        try:
            for v in project.versions():
                candidates.append(int(str(getattr(v, "version", "")).split("/")[-1]))
        except Exception as e:
            print(f"  (versions falhou: {e})")
    if candidates:
        return max(candidates)
    for n in range(30, 0, -1):  # último recurso: testa versões de 30 a 1
        try:
            project.version(n)
            return n
        except Exception:
            continue
    raise RuntimeError("Nenhuma versão publicada encontrada; informe com --version nome=N")


def fork_version(rf, original_slug: str):
    """Acha o fork (slug começa com o slug original) no workspace da chave e gera uma versão se preciso."""
    own = rf.workspace()
    matches = [p for p in own.projects() if p.split("/")[-1].startswith(original_slug)]
    if not matches:
        sys.exit(f"Faça o fork de {original_slug} no Roboflow Universe (botão 'Fork Dataset') e rode de novo.")
    fork_ws, fork_proj = matches[0].split("/")[0], matches[0].split("/")[-1]
    project = own.project(fork_proj)
    try:
        return project, latest_version(project), fork_ws, fork_proj
    except RuntimeError:
        pass
    print(f"  gerando versão de {matches[0]} (sem augmentation, só auto-orient) ...")
    settings = {"preprocessing": {"auto-orient": True}, "augmentation": {}}
    v = project.generate_version(settings)
    return project, int(v), fork_ws, fork_proj


def cmd_roboflow(args):
    key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Defina ROBOFLOW_API_KEY (chave gratuita em app.roboflow.com -> Settings -> API Keys).")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    names = args.only or list(ROBOFLOW_SOURCES)
    for name in names:
        ws, proj = ROBOFLOW_SOURCES[name]
        project = rf.workspace(ws).project(proj)
        version = args.version.get(name) if args.version else None
        if version is None:
            try:
                version = latest_version(project)
            except RuntimeError:
                # Base sem versão publicada: usa o fork na conta da equipe (Universe -> "Fork Dataset")
                print(f"  {ws}/{proj} não tem versão publicada; procurando fork na sua conta ...")
                project, version, ws, proj = fork_version(rf, proj)
        dst = RAW / name
        print(f"- {name}: {ws}/{proj} versão {version} -> {dst}")
        for attempt in range(30):  # versão recém-gerada pode levar alguns minutos para ficar pronta
            try:
                project.version(int(version)).download("yolov8", location=str(dst), overwrite=True)
                break
            except Exception as e:
                print(f"  versão ainda não pronta ({e.__class__.__name__}: {e}); tentando de novo em 20 s ...")
                time.sleep(20)
        else:
            sys.exit(f"Versão {version} de {ws}/{proj} não ficou pronta a tempo; rode de novo mais tarde.")
        (dst / "SOURCE.txt").write_text(
            f"https://universe.roboflow.com/{ws}/{proj}/dataset/{version}\n", encoding="utf-8")


def cmd_roboflow_own(args):
    """Baixa o projeto da PRÓPRIA equipe no Roboflow (ex.: imagens Mapillary anotadas) em formato YOLO."""
    key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Defina ROBOFLOW_API_KEY.")
    from roboflow import Roboflow

    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)
    dst = RAW / args.name
    print(f"- {args.workspace}/{args.project} v{args.version} -> {dst}")
    project.version(args.version).download("yolov8", location=str(dst), overwrite=True)


def cmd_rdd2022(args):
    meta = requests.get(FIGSHARE_RDD2022, timeout=60).json()
    files = meta.get("files", [])
    print(f"Licença informada pelo figshare: {meta.get('license', {}).get('name', 'não informada')}")
    for f in files:
        print(f"  {f['name']:40s} {f['size'] / 1e9:6.2f} GB")
    if args.list:
        return
    chosen = [f for f in files if not args.pattern or args.pattern.lower() in f["name"].lower()]
    if not chosen:
        sys.exit("Nenhum arquivo corresponde a --pattern")
    dst = RAW / "rdd2022"
    for f in chosen:
        z = dst / f["name"]
        if not z.exists():
            _stream(f["download_url"], z)
        if z.suffix == ".zip":
            _unzip(z, dst)
            # zips internos (um por país) também são extraídos
            for inner in list(dst.rglob("*.zip")):
                if inner != z:
                    _unzip(inner, inner.parent)
                    inner.unlink()


def cmd_pedestrian(args):
    meta = requests.get(ZENODO_PEDESTRIAN, timeout=60).json()
    lic = meta.get("metadata", {}).get("license", {})
    print(f"Licença informada pelo Zenodo: {lic.get('id', lic) if lic else 'não informada'}")
    for f in meta.get("files", []):
        print(f"  {f['key']:40s} {f['size'] / 1e9:6.2f} GB")
    if args.list:
        return
    dst = RAW / "pedestrian"
    for f in meta.get("files", []):
        z = dst / f["key"]
        if not z.exists():
            _stream(f["links"]["self"], z)
        if z.suffix == ".zip":
            _unzip(z, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("roboflow")
    p.add_argument("--api-key")
    p.add_argument("--only", nargs="*", choices=list(ROBOFLOW_SOURCES))
    p.add_argument("--version", type=lambda s: dict(kv.split("=") for kv in s.split(",")),
                   help="ex.: pothole_chitholian=1,sidewalk_walkability=2 (padrão: última versão)")
    p.set_defaults(func=cmd_roboflow)
    p = sub.add_parser("roboflow-own", help="projeto anotado pela equipe")
    p.add_argument("--api-key")
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--name", default="mapillary_br_annotated")
    p.set_defaults(func=cmd_roboflow_own)
    for name, fn in (("rdd2022", cmd_rdd2022), ("pedestrian", cmd_pedestrian)):
        p = sub.add_parser(name)
        p.add_argument("--list", action="store_true", help="só lista arquivos, tamanhos e licença")
        if name == "rdd2022":
            p.add_argument("--pattern", help="baixa só arquivos cujo nome contém este texto")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
