"""Empacota as fotos Mapillary baixadas para subir no Roboflow e anotar."""
import sys, zipfile
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
imgs = sorted((src / "images").glob("*.jpg"))
with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
    for p in imgs:
        z.write(p, p.name)
print(f"{len(imgs)} imagens em {dst}")
