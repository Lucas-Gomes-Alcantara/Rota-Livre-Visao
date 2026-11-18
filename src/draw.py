"""Desenho de caixas delimitadoras com classe e confiança (usado na visualização e na demonstração)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DISPLAY = {"buraco": "Buraco", "desnivel": "Desnível", "obstaculo": "Obstáculo"}
COLORS = {"buraco": (220, 38, 38), "desnivel": (234, 140, 0), "obstaculo": (37, 99, 235)}
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def get_font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def draw_boxes(img: Image.Image, dets: list[dict]) -> Image.Image:
    """dets: [{"cls": "buraco", "conf": 0.87 | None, "xyxy": (x1, y1, x2, y2) em pixels}]"""
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, round(min(W, H) / 250))
    font = get_font(max(12, round(min(W, H) / 35)))
    for det in dets:
        x1, y1, x2, y2 = det["xyxy"]
        color = COLORS.get(det["cls"], (16, 185, 129))
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        text = DISPLAY.get(det["cls"], det["cls"])
        if det.get("conf") is not None:
            text += f" {det['conf']:.2f}"
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 2 * lw if y1 - th - 2 * lw > 0 else y1
        d.rectangle([x1, ty, x1 + tw + 2 * lw, ty + th + 2 * lw], fill=color)
        d.text((x1 + lw, ty + lw - tb[1]), text, fill=(255, 255, 255), font=font)
    return img


def yolo_to_xyxy(xc, yc, w, h, W, H):
    return ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
