"""Rota Livre — assistente de alerta de barreiras na calçada (vídeo em primeira pessoa ou câmera ao vivo).

Para cada quadro:
 1. o detector treinado (models/best.pt) encontra buracos, desníveis e obstáculos;
 2. só conta como PERIGO o que está no CORREDOR DE CAMINHADA (trapézio à frente do usuário);
 3. a distância é estimada pela altura da base da caixa no quadro (mais baixo = mais perto);
 4. um filtro temporal exige a detecção em k de N quadros seguidos (evita alerta piscando);
 5. o alerta é mostrado na tela e falado (voz pt-BR do Windows) ou, sem voz, sinalizado com bipes.

Saídas (modo vídeo):
  results/demo/<nome>_alerta.mp4     vídeo anotado com áudio dos alertas
  results/demo/<nome>_eventos.csv    linha do tempo dos alertas (tempo, classe, distância, confiança)

Uso:
  python src/alerta.py --source "Videos RUA/20261119_125352.mp4"
  python src/alerta.py --source 0 --live          # webcam/câmera ao vivo, com voz
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
import wave
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common import ROOT

NOMES = {"buraco": "BURACO", "desnivel": "DESNÍVEL", "obstaculo": "OBSTÁCULO"}
FALA = {"buraco": "buraco", "desnivel": "desnível", "obstaculo": "obstáculo"}
CORES = {"buraco": (38, 38, 220), "desnivel": (0, 140, 234), "obstaculo": (235, 99, 37)}  # BGR
# Corredor de caminhada em coordenadas normalizadas (x, y), câmera no peito/mão apontando à frente.
BANNER = 130  # altura da faixa de status (px, no quadro de saída)
DIST_RANK = {"à frente": 0, "perto": 1, "muito perto": 2}
CORREDOR = np.array([[0.12, 1.0], [0.88, 1.0], [0.68, 0.40], [0.32, 0.40]], dtype=np.float32)


def distancia(y_base: float) -> str:
    return "muito perto" if y_base > 0.85 else "perto" if y_base > 0.65 else "à frente"


def no_corredor(xyxy_n, mask_small) -> float:
    """Fração da caixa (metade inferior, onde o objeto toca o chão) dentro do corredor."""
    h, w = mask_small.shape
    x1, y1, x2, y2 = xyxy_n
    y1 = y1 + (y2 - y1) * 0.5
    a, b = int(x1 * w), int(min(x2 * w, w - 1)) + 1
    c, d = int(y1 * h), int(min(y2 * h, h - 1)) + 1
    roi = mask_small[c:d, a:b]
    return float(roi.mean()) if roi.size else 0.0


class Suavizador:
    def __init__(self, n=6, k=3):
        self.n, self.k = n, k
        self.hist = {c: deque(maxlen=n) for c in NOMES}

    def update(self, presentes: set[str]) -> set[str]:
        ativos = set()
        for c, h in self.hist.items():
            h.append(c in presentes)
            if sum(h) >= self.k:
                ativos.add(c)
        return ativos


def render_texts(img, items):
    """Desenha vários textos com acentos num único passe PIL. items: (texto, (x, y), tamanho, cor, fundo|None)."""
    if not items:
        return
    from PIL import Image, ImageDraw
    from draw import get_font
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, org, size, color, bg in items:
        font = get_font(int(size))
        tb = d.textbbox(org, text, font=font)
        if bg is not None:
            d.rectangle([tb[0] - 8, tb[1] - 6, tb[2] + 8, tb[3] + 6], fill=tuple(bg[::-1]))
        d.text(org, text, font=font, fill=tuple(color[::-1]))
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def desenhar(frame, dets, ativos_info, rodape=None):
    H, W = frame.shape[:2]
    texts = []
    over = frame.copy()
    poly = (CORREDOR * [W, H]).astype(np.int32)
    cv2.fillPoly(over, [poly], (80, 200, 80))
    frame[:] = cv2.addWeighted(over, 0.18, frame, 0.82, 0)
    cv2.polylines(frame, [poly], True, (80, 200, 80), 2)
    for d in dets:
        x1, y1, x2, y2 = (np.array(d["xyxy"]) * [W, H, W, H]).astype(int)
        cor = CORES[d["cls"]] if d["perigo"] else (150, 150, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 6 if d["perigo"] else 2)
        ly = y1 - 40 if y1 - 40 > BANNER + 6 else max(y1, BANNER) + 10   # nunca sobre o banner
        texts.append((f"{NOMES[d['cls']].title()} {d['conf']:.2f}", (x1 + 6, ly), 26, (255, 255, 255), cor))
    if ativos_info:
        top = max(ativos_info, key=lambda a: a["y_base"])
        cv2.rectangle(frame, (0, 0), (W, BANNER), CORES[top["cls"]], -1)
        texts.append((f"ATENÇÃO: {NOMES[top['cls']]}", (24, 18), int(W / 15), (255, 255, 255), None))
        texts.append((top["dist"].upper(), (24, 18 + int(W / 13)), int(W / 18), (255, 255, 255), None))
        if len(ativos_info) > 1:
            outros = ", ".join(NOMES[a["cls"]].lower() for a in ativos_info if a is not top)
            texts.append((f"também: {outros}", (24, BANNER + 14), 28, (255, 255, 255), (40, 40, 40)))
    else:
        cv2.rectangle(frame, (0, 0), (W, BANNER), (60, 140, 60), -1)
        texts.append(("Caminho livre", (24, 30), int(W / 15), (255, 255, 255), None))
    if rodape:
        texts.append((rodape, (W - 290, H - 64), 30, (255, 255, 255), (30, 30, 30)))
    render_texts(frame, texts)


# ----------------------------------------------------------------------------- áudio
def gerar_falas(dest: Path) -> dict | None:
    """Gera um WAV por frase com a voz do Windows (SAPI via pyttsx3). Retorna {frase: caminho} ou None."""
    try:
        import pyttsx3
        eng = pyttsx3.init()
        for v in eng.getProperty("voices"):
            if "pt" in (v.id + v.name).lower() or "brazil" in v.name.lower() or "maria" in v.name.lower():
                eng.setProperty("voice", v.id)
                break
        eng.setProperty("rate", 185)
        dest.mkdir(parents=True, exist_ok=True)
        out = {}
        for c in NOMES:
            for dist in ("muito perto", "perto", "à frente"):
                frase = f"Atenção, {FALA[c]} {dist}"
                p = dest / f"{c}_{dist.replace(' ', '_').replace('à', 'a')}.wav"
                eng.save_to_file(frase, str(p))
                out[(c, dist)] = p
        eng.runAndWait()
        return out if all(p.exists() and p.stat().st_size > 1000 for p in out.values()) else None
    except Exception as e:
        print("voz indisponível, usando bipes:", e)
        return None


def ler_wav(p: Path, sr: int) -> np.ndarray:
    with wave.open(str(p)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(1).astype(np.int16)
        src_sr = w.getframerate()
    if src_sr != sr:
        idx = np.linspace(0, len(data) - 1, int(len(data) * sr / src_sr)).astype(int)
        data = data[idx]
    return data.astype(np.float32) / 32768


def bipe(cls: str, sr: int) -> np.ndarray:
    f = {"buraco": 880, "desnivel": 660, "obstaculo": 440}[cls]
    t = np.arange(int(0.15 * sr)) / sr
    b = 0.5 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))
    return np.concatenate([b, np.zeros(int(0.08 * sr)), b])


def montar_audio(eventos, dur_s, falas, sr=22050) -> np.ndarray:
    track = np.zeros(int((dur_s + 3) * sr), dtype=np.float32)
    livre_ate = 0.0
    for ev in eventos:
        if ev["t"] < livre_ate:  # não sobrepor falas
            continue
        clip = ler_wav(falas[(ev["cls"], ev["dist"])], sr) if falas else bipe(ev["cls"], sr)
        i = int(ev["t"] * sr)
        track[i:i + len(clip)] += clip[: len(track) - i]
        livre_ate = ev["t"] + len(clip) / sr + 0.2
    return np.clip(track, -1, 1)


def ffmpeg_bin() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ----------------------------------------------------------------------------- principal
def processar(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    live = args.live
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"não consegui ler {args.source}")
    H, W = frame.shape[:2]
    scale = args.out_width / W if args.out_width and W > args.out_width else 1.0
    OW, OH = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    mask_small = np.zeros((90, 50), np.uint8)
    cv2.fillPoly(mask_small, [(CORREDOR * [50, 90]).astype(np.int32)], 1)

    out_dir = ROOT / "results/demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    nome = "ao_vivo" if live else Path(str(args.source)).stem
    tmp_video = out_dir / f"{nome}_tmp.mp4"
    writer = None if live else cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))

    suav = Suavizador(args.n, args.k)
    ativos_antes: set = set()
    ultimo_aviso, ultima_det, ativo_ate, dist_falada = {}, {}, {}, {}
    eventos, tempos = [], []
    voz = None
    if live:
        try:
            import pyttsx3
            voz = pyttsx3.init()
        except Exception:
            voz = None
    idx = 0
    while ok:
        t = idx / fps
        t0 = time.perf_counter()
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device)[0]
        tempos.append((time.perf_counter() - t0) * 1000)
        dets = []
        for xyxyn, c, cf in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            cls = names[int(c)]
            frac = no_corredor(xyxyn, mask_small)
            dets.append({"cls": cls, "conf": cf, "xyxy": xyxyn, "y_base": xyxyn[3],
                         "perigo": frac >= args.min_overlap})
        presentes = {d["cls"] for d in dets if d["perigo"]}
        confirmados = suav.update(presentes)
        # memória da última detecção de perigo de cada classe (posição/distância)
        for c in presentes:
            d = max((d for d in dets if d["cls"] == c and d["perigo"]), key=lambda d: d["y_base"])
            ultima_det[c] = {**d, "dist": distancia(d["y_base"])}
        for c in confirmados:
            ativo_ate[c] = t + args.hold_s           # mantém o alerta por hold_s após a última confirmação
        info = [ultima_det[c] for c in NOMES if t <= ativo_ate.get(c, -1) and c in ultima_det]
        for a in info:
            c, rank = a["cls"], DIST_RANK[a["dist"]]
            desde = t - ultimo_aviso.get(c, -99)
            novo = c not in ativos_antes
            mais_perto = rank > dist_falada.get(c, -1) and desde >= 1.0
            if (novo and desde >= args.min_gap_s) or desde >= args.repeat_s or mais_perto:
                eventos.append({"t": round(t, 2), "cls": c, "dist": a["dist"], "conf": round(a["conf"], 3),
                                "tipo": "novo" if novo else ("mais perto" if mais_perto else "repetição")})
                ultimo_aviso[c], dist_falada[c] = t, rank
                if voz is not None:
                    voz.say(f"Atenção, {FALA[c]} {a['dist']}")
                    voz.runAndWait()
        ativos_antes = {a["cls"] for a in info}
        for c in list(dist_falada):
            if c not in ativos_antes:
                dist_falada.pop(c)

        vis = cv2.resize(frame, (OW, OH)) if scale != 1 else frame
        desenhar(vis, dets, info, rodape=f"{tempos[-1]:.0f} ms/quadro")
        if live:
            cv2.imshow("Rota Livre", cv2.resize(vis, (OW // 2, OH // 2)))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        else:
            writer.write(vis)
        ok, frame = cap.read()
        idx += 1
        if idx % 60 == 0:
            print(f"  {idx} quadros, {np.mean(tempos):.1f} ms/quadro", flush=True)
    cap.release()
    if live:
        cv2.destroyAllWindows()
        return
    writer.release()
    dur = idx / fps

    with open(out_dir / f"{nome}_eventos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "cls", "dist", "conf", "tipo"])
        w.writeheader()
        w.writerows(eventos)

    falas = None if args.beep else gerar_falas(ROOT / "work/falas")
    audio = montar_audio(eventos, dur, falas)
    wav = out_dir / f"{nome}_alertas.wav"
    with wave.open(str(wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    final = out_dir / f"{nome}_alerta.mp4"
    ff = ffmpeg_bin()
    if ff:
        subprocess.run([ff, "-y", "-v", "error", "-i", str(tmp_video), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(final)], check=True)
        tmp_video.unlink(missing_ok=True)
    else:
        tmp_video.replace(final)
        print("ffmpeg não encontrado: vídeo salvo sem áudio; áudio em", wav)
    print(f"{nome}: {idx} quadros, {len(eventos)} alertas, média {np.mean(tempos):.1f} ms/quadro "
          f"({1000 / np.mean(tempos):.0f} FPS) | voz: {'sim' if falas else 'bipes'} -> {final}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="arquivo de vídeo, ou índice da câmera (0)")
    ap.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-overlap", type=float, default=0.25, help="fração mínima da base da caixa no corredor")
    ap.add_argument("--n", type=int, default=6, help="janela do filtro temporal (quadros)")
    ap.add_argument("--k", type=int, default=3, help="detecções mínimas na janela para alertar")
    ap.add_argument("--repeat-s", type=float, default=5.0, help="repetir alerta ativo da mesma classe a cada X s")
    ap.add_argument("--min-gap-s", type=float, default=2.5, help="intervalo mínimo entre avisos 'novos' da mesma classe")
    ap.add_argument("--hold-s", type=float, default=0.8, help="mantém o alerta na tela X s após a última confirmação")
    ap.add_argument("--out-width", type=int, default=720)
    ap.add_argument("--beep", action="store_true", help="usar bipes em vez de voz")
    ap.add_argument("--live", action="store_true")
    processar(ap.parse_args())


if __name__ == "__main__":
    main()
