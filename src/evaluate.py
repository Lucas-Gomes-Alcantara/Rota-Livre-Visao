"""Avalia o modelo final no conjunto de TESTE (imagens nunca usadas no treino nem na seleção do modelo).

Métricas (geral e por classe):
  Precision, Recall, F1 = 2PR/(P+R), mAP@50, mAP@50-95, matriz de confusão e tempo de inferência.
  Obs.: a Ultralytics reporta P e R no limiar de confiança que maximiza o F1 médio das classes.

Saída (com --tag test, padrão):
  results/metrics/eval_test.json        tudo em formato estruturado
  results/metrics/eval_test.md          tabelas prontas para o README/PDF
  results/metrics/confusion_test.csv    matriz de confusão (linhas = previsto, colunas = real)
  results/figures/eval_test_*.png       matriz de confusão, curvas PR/F1 geradas pela Ultralytics

Uso:
  python src/evaluate.py
  python src/evaluate.py --data configs/data_test_br.yaml --tag test_br   # só imagens brasileiras
  python src/evaluate.py --bench-cpu                                      # mede também em CPU
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from common import IMG_EXTS, ROOT, load_classes, load_yaml, save_json


def f1(p, r):
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def latency_benchmark(model, images: list[Path], device: str, imgsz: int, n: int = 50, warmup: int = 5) -> dict:
    """Tempo ponta a ponta por imagem (leitura + pré-processamento + rede + NMS), batch = 1."""
    imgs = images[: n + warmup]
    if len(imgs) <= warmup:
        return {}
    times = []
    for k, p in enumerate(imgs):
        t = time.perf_counter()
        model.predict(str(p), device=device, imgsz=imgsz, verbose=False)
        dt = (time.perf_counter() - t) * 1000
        if k >= warmup:
            times.append(dt)
    times.sort()
    return {
        "dispositivo": device,
        "imagens_medidas": len(times),
        "media_ms": round(statistics.mean(times), 1),
        "mediana_ms": round(statistics.median(times), 1),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1),
        "fps_aprox": round(1000 / statistics.mean(times), 1),
    }


def top_errors(cm: np.ndarray, names: list[str], k: int = 8) -> list[dict]:
    """cm[previsto, real]; o último índice é 'fundo' (nada)."""
    labels = names + ["fundo"]
    nc = len(names)
    errs = []
    for p in range(nc + 1):
        for t in range(nc + 1):
            if p == t or cm[p, t] == 0:
                continue
            if t == nc:
                kind = "falso positivo (detectou onde não havia nada)"
            elif p == nc:
                kind = "falso negativo (não detectou)"
            else:
                kind = "confusão entre classes"
            errs.append({"real": labels[t], "previsto": labels[p], "quantidade": int(cm[p, t]), "tipo": kind})
    return sorted(errs, key=lambda e: -e["quantidade"])[:k]


def to_markdown(res: dict) -> str:
    L = [f"## Avaliação — {res['tag']}", "",
         f"Imagens: {res['imagens']} | Instâncias: {res['instancias']} | Modelo: `{res['modelo']}`", "",
         "| Classe | Precision | Recall | F1 | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|---|"]
    for row in res["por_classe"] + [res["geral"]]:
        L.append(f"| {row['classe']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
                 f"{row['map50']:.3f} | {row['map50_95']:.3f} |")
    s = res["tempo_ultralytics_ms"]
    L += ["", f"Tempo por imagem (Ultralytics, {res['dispositivo_validacao']}): "
              f"pré {s.get('preprocess', 0):.1f} ms + inferência {s.get('inference', 0):.1f} ms + "
              f"pós {s.get('postprocess', 0):.1f} ms"]
    for b in res.get("latencia_ponta_a_ponta", []):
        L.append(f"Latência ponta a ponta ({b['dispositivo']}, batch 1, {b['imagens_medidas']} imagens): "
                 f"média {b['media_ms']} ms, mediana {b['mediana_ms']} ms, p95 {b['p95_ms']} ms (~{b['fps_aprox']} FPS)")
    L += ["", "### Erros mais frequentes", "", "| Real | Previsto | Qtde | Tipo |", "|---|---|---|---|"]
    for e in res["erros_mais_frequentes"]:
        L.append(f"| {e['real']} | {e['previsto']} | {e['quantidade']} | {e['tipo']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="0 para GPU, cpu para CPU (padrão: automático)")
    ap.add_argument("--bench-n", type=int, default=50)
    ap.add_argument("--bench-cpu", action="store_true", help="mede a latência também em CPU")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    classes = load_classes()
    model = YOLO(args.weights)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=16, device=device,
                  plots=True, project=str(ROOT / "runs/rota_livre"), name=f"eval_{args.tag}", exist_ok=True,
                  verbose=False)

    box = m.box
    idx = list(map(int, box.ap_class_index))
    per_class = []
    for j, ci in enumerate(idx):
        p, r = float(box.p[j]), float(box.r[j])
        per_class.append({"classe": classes[ci], "precision": p, "recall": r, "f1": f1(p, r),
                          "map50": float(box.ap50[j]), "map50_95": float(box.ap[j])})
    missing = [c for i, c in enumerate(classes) if i not in idx]
    geral = {"classe": "**Geral (média)**", "precision": float(box.mp), "recall": float(box.mr),
             "f1": f1(float(box.mp), float(box.mr)), "map50": float(box.map50), "map50_95": float(box.map)}

    # Ultralytics: matrix[previsto, real], último índice = fundo; calculada com conf=0.25 e IoU=0.45.
    try:
        cm = np.array(m.confusion_matrix.matrix)
    except AttributeError as e:
        raise SystemExit("Esta versão da ultralytics não expõe confusion_matrix no resultado de val(); "
                         "use a versão registrada em requirements.txt") from e
    header = classes + ["fundo"]
    csv_lines = ["previsto\\real," + ",".join(header)]
    for i, name in enumerate(header):
        csv_lines.append(name + "," + ",".join(str(int(v)) for v in cm[i]))
    (ROOT / "results/metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / f"results/metrics/confusion_{args.tag}.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    save_dir = Path(getattr(m, "save_dir", ROOT / "runs/rota_livre" / f"eval_{args.tag}"))
    for png in save_dir.glob("*.png"):
        shutil.copy2(png, ROOT / "results/figures" / f"eval_{args.tag}_{png.name}")

    # imagens do split avaliado, para medir latência
    data_cfg = load_yaml(args.data)
    split_ref = Path(data_cfg["path"]) / data_cfg[args.split]
    if split_ref.suffix == ".txt":
        images = [Path(l) for l in split_ref.read_text(encoding="utf-8").split() if l]
    else:
        images = sorted(p for p in split_ref.iterdir() if p.suffix.lower() in IMG_EXTS)
    benches = [latency_benchmark(model, images, device, args.imgsz, args.bench_n)]
    if args.bench_cpu and device != "cpu":
        benches.append(latency_benchmark(model, images, "cpu", args.imgsz, min(args.bench_n, 20)))
    benches = [b for b in benches if b]
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        for b in benches:
            if b["dispositivo"] != "cpu":
                b["dispositivo"] = f"GPU {gpu}"

    res = {
        "tag": args.tag, "modelo": Path(args.weights).name, "dados": args.data, "split": args.split,
        "imagens": len(images), "instancias": int(cm[:, :-1].sum()),
        "por_classe": per_class, "classes_sem_instancias_no_split": missing, "geral": geral,
        "tempo_ultralytics_ms": {k: float(v) for k, v in m.speed.items()},
        "dispositivo_validacao": "GPU" if device != "cpu" else "CPU",
        "latencia_ponta_a_ponta": benches,
        "matriz_confusao": {"rotulos": header, "linhas_previsto_colunas_real": cm.astype(int).tolist()},
        "erros_mais_frequentes": top_errors(cm, classes),
    }
    save_json(res, ROOT / f"results/metrics/eval_{args.tag}.json")
    md = to_markdown(res)
    (ROOT / f"results/metrics/eval_{args.tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
