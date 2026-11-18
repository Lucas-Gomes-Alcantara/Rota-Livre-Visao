"""Treina o detector Rota Livre (Ultralytics YOLO26) a partir dos pesos pré-treinados no COCO.

Registra, para o relatório:
  results/metrics/train_env.json          versões (ultralytics, torch, CUDA), GPU, tempo total
  results/metrics/train_config_used.json  todos os hiperparâmetros efetivamente usados
  results/metrics/train_history.csv       cópia do results.csv da Ultralytics (perdas e métricas por época)
  models/best.pt                          modelo final = melhor época na VALIDAÇÃO (fitness da Ultralytics)

Uso:
  python src/train.py                         # usa configs/train.yaml
  python src/train.py --epochs 50 --batch 8   # sobrescreve qualquer chave
  python src/train.py --resume                # retoma o último treino interrompido (Colab caiu)
"""
from __future__ import annotations

import argparse
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, load_yaml, save_json


def env_info() -> dict:
    import torch
    import ultralytics

    info = {
        "data_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_disponivel": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memoria_gb=round(p.total_memory / 1e9, 1), cuda=torch.version.cuda)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/train.yaml"))
    ap.add_argument("--resume", action="store_true")
    args, extra = ap.parse_known_args()

    cfg = load_yaml(args.config)
    # sobrescritas do tipo --epochs 50 --batch 8
    for k, v in zip(extra[0::2], extra[1::2]):
        key = k.lstrip("-").replace("-", "_")
        cfg[key] = load_yaml_scalar(v)

    from ultralytics import YOLO

    cfg["data"] = str((ROOT / cfg["data"]).resolve())
    cfg["project"] = str((ROOT / cfg["project"]).resolve())
    if not Path(cfg["data"]).exists():
        raise SystemExit(f"{cfg['data']} não existe. Rode src/split.py antes.")

    info = env_info()
    print("Ambiente:", info)
    run_dir = Path(cfg["project"]) / cfg["name"]

    t0 = time.time()
    if args.resume:
        last = run_dir / "weights/last.pt"
        if not last.exists():
            raise SystemExit(f"Nada para retomar em {last}")
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        model_name = cfg.pop("model")
        model = YOLO(model_name)
        results = model.train(**cfg, exist_ok=False)
        cfg["model"] = model_name
    elapsed = time.time() - t0

    # A Ultralytics pode ter incrementado o nome (yolo26s_v12...) — usa o diretório real.
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights/best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models/best.pt")
    if (run_dir / "results.csv").exists():
        shutil.copy2(run_dir / "results.csv", ROOT / "results/metrics/train_history.csv")
    for fig in ("results.png", "labels.jpg", "BoxPR_curve.png", "BoxF1_curve.png"):
        if (run_dir / fig).exists():
            shutil.copy2(run_dir / fig, ROOT / "results/figures" / f"train_{fig}")

    info.update(
        tempo_treino_segundos=round(elapsed, 1),
        tempo_treino_legivel=f"{int(elapsed // 3600)}h{int(elapsed % 3600 // 60):02d}min",
        diretorio_execucao=str(run_dir),
        epocas_executadas=int(model.trainer.epoch) + 1,
        melhor_fitness_validacao=float(getattr(model.trainer, "best_fitness", 0) or 0),
        retomado=args.resume,
    )
    save_json(info, ROOT / "results/metrics/train_env.json")
    used = load_yaml(run_dir / "args.yaml") if (run_dir / "args.yaml").exists() else cfg
    save_json(used, ROOT / "results/metrics/train_config_used.json")
    print(f"\nTreino concluído em {info['tempo_treino_legivel']}. Modelo final: models/best.pt")


def load_yaml_scalar(text: str):
    import yaml
    return yaml.safe_load(text)


if __name__ == "__main__":
    main()
