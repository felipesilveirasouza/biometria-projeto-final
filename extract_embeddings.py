"""Gera os embeddings (templates) a partir dos recortes de preprocess.py.

Saída: cache/embeddings/<variante>/<base>.npz (chaves = caminho da amostra)
e results/efficiency.json (parâmetros, tamanho, tempo de inferência, hardware).

Uso: python extract_embeddings.py --dataset cfp_fp [--variant aligned] [--timing]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import emb_file, get_device, load_config, path, set_seed, variant_name, write_json
from model import build_model, embed, hardware, model_info
from preprocess import crops_dir


def extract(cfg: dict, dataset: str, variant: str, model=None, device=None) -> Path:
    device = device or get_device(cfg)
    model = model or build_model(cfg, device)
    cdir = crops_dir(cfg, variant, dataset)
    q = pd.read_csv(cdir / "quality.csv")
    if cfg["preprocess"]["fta_policy"] == "reject":
        q = q[~q["fta"].astype(bool)]  # sem template: o par recebe score -1 em common.cosine_scores
    keys = q["amostra"].tolist()
    bs = cfg["model"]["batch_size"]
    embs = []
    for i in tqdm(range(0, len(keys), bs), desc=f"emb {dataset}/{variant}"):
        batch = [Image.open(cdir / (k.rsplit(".", 1)[0] + ".png")) for k in keys[i:i + bs]]
        embs.append(embed(model, batch, device))
    out = emb_file(cfg, variant, dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, keys=np.array(keys), emb=np.concatenate(embs).astype(np.float32))
    return out


def timing(cfg: dict, dataset: str, variant: str, model, device: str) -> dict:
    """Tempo médio de inferência da CNN por amostra (batch = 1) e vazão em lote."""
    cdir = crops_dir(cfg, variant, dataset)
    keys = pd.read_csv(cdir / "quality.csv")["amostra"].tolist()[: cfg["model"]["timing_images"]]
    imgs = [Image.open(cdir / (k.rsplit(".", 1)[0] + ".png")).copy() for k in keys]
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for im in imgs[:10]:  # aquecimento
        embed(model, [im], device)
    sync()
    t0 = time.perf_counter()
    for im in imgs:
        embed(model, [im], device)
    sync()
    single = (time.perf_counter() - t0) / len(imgs)
    bs = cfg["model"]["batch_size"]
    t0 = time.perf_counter()
    for i in range(0, len(imgs), bs):
        embed(model, imgs[i:i + bs], device)
    sync()
    batched = (time.perf_counter() - t0) / len(imgs)
    return {"device": device, "ms_per_image_batch1": round(single * 1e3, 3),
            f"ms_per_image_batch{bs}": round(batched * 1e3, 3), "n_images": len(imgs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfp_fp", "cfp_ff", "lfw"], required=True)
    ap.add_argument("--variant", default=None, help="padrão: aligned (ou bbox se align=false)")
    ap.add_argument("--timing", action="store_true", help="mede o tempo de inferência e salva efficiency.json")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    variant = args.variant or variant_name(cfg)
    device = get_device(cfg)
    model = build_model(cfg, device)
    print("salvo em", extract(cfg, args.dataset, variant, model, device))
    if args.timing:
        info = {**model_info(model), "pretrained": cfg["model"]["pretrained"],
                "timing": timing(cfg, args.dataset, variant, model, device), "hardware": hardware()}
        write_json(info, path(cfg, "results") / "efficiency.json")
        print(info)


if __name__ == "__main__":
    main()
