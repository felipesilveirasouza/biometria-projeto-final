"""Figura com exemplos reais das etapas do pipeline (§6), para o artigo e os slides.

Linhas: amostras de exemplo. Colunas: imagem bruta → detecção (bbox + 5
landmarks) → recorte alinhado → recorte só pela bbox (ablação) → entrada
normalizada da CNN.

Uso: python figures.py --dataset cfp_fp [--n 3]
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from common import dataset_root, get_device, load_config, load_pairs, path
from preprocess import FacePreprocessor, align, bbox_crop, plausible_landmarks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfp_fp", "lfw"], required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    c = cfg["preprocess"]
    pairs = load_pairs(cfg, args.dataset)
    g = pairs[pairs.tipo == "genuine"].sample(args.n, random_state=cfg["seed"])
    keys = [k for a, b in zip(g.amostra_1, g.amostra_2) for k in (a, b)][: 2 * args.n]
    prep = FacePreprocessor(cfg, get_device(cfg))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    cols = ["Bruta", "Detecção + landmarks", "Alinhada (landmarks)", "Só bbox (ablação)", "Normalizada"]
    fig, axes = plt.subplots(len(keys), 5, figsize=(11, 2.3 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        img = Image.open(dataset_root(cfg, args.dataset) / key).convert("RGB")
        det = prep.detect(img)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(img)
        if det:
            box, prob, pts = det
            axes[i, 1].add_patch(patches.Rectangle(box[:2], box[2] - box[0], box[3] - box[1],
                                                   fill=False, color="lime", lw=1.5))
            axes[i, 1].scatter(pts[:, 0], pts[:, 1], s=8, c="red")
            axes[i, 1].set_xlim(0, img.size[0])
            axes[i, 1].set_ylim(img.size[1], 0)
            bb = bbox_crop(img, box, c["image_size"], c["bbox_margin"])
            ok, why = plausible_landmarks(img, box, pts, c["image_size"], c)
            al = align(img, pts, c["image_size"], c["align_scale"]) if ok else bb
            axes[i, 1].set_xlabel(f"p = {prob:.3f}" + ("" if ok else f"\n{why} → usa bbox"), fontsize=7)
            axes[i, 2].imshow(al)
            axes[i, 3].imshow(bb)
            norm = (np.asarray(al, dtype=np.float32) - 127.5) / 128.0
            axes[i, 4].imshow(np.clip((norm + 1) / 2, 0, 1))
        else:
            axes[i, 1].set_xlabel("sem detecção (FTA)", fontsize=7)
        axes[i, 0].set_ylabel(key.split("/")[-3] if args.dataset == "cfp_fp" else key.split("/")[0], fontsize=7)
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=8)
    fig.tight_layout()
    out = path(cfg, "results") / f"pipeline_{args.dataset}.png"
    fig.savefig(out, dpi=150)
    print("salvo em", out)


if __name__ == "__main__":
    main()
