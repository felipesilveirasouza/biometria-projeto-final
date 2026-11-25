"""Figura com exemplos reais das etapas do pipeline (§6), para o artigo e os slides.

Linhas: amostras de exemplo. Colunas: imagem bruta → detecção (bbox + 5
landmarks) → recorte pela bbox (principal) → recorte alinhado por landmarks
(ablação) → entrada normalizada da CNN (da variante principal).

Uso: python figures.py --dataset cfp_fp [--n 3] [--out nome.png]
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from common import dataset_root, get_device, load_config, load_pairs, path
from preprocess import FacePreprocessor, align, bbox_crop, plausible_landmarks


def dataset_examples(cfg: dict, out_name: str = "exemplos_bases.png") -> None:
    """Exemplos das duas bases (§21, seção 3): no CFP-FP, 3 identidades com 2 frontais
    e 2 perfis; no LFW, 3 pares genuínos. Imagens brutas, sem pré-processamento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(cfg["seed"])
    cfp_root, lfw_root = dataset_root(cfg, "cfp_fp"), dataset_root(cfg, "lfw")
    ids = sorted(p.name for p in (cfp_root / "Data" / "Images").iterdir())
    rows = []
    for i in rng.choice(len(ids), 3, replace=False):
        d = cfp_root / "Data" / "Images" / ids[i]
        rows.append([d / "frontal" / "01.jpg", d / "frontal" / "02.jpg", d / "profile" / "01.jpg", d / "profile" / "02.jpg"])
    g = load_pairs(cfg, "lfw")
    g = g[g.tipo == "genuine"].sample(6, random_state=cfg["seed"])
    lfw = [lfw_root / k for a, b in zip(g.amostra_1, g.amostra_2) for k in (a, b)]
    fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
    for r in range(3):
        for c in range(4):
            axes[r, c].imshow(Image.open(rows[r][c]).convert("RGB"))
            axes[r, 4 + c].imshow(Image.open(lfw[4 * r + c]).convert("RGB"))
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    for c, t in enumerate(["frontal", "frontal", "perfil", "perfil"]):
        axes[0, c].set_title(f"CFP-FP: {t}", fontsize=9)
    for c in range(4):
        axes[0, 4 + c].set_title(f"LFW: par {c // 2 + 1}, img {c % 2 + 1}", fontsize=9)
    fig.tight_layout()
    out = path(cfg, "results") / out_name
    fig.savefig(out, dpi=130)
    print("salvo em", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfp_fp", "cfp_ff", "lfw"])
    ap.add_argument("--examples", action="store_true", help="figura de exemplos das duas bases")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=None, help="nome do arquivo em results/ (padrão: pipeline_<base>.png)")
    args = ap.parse_args()
    cfg = load_config()
    if args.examples:
        return dataset_examples(cfg)
    if not args.dataset:
        ap.error("--dataset é obrigatório sem --examples")
    c = cfg["preprocess"]
    pairs = load_pairs(cfg, args.dataset)
    g = pairs[pairs.tipo == "genuine"].sample(args.n, random_state=cfg["seed"])
    keys = [k for a, b in zip(g.amostra_1, g.amostra_2) for k in (a, b)][: 2 * args.n]
    prep = FacePreprocessor(cfg, get_device(cfg))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    cols = ["Bruta", "Detecção + landmarks", "Bbox (principal)", "Alinhada (ablação)", "Normalizada"]
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
            main = al if c["align"] else bb
            axes[i, 2].imshow(bb)
            axes[i, 3].imshow(al)
            norm = (np.asarray(main, dtype=np.float32) - 127.5) / 128.0
            axes[i, 4].imshow(np.clip((norm + 1) / 2, 0, 1))
        else:
            axes[i, 1].set_xlabel("sem detecção (FTA)", fontsize=7)
        axes[i, 0].set_ylabel(key.split("/")[-3] if args.dataset .startswith("cfp") else key.split("/")[0], fontsize=7)
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=8)
    fig.tight_layout()
    out = path(cfg, "results") / (args.out or f"pipeline_{args.dataset}.png")
    fig.savefig(out, dpi=150)
    print("salvo em", out)


if __name__ == "__main__":
    main()
