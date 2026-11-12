"""Análise de erros (§19): falsos matches e falsos non-matches.

Usa os scores e os thresholds por fold de uma avaliação já feita. Um falso
match é um par impostor com score >= threshold do seu fold; um falso
non-match é um par genuíno com score < threshold. Mostra os mais extremos
(impostores com maior score, genuínos com menor score).

Saídas em results/<tag>/: errors.csv e errors.png (imagem bruta + recorte
alinhado das duas amostras, score e threshold). A hipótese de causa de cada
erro vai na coluna 'hipotese' do CSV, preenchida à mão depois de olhar as imagens.

Uso: python errors.py --tag cfp_fp_aligned
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from PIL import Image

from common import dataset_root, load_config, path
from preprocess import crops_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config()
    res = path(cfg, "results") / args.tag
    m = json.loads((res / "metrics.json").read_text())
    ds, variant = m["dataset"], m["variant"]
    sc = pd.read_csv(res / "scores.csv")
    if (res / "folds.csv").exists():
        thr = pd.read_csv(res / "folds.csv").set_index("fold")["threshold"]
        sc["threshold"] = sc["fold"].map(thr)
    else:
        sc["threshold"] = m["threshold"]
    n = cfg["errors"]["per_type"]
    fm = sc[(sc.tipo == "impostor") & (sc.score >= sc.threshold)].nlargest(n, "score").assign(erro="falso match")
    fnm = sc[(sc.tipo == "genuine") & (sc.score < sc.threshold)].nsmallest(n, "score").assign(erro="falso non-match")
    err = pd.concat([fm, fnm], ignore_index=True)
    err["hipotese"] = ""
    err.to_csv(res / "errors.csv", index=False)
    print(f"{len(fm)} falsos matches e {len(fnm)} falsos non-matches selecionados")
    if err.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, cdir = dataset_root(cfg, ds), crops_dir(cfg, variant, ds)
    fig, axes = plt.subplots(len(err), 4, figsize=(9, 2.5 * len(err)), squeeze=False)
    for i, r in err.iterrows():
        for j, key in enumerate([r.amostra_1, r.amostra_2]):
            axes[i, 2 * j].imshow(Image.open(root / key).convert("RGB"))
            axes[i, 2 * j + 1].imshow(Image.open(cdir / (key.rsplit(".", 1)[0] + ".png")))
            axes[i, 2 * j].set_title(key.split("/")[-3] if ds == "cfp_fp" else key.split("/")[0], fontsize=7)
        axes[i, 0].set_ylabel(f"{r.erro}\nscore={r.score:.3f}\nthr={r.threshold:.3f}", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(res / "errors.png", dpi=150)


if __name__ == "__main__":
    main()
