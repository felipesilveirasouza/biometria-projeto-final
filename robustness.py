"""Experimento D: robustez a blur e oclusão em 4 níveis.

Só a segunda amostra de cada par é degradada; a primeira continua limpa,
como uma referência cadastrada. O threshold de cada fold é o da avaliação
limpa (validação limpa), congelado: a degradação é só condição de teste.

- blur: filtro gaussiano na imagem BRUTA, antes da detecção, então pode
  aumentar o FTA.
- oclusão: bloco cinza no recorte alinhado, em posição aleatória com seed fixa.

Saídas: results/robustness/<tipo>_<nível>/ (scores, métricas) e
results/robustness/summary.csv + robustness.png.

Uso: python robustness.py   (requer evaluate.py já rodado na base limpa)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (cosine_scores, get_device, load_config, load_embeddings, load_pairs, path, rates_at,
                    set_seed, threshold_free, variant_name)
from extract_embeddings import extract
from model import build_model
from preprocess import FacePreprocessor, gaussian_blur, occlusion, run


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    r = cfg["robustness"]
    ds, base_variant = r["dataset"], variant_name(cfg)
    pairs = load_pairs(cfg, ds)
    labels = (pairs["tipo"] == "genuine").to_numpy().astype(int)
    folds = pairs["fold"].to_numpy()
    clean_tag = f"{ds}_{base_variant}"
    fold_thr = pd.read_csv(path(cfg, "results") / clean_tag / "folds.csv").set_index("fold")["threshold"]
    clean = load_embeddings(cfg, base_variant, ds)
    probes = sorted(set(pairs["amostra_2"])) if r["apply_to"] == "second" else sorted(
        set(pairs.amostra_1) | set(pairs.amostra_2))

    device = get_device(cfg)
    prep, model = FacePreprocessor(cfg, device), build_model(cfg, device)

    levels = ["original", "baixa", "média", "alta"]
    plans = [("blur", s, {"pre": gaussian_blur(s)}) for s in r["blur_sigmas"]] + \
            [("oclusao", f, {"post": occlusion(f, cfg["seed"])}) for f in r["occlusion_fracs"]]

    rows = []
    for kind, value, fns in plans:
        lvl = levels[(r["blur_sigmas"] if kind == "blur" else r["occlusion_fracs"]).index(value)]
        variant = f"{base_variant}_{kind}_{value}"
        if value == 0:
            emb = clean
            fta = None
        else:
            q = run(cfg, ds, probes, variant=variant, prep=prep, **fns)
            fta = float(q["fta"].mean())
            extract(cfg, ds, variant, model, device)
            deg = load_embeddings(cfg, variant, ds)
            # referência limpa para a 1ª amostra, degradada para a 2ª
            emb = {**clean, **{f"__deg__{k}": v for k, v in deg.items()}}
        p = pairs.copy()
        if value != 0:
            p["amostra_2"] = "__deg__" + p["amostra_2"]
        scores = cosine_scores(emb, p)
        per_fold = [rates_at(scores[folds == k], labels[folds == k], fold_thr[k]) for k in sorted(set(folds))]
        tf = threshold_free(scores, labels)
        row = {"perturbacao": kind, "nivel": lvl, "parametro": value,
               "EER": tf["EER"], "AUC": tf["AUC"], "TAR@FAR=1%": tf["TAR@FAR=1%"],
               "FMR": float(np.mean([x["FMR"] for x in per_fold])),
               "FNMR": float(np.mean([x["FNMR"] for x in per_fold])),
               "accuracy": float(np.mean([x["accuracy"] for x in per_fold])),
               "FTA_probe": fta}
        rows.append(row)
        out = path(cfg, "results") / "robustness" / f"{kind}_{value}"
        out.mkdir(parents=True, exist_ok=True)
        sc = pairs[["amostra_1", "amostra_2", "tipo"]].copy()
        sc["score"], sc["fold"] = np.round(scores, 6), folds
        sc.to_csv(out / "scores.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False))
        print(row)

    df = pd.DataFrame(rows)
    out = path(cfg, "results") / "robustness"
    df.to_csv(out / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric in zip(axes, ["EER", "FNMR"]):
        for kind, g in df.groupby("perturbacao"):
            ax.plot(range(len(g)), g[metric], marker="o", label=kind)
        ax.set_xticks(range(4), levels)
        ax.set_ylabel(metric + (" (threshold congelado)" if metric == "FNMR" else ""))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Robustez — {ds}")
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
