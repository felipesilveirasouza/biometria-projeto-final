"""Matching, scores, threshold e métricas.

Modo padrão (protocolo oficial em 10 folds): para cada fold k, o threshold é
definido nos outros 9 folds (validação), congelado e aplicado ao fold k (teste).
FMR/FNMR/acurácia são reportados como média ± desvio-padrão dos 10 folds.
EER, AUC, ROC e TAR@FAR=1% não dependem do threshold e usam todos os pares.

Modo cross-dataset (--frozen-from <tag>): o threshold global salvo pela
avaliação da base de origem é aplicado, sem nenhum ajuste, a todos os pares
da base de destino. Não há fine-tuning em nenhum momento (o modelo é congelado).

Saídas em results/<tag>/: scores.csv, folds.csv, metrics.json, threshold.json,
roc.png, distributions.png.

Exemplos:
  python evaluate.py --dataset cfp_fp                          # Experimento A
  python evaluate.py --dataset lfw                             # Experimento B (LFW, protocolo próprio)
  python evaluate.py --dataset lfw --frozen-from cfp_fp_aligned --tag cross_cfp_fp_to_lfw   # Experimento C
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (cosine_scores, kfold_protocol, load_config, load_embeddings, load_pairs, path,
                    plot_distributions, plot_roc, rates_at, roc, threshold_for, threshold_free, variant_name,
                    write_json)
from preprocess import crops_dir


def fta_stats(cfg: dict, dataset: str, variant: str, pairs: pd.DataFrame) -> dict:
    q = pd.read_csv(crops_dir(cfg, variant, dataset) / "quality.csv")
    q = q[q["amostra"].isin(set(pairs.amostra_1) | set(pairs.amostra_2))]
    bad = set(q.loc[q["fta"].astype(bool), "amostra"])
    affected = pairs.amostra_1.isin(bad) | pairs.amostra_2.isin(bad)
    return {"FTA": float(len(bad) / max(len(q), 1)), "fta_images": len(bad),
            "pairs_with_fta": int(affected.sum()), "fta_policy": cfg["preprocess"]["fta_policy"]}


def evaluate(cfg: dict, dataset: str, variant: str, tag: str, frozen_from: str | None = None,
             emb: dict | None = None, quiet: bool = False) -> dict:
    out = path(cfg, "results") / tag
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(cfg, dataset)
    emb = emb if emb is not None else load_embeddings(cfg, variant, dataset)
    scores = cosine_scores(emb, pairs)
    labels = (pairs["tipo"] == "genuine").to_numpy().astype(int)
    folds = pairs["fold"].to_numpy()

    sc = pairs[["amostra_1", "amostra_2", "tipo"]].copy()
    sc["score"] = np.round(scores, 6)
    sc["fold"] = folds
    sc.to_csv(out / "scores.csv", index=False)

    metrics = {"dataset": dataset, "variant": variant, "matching": "cosseno (embeddings L2-normalizados)",
               "threshold_rule": cfg["threshold"]["rule"], **threshold_free(scores, labels)}

    if frozen_from:
        thr = json.loads((path(cfg, "results") / frozen_from / "threshold.json").read_text())["threshold"]
        metrics.update({"protocol": f"cross-dataset: threshold congelado de '{frozen_from}'",
                        "threshold": thr, **rates_at(scores, labels, thr)})
        plot_thr = thr
    else:
        per_fold, summary = kfold_protocol(scores, labels, folds, cfg)
        per_fold.to_csv(out / "folds.csv", index=False)
        metrics.update({"protocol": "oficial 10 folds (threshold nos 9 folds de validação, teste no restante)",
                        "kfold": summary,
                        "threshold": summary["threshold"]["mean"],
                        "FMR": summary["FMR"]["mean"], "FNMR": summary["FNMR"]["mean"],
                        "accuracy": summary["accuracy"]["mean"]})
        plot_thr = summary["threshold"]["mean"]
        # Threshold global desta base, congelado para uso no cross-dataset.
        write_json({"dataset": dataset, "variant": variant, "rule": cfg["threshold"]["rule"],
                    "threshold": threshold_for(scores, labels, cfg)}, out / "threshold.json")

    try:
        metrics.update(fta_stats(cfg, dataset, variant, pairs))
    except FileNotFoundError:
        pass
    write_json(metrics, out / "metrics.json")

    fmr, tmr, _ = roc(scores, labels)
    np.savez(out / "roc.npz", fmr=fmr, tmr=tmr)
    plot_roc({f"{dataset} (AUC={metrics['AUC']:.4f})": (fmr, tmr)}, out / "roc.png", f"ROC — {tag}")
    plot_distributions(scores, labels, plot_thr, out / "distributions.png", f"Scores — {tag}")
    if not quiet:
        print(json.dumps({k: v for k, v in metrics.items() if k != "kfold"}, indent=2, ensure_ascii=False))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfp_fp", "cfp_ff", "lfw"], required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--tag", default=None, help="pasta em results/ (padrão: <base>_<variante>)")
    ap.add_argument("--frozen-from", default=None, help="tag cuja threshold.json será aplicada (cross-dataset)")
    args = ap.parse_args()
    cfg = load_config()
    variant = args.variant or variant_name(cfg)
    evaluate(cfg, args.dataset, variant, args.tag or f"{args.dataset}_{variant}", args.frozen_from)


if __name__ == "__main__":
    main()
