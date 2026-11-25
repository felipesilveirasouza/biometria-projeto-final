"""Funções compartilhadas: configuração, protocolos, métricas e gráficos."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: dict) -> str:
    import torch

    want = cfg.get("device", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def dataset_root(cfg: dict, dataset: str) -> Path:
    return path(cfg, {"cfp_fp": "cfp_fp_root", "cfp_ff": "cfp_fp_root", "lfw": "lfw_root"}[dataset])


def variant_name(cfg: dict) -> str:
    return "aligned" if cfg["preprocess"]["align"] else "bbox"


# ---------------------------------------------------------------- protocolos

def load_pairs(cfg: dict, dataset: str) -> pd.DataFrame:
    """Pares do protocolo oficial: fold, amostra_1, amostra_2, tipo, id_1, id_2."""
    f = path(cfg, "protocols") / dataset / "pairs.csv"
    if not f.exists():
        raise SystemExit(f"{f} não existe. Rode: python make_protocols.py")
    return pd.read_csv(f, dtype={"id_1": str, "id_2": str})


def pair_images(pairs: pd.DataFrame) -> list[str]:
    return sorted(set(pairs["amostra_1"]) | set(pairs["amostra_2"]))


def emb_file(cfg: dict, variant: str, dataset: str) -> Path:
    return path(cfg, "cache") / "embeddings" / variant / f"{dataset}.npz"


def load_embeddings(cfg: dict, variant: str, dataset: str) -> dict[str, np.ndarray]:
    f = emb_file(cfg, variant, dataset)
    if not f.exists():
        raise SystemExit(f"{f} não existe. Rode preprocess.py e extract_embeddings.py antes.")
    z = np.load(f, allow_pickle=False)
    return dict(zip(z["keys"].tolist(), z["emb"]))


# ------------------------------------------------------------------ métricas

def cosine_scores(emb: dict[str, np.ndarray], pairs: pd.DataFrame) -> np.ndarray:
    """Similaridade de cosseno entre embeddings já normalizados (L2).

    Amostra ausente (falha de aquisição com fta_policy = reject) recebe -1,
    ou seja, o par é sempre rejeitado.
    """
    out = np.empty(len(pairs), dtype=np.float64)
    for i, (a, b) in enumerate(zip(pairs["amostra_1"], pairs["amostra_2"])):
        if a not in emb or b not in emb:
            out[i] = -1.0
        else:
            out[i] = float(np.dot(emb[a], emb[b]))
    return out


def roc(scores: np.ndarray, labels: np.ndarray):
    """FMR, TMR e limiares (maior score = mais parecido)."""
    from sklearn.metrics import roc_curve

    fmr, tmr, thr = roc_curve(labels, scores)
    return fmr, tmr, thr


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Retorna (EER, limiar no ponto de EER)."""
    fmr, tmr, thr = roc(scores, labels)
    fnmr = 1 - tmr
    i = int(np.nanargmin(np.abs(fnmr - fmr)))
    return float((fmr[i] + fnmr[i]) / 2), float(thr[i])


def threshold_for(scores: np.ndarray, labels: np.ndarray, cfg: dict) -> float:
    rule = cfg["threshold"]["rule"]
    if rule == "eer":
        return eer(scores, labels)[1]
    if rule == "far":
        imp = np.sort(scores[labels == 0])[::-1]
        k = int(np.floor(cfg["threshold"]["target_far"] * len(imp)))
        # aceita score >= limiar: limiar logo acima do (k+1)-ésimo maior impostor,
        # de modo que no máximo k impostores (FAR <= alvo) sejam aceitos
        return float(np.nextafter(imp[min(k, len(imp) - 1)], np.inf))
    raise ValueError(f"regra de threshold desconhecida: {rule}")


def rates_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    """Decisão: aceita (genuíno) quando score >= limiar."""
    acc = scores >= thr
    gen, imp = labels == 1, labels == 0
    fnmr = float(np.mean(~acc[gen])) if gen.any() else float("nan")
    fmr = float(np.mean(acc[imp])) if imp.any() else float("nan")
    accuracy = float(np.mean(acc == (labels == 1)))
    return {"FMR": fmr, "FNMR": fnmr, "accuracy": accuracy}


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float) -> float | None:
    """TAR@FAR; None quando há poucos impostores para estimar esse FAR."""
    n_imp = int(np.sum(labels == 0))
    if n_imp * far < 10:  # menos de 10 falsos aceites esperados: estimativa instável
        return None
    fmr, tmr, _ = roc(scores, labels)
    return float(np.interp(far, fmr, tmr))


def threshold_free(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    e, t = eer(scores, labels)
    return {
        "EER": e,
        "EER_threshold": t,
        "AUC": float(roc_auc_score(labels, scores)),
        "TAR@FAR=1%": tar_at_far(scores, labels, 0.01),
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
    }


def kfold_protocol(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray, cfg: dict):
    """Protocolo oficial em k folds.

    Para cada fold k: limiar definido só nos outros folds (validação), congelado
    e aplicado ao fold k (teste). Nenhum score de teste influencia o limiar.
    """
    rows = []
    for k in sorted(np.unique(folds)):
        val, test = folds != k, folds == k
        thr = threshold_for(scores[val], labels[val], cfg)
        r = rates_at(scores[test], labels[test], thr)
        rows.append({"fold": int(k), "threshold": thr, **r})
    df = pd.DataFrame(rows)
    summary = {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))}
               for m in ["FMR", "FNMR", "accuracy", "threshold"]}
    return df, summary


# ------------------------------------------------------------------- gráficos

def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (fmr, tmr) in curves.items():
        ax.plot(fmr, tmr, label=name)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("FMR (escala log)")
    ax.set_ylabel("TMR = 1 − FNMR")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_distributions(scores: np.ndarray, labels: np.ndarray, thr: float, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-0.4, 1.0, 71)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, density=True, label="Impostor")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, density=True, label="Genuíno")
    ax.axvline(thr, color="k", ls="--", lw=1.2, label=f"Threshold = {thr:.3f}")
    ax.set_xlabel("Similaridade de cosseno")
    ax.set_ylabel("Densidade")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_json(obj, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
