"""Gera artigo/numeros.tex (macros LaTeX) a partir de results/, para que todo número
do artigo venha direto dos arquivos de resultado. Resultado ausente vira ``??''.

Uso (da raiz do repositório): python artigo/gerar_numeros.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPO_URL = r"\todo{URL do repositório}"


def load(tag: str) -> dict:
    f = RES / tag / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else {}


def pct(x, d=2) -> str:
    return "??" if x is None else f"{100 * x:.{d}f}".replace(".", "{,}")


def num(x, d=3) -> str:
    return "??" if x is None else f"{x:.{d}f}".replace(".", "{,}")


def pm(m: dict, key: str) -> str:
    k = m.get("kfold", {}).get(key)
    return "??" if not k else f"{pct(k['mean'])} $\\pm$ {pct(k['std'])}"


def aligned_share(dataset: str) -> str:
    f = ROOT / "cache" / "crops" / "aligned" / dataset / "quality.csv"
    if not f.exists():
        return "??"
    q = pd.read_csv(f)
    return f"{pct((q['method'] == 'landmarks').mean(), 1)}\\%"


def main() -> None:
    macros: dict[str, str] = {"repourl": REPO_URL}
    # Principal = recorte pela bbox; "Aln" = ablação com alinhamento por landmarks.
    for name, tag in [("CFP", "cfp_fp_bbox"), ("LFW", "lfw_bbox"),
                      ("AlnCFP", "cfp_fp_aligned"), ("AlnLFW", "lfw_aligned")]:
        m = load(tag)
        macros |= {f"eer{name}": pct(m.get("EER")), f"auc{name}": num(m.get("AUC"), 4),
                   f"tar{name}": pct(m.get("TAR@FAR=1%")), f"fta{name}": pct(m.get("FTA")),
                   f"fmr{name}": pm(m, "FMR"), f"fnmr{name}": pm(m, "FNMR"),
                   f"acc{name}": pm(m, "accuracy")}
    for name, tag in [("CFP", "cfp_fp_bbox"), ("LFW", "lfw_bbox")]:
        t = RES / tag / "threshold.json"
        macros[f"thr{name}"] = num(json.loads(t.read_text())["threshold"]) if t.exists() else "??"
    c = load("cross_cfp_fp_to_lfw")
    macros |= {"fmrCross": pct(c.get("FMR")), "fnmrCross": pct(c.get("FNMR")),
               "accCross": pct(c.get("accuracy"))}
    macros |= {"alignCFP": aligned_share("cfp_fp"), "alignLFW": aligned_share("lfw")}

    e = RES / "efficiency.json"
    if e.exists():
        e = json.loads(e.read_text())
        t, h = e["timing"], e["hardware"]
        batched = next(v for k, v in t.items() if k.startswith("ms_per_image_batch") and k != "ms_per_image_batch1")
        macros |= {"paramsM": num(e["params_millions"], 2), "pesoMB": num(e["weights_size_mb"], 1),
                   "msBatchOne": num(t["ms_per_image_batch1"], 1), "msBatch": num(batched, 2),
                   "cpu": h["cpu"], "gpu": h["gpu"], "torchv": h["torch"].replace("+", r"\texttt{+}"),
                   "pythonv": h["python"]}
    else:
        macros |= {k: "??" for k in ["paramsM", "pesoMB", "msBatchOne", "msBatch", "cpu", "gpu", "torchv", "pythonv"]}

    out = Path(__file__).with_name("numeros.tex")
    out.write_text("% Gerado por artigo/gerar_numeros.py. Não editar à mão.\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print("salvo em", out)


if __name__ == "__main__":
    main()
