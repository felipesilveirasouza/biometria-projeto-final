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
REPO_URL = r"\url{https://github.com/felipesilveirasouza/biometria-projeto-final}"


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


def aligned_share_view(dataset: str, view: str) -> str:
    f = ROOT / "cache" / "crops" / "aligned" / dataset / "quality.csv"
    if not f.exists():
        return "??"
    q = pd.read_csv(f)
    q = q[q["amostra"].str.split("/").str[3] == view]
    return f"{pct((q['method'] == 'landmarks').mean(), 1)}\\%"


def main() -> None:
    macros: dict[str, str] = {"repourl": REPO_URL}
    # Principal = recorte pela bbox; "Aln" = ablação com alinhamento por landmarks.
    for name, tag in [("CFP", "cfp_fp_bbox"), ("LFW", "lfw_bbox"), ("FF", "cfp_ff_bbox"),
                      ("AlnCFP", "cfp_fp_aligned"), ("AlnLFW", "lfw_aligned")]:
        m = load(tag)
        macros |= {f"eer{name}": pct(m.get("EER")), f"auc{name}": num(m.get("AUC"), 4),
                   f"tar{name}": pct(m.get("TAR@FAR=1%")), f"fta{name}": pct(m.get("FTA")),
                   f"fmr{name}": pm(m, "FMR"), f"fnmr{name}": pm(m, "FNMR"),
                   f"acc{name}": pm(m, "accuracy")}
    for name, tag in [("CFP", "cfp_fp_bbox"), ("LFW", "lfw_bbox"), ("FF", "cfp_ff_bbox")]:
        t = RES / tag / "threshold.json"
        macros[f"thr{name}"] = num(json.loads(t.read_text())["threshold"]) if t.exists() else "??"
    c = load("cross_cfp_fp_to_lfw")
    macros |= {"fmrCross": pct(c.get("FMR")), "fnmrCross": pct(c.get("FNMR")),
               "accCross": pct(c.get("accuracy"))}
    macros |= {"alignCFP": aligned_share("cfp_fp"), "alignLFW": aligned_share("lfw")}

    # Ablação pareada por fold (acurácia bbox x alinhado) com teste de Wilcoxon.
    from scipy.stats import wilcoxon
    for name, ds in [("CFP", "cfp_fp"), ("LFW", "lfw")]:
        fb, fa = RES / f"{ds}_bbox" / "folds.csv", RES / f"{ds}_aligned" / "folds.csv"
        if fb.exists() and fa.exists():
            b = pd.read_csv(fb).set_index("fold").accuracy
            a = pd.read_csv(fa).set_index("fold").accuracy
            d = b - a
            macros |= {f"ablWins{name}": str(int((d > 0).sum())), f"ablTies{name}": str(int((d == 0).sum())),
                       f"ablDiff{name}": num(100 * d.mean(), 2), f"ablP{name}": num(wilcoxon(b, a).pvalue, 3)}

    # Cross-dataset: impostores aceitos com o threshold do CFP-FP e quantos o do LFW rejeitaria.
    f, tc, tl = RES / "cross_cfp_fp_to_lfw" / "scores.csv", RES / "cfp_fp_bbox" / "threshold.json", RES / "lfw_bbox" / "threshold.json"
    if f.exists() and tc.exists() and tl.exists():
        sc = pd.read_csv(f)
        tc, tl = json.loads(tc.read_text())["threshold"], json.loads(tl.read_text())["threshold"]
        fm = sc[(sc.tipo == "impostor") & (sc.score >= tc)]
        macros |= {"nFmCross": str(len(fm)), "nFmCrossLFW": str(int((fm.score >= tl).sum()))}

    # Médias dos scores genuínos e impostores (configuração principal).
    for name, tag in [("CFP", "cfp_fp_bbox"), ("LFW", "lfw_bbox"), ("FF", "cfp_ff_bbox")]:
        f = RES / tag / "scores.csv"
        mu = pd.read_csv(f).groupby("tipo").score.mean() if f.exists() else {}
        macros[f"muGen{name}"] = num(mu.get("genuine"), 2) if len(mu) else "??"
        macros[f"muImp{name}"] = num(mu.get("impostor"), 2) if len(mu) else "??"

    # FTA por vista no CFP-FP (frontal x perfil).
    f = ROOT / "cache" / "crops" / "bbox" / "cfp_fp" / "quality.csv"
    if f.exists():
        q = pd.read_csv(f)
        vista = q["amostra"].str.split("/").str[3]
        macros |= {"ftaFrontal": pct(q[vista == "frontal"].fta.mean()),
                   "ftaPerfil": pct(q[vista == "profile"].fta.mean()),
                   "landmarksPerfil": aligned_share_view("cfp_fp", "profile"),
                   "landmarksFrontal": aligned_share_view("cfp_fp", "frontal")}

    # Robustez: EER, FNMR (threshold congelado) e FTA da amostra degradada por nível.
    f = RES / "robustness" / "summary.csv"
    if f.exists():
        r = pd.read_csv(f)
        for kind, nm in [("blur", "Blur"), ("oclusao", "Ocl")]:
            sub = r[r.perturbacao == kind].reset_index(drop=True)
            for i, lvl in enumerate(["Orig", "Baixa", "Media", "Alta"]):
                macros[f"eer{nm}{lvl}"] = pct(sub.EER[i])
                macros[f"fnmr{nm}{lvl}"] = pct(sub.FNMR[i])
                macros[f"fmr{nm}{lvl}"] = pct(sub.FMR[i])
                macros[f"fta{nm}{lvl}"] = pct(sub.FTA_probe[i]) if pd.notna(sub.FTA_probe[i]) else "--"

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
