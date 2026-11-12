"""Monta a tabela padronizada de resultados (§26) e a comparação dos experimentos.

Lê results/*/metrics.json e results/efficiency.json e escreve
results/tabela_padronizada.md e results/experimentos.csv.

Uso: python report.py
"""
from __future__ import annotations

import json

import pandas as pd

from common import load_config, path


def fmt(x, pct=False, nd=4):
    if x is None:
        return "n/d"
    return f"{100 * x:.2f}%" if pct else f"{x:.{nd}f}"


def main() -> None:
    cfg = load_config()
    res = path(cfg, "results")
    eff = json.loads((res / "efficiency.json").read_text()) if (res / "efficiency.json").exists() else {}
    rows = []
    for f in sorted(res.glob("*/metrics.json")):
        if f.parent.name == "robustness":
            continue
        m = json.loads(f.read_text())
        k = m.get("kfold", {})
        rows.append({
            "experimento": f.parent.name, "base": m["dataset"], "variante": m["variant"],
            "protocolo": m["protocol"], "EER": m["EER"], "AUC": m["AUC"], "TAR@FAR=1%": m["TAR@FAR=1%"],
            "threshold": m["threshold"], "FMR": m["FMR"], "FMR_std": k.get("FMR", {}).get("std"),
            "FNMR": m["FNMR"], "FNMR_std": k.get("FNMR", {}).get("std"),
            "accuracy": m["accuracy"], "FTA": m.get("FTA"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(res / "experimentos.csv", index=False)

    t = eff.get("timing", {})
    hw = eff.get("hardware", {})
    lines = ["# Tabela padronizada de resultados (§26)", ""]
    for r in rows:
        lines += [
            f"## {r['experimento']}", "",
            "| Informação | Resultado |", "|---|---|",
            "| Modalidade | Face |",
            f"| Dataset | {r['base']} |",
            "| CNN | Inception-ResNet-v1 (FaceNet) |",
            f"| Pré-treinamento | {eff.get('pretrained', cfg['model']['pretrained'])} (pesos congelados) |",
            f"| Nº de parâmetros | {eff.get('params_millions', 'n/d')} M |",
            "| Dimensão do embedding | 512 |",
            "| Matching | similaridade de cosseno (L2) |",
            f"| Threshold | {fmt(r['threshold'])} ({cfg['threshold']['rule']}) |",
            f"| EER | {fmt(r['EER'], True)} |",
            f"| AUC | {fmt(r['AUC'])} |",
            f"| FMR | {fmt(r['FMR'], True)}" + (f" ± {fmt(r['FMR_std'], True)}" if r['FMR_std'] is not None else "") + " |",
            f"| FNMR | {fmt(r['FNMR'], True)}" + (f" ± {fmt(r['FNMR_std'], True)}" if r['FNMR_std'] is not None else "") + " |",
            f"| TAR@FAR=1% | {fmt(r['TAR@FAR=1%'], True)} |",
            f"| FTA | {fmt(r['FTA'], True)} |",
            f"| Tempo de inferência | {t.get('ms_per_image_batch1', 'n/d')} ms/imagem (batch 1, {t.get('device', '?')}) |",
            f"| Tamanho do modelo | {eff.get('weights_size_mb', 'n/d')} MB |",
            f"| Hardware | {hw.get('gpu') or ''} / {hw.get('cpu', 'n/d')} |",
            "",
        ]
    (res / "tabela_padronizada.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print("\nTabela em", res / "tabela_padronizada.md")


if __name__ == "__main__":
    main()
