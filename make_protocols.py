"""Gera protocols/<base>/ a partir dos protocolos OFICIAIS de CFP-FP e LFW.

Saídas por base:
  pairs.csv           todos os pares: fold, amostra_1, amostra_2, tipo, id_1, id_2
  genuine_pairs.csv   só os pares genuínos
  impostor_pairs.csv  só os pares impostores
  folds.csv           identidades presentes em cada fold
  summary.json        contagens e verificação de vazamento de identidade entre folds

No protocolo k-fold, para cada fold k o teste é o fold k e a validação
(definição do threshold) são os demais folds. Não há conjunto de treino porque
o modelo é usado com os pesos pré-treinados congelados.

Uso: python make_protocols.py [--dataset cfp_fp|cfp_ff|lfw]
"""
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

from common import dataset_root, load_config, path, write_json


def _read_index(f: Path) -> dict[int, str]:
    """Pair_list_F.txt / Pair_list_P.txt: '<índice> <caminho relativo a Protocol/>'."""
    idx = {}
    for line in f.read_text().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            idx[int(parts[0])] = parts[1]
    return idx


def _norm_cfp(rel: str) -> str:
    # '../Data/Images/001/frontal/01.jpg' -> 'Data/Images/001/frontal/01.jpg'
    return re.sub(r"^(\.\./)+", "", rel.replace("\\", "/"))


def cfp_fp(cfg: dict) -> pd.DataFrame:
    return _cfp(cfg, "FP")


def cfp_ff(cfg: dict) -> pd.DataFrame:
    """Protocolo frontal x frontal do CFP: mesmas pessoas, sem variação de pose.
    Serve de referência para isolar o efeito da pose (FF x FP)."""
    return _cfp(cfg, "FF")


def _cfp(cfg: dict, split: str) -> pd.DataFrame:
    root = dataset_root(cfg, "cfp_fp")
    proto = root / "Protocol"
    F = _read_index(proto / "Pair_list_F.txt")
    P = F if split == "FF" else _read_index(proto / "Pair_list_P.txt")
    rows = []
    split_dir = proto / "Split" / split
    folds = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if len(folds) != 10:
        raise SystemExit(f"Esperava 10 folds em {split_dir}, achei {len(folds)}")
    for k, d in enumerate(folds, start=1):
        for fname, tipo in (("same.txt", "genuine"), ("diff.txt", "impostor")):
            for line in (d / fname).read_text().split("\n"):
                nums = re.findall(r"\d+", line)
                if len(nums) != 2:
                    continue
                a, b = _norm_cfp(F[int(nums[0])]), _norm_cfp(P[int(nums[1])])
                rows.append({"fold": k, "amostra_1": a, "amostra_2": b, "tipo": tipo,
                             "id_1": a.split("/")[-3], "id_2": b.split("/")[-3]})
    return pd.DataFrame(rows)


def lfw(cfg: dict) -> pd.DataFrame:
    lines = path(cfg, "lfw_pairs").read_text().strip().split("\n")
    n_folds, n = map(int, lines[0].split())
    body = [l.split() for l in lines[1:]]
    if len(body) != n_folds * 2 * n:
        raise SystemExit(f"pairs.txt com {len(body)} linhas; esperava {n_folds * 2 * n}")
    img = lambda name, i: f"{name}/{name}_{int(i):04d}.jpg"
    rows = []
    for k in range(n_folds):
        block = body[k * 2 * n:(k + 1) * 2 * n]
        for p in block:
            if len(p) == 3:
                rows.append({"fold": k + 1, "amostra_1": img(p[0], p[1]), "amostra_2": img(p[0], p[2]),
                             "tipo": "genuine", "id_1": p[0], "id_2": p[0]})
            else:
                rows.append({"fold": k + 1, "amostra_1": img(p[0], p[1]), "amostra_2": img(p[2], p[3]),
                             "tipo": "impostor", "id_1": p[0], "id_2": p[2]})
    return pd.DataFrame(rows)


def check_and_save(cfg: dict, name: str, df: pd.DataFrame) -> None:
    root = dataset_root(cfg, name)
    missing = [p for p in sorted(set(df.amostra_1) | set(df.amostra_2)) if not (root / p).exists()]

    ids_per_fold = {k: set(g.id_1) | set(g.id_2) for k, g in df.groupby("fold")}
    leaks = {f"{a}-{b}": sorted(ids_per_fold[a] & ids_per_fold[b])
             for a, b in combinations(sorted(ids_per_fold), 2)
             if ids_per_fold[a] & ids_per_fold[b]}

    out = path(cfg, "protocols") / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pairs.csv", index=False)
    df[df.tipo == "genuine"].to_csv(out / "genuine_pairs.csv", index=False)
    df[df.tipo == "impostor"].to_csv(out / "impostor_pairs.csv", index=False)
    pd.DataFrame([{"fold": k, "identidade": i} for k, s in ids_per_fold.items() for i in sorted(s)]
                 ).to_csv(out / "folds.csv", index=False)

    summary = {
        "dataset": name,
        "folds": int(df.fold.nunique()),
        "pairs": len(df),
        "genuine": int((df.tipo == "genuine").sum()),
        "impostor": int((df.tipo == "impostor").sum()),
        "images": len(set(df.amostra_1) | set(df.amostra_2)),
        "identities": len(set(df.id_1) | set(df.id_2)),
        "missing_images": len(missing),
        "identity_disjoint_folds": not leaks,
        "folds_sharing_identities": {k: len(v) for k, v in leaks.items()},
    }
    write_json(summary, out / "summary.json")
    print(f"[{name}] {summary['pairs']} pares ({summary['genuine']} gen / {summary['impostor']} imp), "
          f"{summary['images']} imagens, {summary['identities']} identidades")
    print(f"[{name}] folds disjuntos por identidade: {summary['identity_disjoint_folds']}")
    if leaks:
        print(f"[{name}] ATENÇÃO: {len(leaks)} pares de folds compartilham identidades. "
              f"Isso afeta a separação validação/teste; registre no artigo.")
    if missing:
        print(f"[{name}] ATENÇÃO: {len(missing)} imagens não encontradas, ex.: {missing[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfp_fp", "cfp_ff", "lfw"], action="append")
    args = ap.parse_args()
    cfg = load_config()
    for name in args.dataset or ["cfp_fp", "cfp_ff", "lfw"]:
        check_and_save(cfg, name, {"cfp_fp": cfp_fp, "cfp_ff": cfp_ff, "lfw": lfw}[name](cfg))


if __name__ == "__main__":
    main()
