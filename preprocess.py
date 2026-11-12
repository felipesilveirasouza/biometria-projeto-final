"""Etapas antes da CNN: controle de qualidade → detecção → alinhamento → recorte.

  1. Controle de qualidade: a imagem precisa abrir e a MTCNN precisa achar um
     rosto com probabilidade >= min_face_prob. Caso contrário, conta como falha
     de aquisição (FTA).
  2. Detecção e localização: MTCNN (bbox + 5 landmarks: olhos, nariz, cantos da boca).
  3. Alinhamento: transformação de similaridade (Umeyama) que leva os 5 landmarks
     ao template padrão de face; com align = false usa só a bbox (ablação).
  4. Recorte para image_size x image_size (160).
  A normalização de intensidade, (x − 127.5) / 128, é feita em model.py.

Os recortes ficam em cache/crops/<variante>/<base>/<caminho>.png.

Uso: python preprocess.py --dataset cfp_fp [--no-align]
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import dataset_root, get_device, load_config, load_pairs, pair_images, path, variant_name

# Template de 5 pontos para 112x112 (o mesmo usado por ArcFace/InsightFace).
TEMPLATE_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def template(size: int, scale: float) -> np.ndarray:
    return (TEMPLATE_112 - 56.0) * scale * (size / 112.0) + size / 2.0


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similaridade (rotação + escala uniforme + translação) que leva src a dst. Retorna 3x3."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / s.var(0).sum()
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = mu_d - scale * R @ mu_s
    return T


def align(img: Image.Image, points: np.ndarray, size: int, scale: float) -> Image.Image:
    M = umeyama(points.astype(np.float64), template(size, scale))
    inv = np.linalg.inv(M)  # PIL mapeia saída -> entrada
    return img.transform((size, size), Image.AFFINE, inv[:2].flatten().tolist(), resample=Image.BILINEAR)


def bbox_crop(img: Image.Image, box: np.ndarray, size: int, margin: int) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1, y1 = max(int(x1 - margin / 2), 0), max(int(y1 - margin / 2), 0)
    x2, y2 = min(int(x2 + margin / 2), w), min(int(y2 + margin / 2), h)
    return img.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)


def center_crop(img: Image.Image, size: int, frac: float = 0.6) -> Image.Image:
    w, h = img.size
    side = int(min(w, h) * frac)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)


class FacePreprocessor:
    def __init__(self, cfg: dict, device: str):
        from facenet_pytorch import MTCNN

        self.cfg = cfg["preprocess"]
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, img: Image.Image):
        """Melhor rosto: o de maior área entre os detectados. None se nenhum."""
        boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
        if boxes is None:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ok = probs >= self.cfg["min_face_prob"]
        pool = np.where(ok)[0] if ok.any() else np.arange(len(boxes))
        i = pool[np.argmax(areas[pool])]
        return boxes[i], float(probs[i]), points[i]

    def __call__(self, img: Image.Image) -> tuple[Image.Image, dict]:
        c, size = self.cfg, self.cfg["image_size"]
        det = self.detect(img)
        rec = {"detected": det is not None, "prob": det[1] if det else 0.0}
        rec["fta"] = (det is None) or (det[1] < c["min_face_prob"])
        if det is None:
            return center_crop(img, size), {**rec, "method": "center_fallback"}
        box, _, pts = det
        if c["align"]:
            ok, why = plausible_landmarks(img, box, pts, size, c)
            if ok:
                return align(img, pts, size, c["align_scale"]), {**rec, "method": "landmarks"}
            # Perfis: a MTCNN costuma inventar landmarks fora do rosto e o template
            # frontal gera um recorte girado e vazio. Nesses casos usa a bbox.
            return bbox_crop(img, box, size, c["bbox_margin"]), {**rec, "method": f"bbox_fallback ({why})"}
        return bbox_crop(img, box, size, c["bbox_margin"]), {**rec, "method": "bbox"}


def plausible_landmarks(img: Image.Image, box: np.ndarray, pts: np.ndarray, size: int, c: dict) -> tuple[bool, str]:
    """Landmarks utilizáveis para o alinhamento? Dentro da imagem e da bbox, e rotação moderada."""
    w, h = img.size
    if (pts[:, 0] < 0).any() or (pts[:, 1] < 0).any() or (pts[:, 0] >= w).any() or (pts[:, 1] >= h).any():
        return False, "landmark fora da imagem"
    x1, y1, x2, y2 = box
    tol = 0.05 * max(x2 - x1, y2 - y1)
    if ((pts[:, 0] < x1 - tol) | (pts[:, 0] > x2 + tol) | (pts[:, 1] < y1 - tol) | (pts[:, 1] > y2 + tol)).any():
        return False, "landmark fora da bbox"
    (lx, _), (rx, _), (nx, _) = pts[0], pts[1], pts[2]
    if not (min(lx, rx) <= nx <= max(lx, rx)):
        return False, "nariz fora do intervalo dos olhos (perfil)"
    M = umeyama(pts.astype(np.float64), template(size, c["align_scale"]))
    angle = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if angle > c["max_align_rotation"]:
        return False, f"rotação {angle:.0f}°"
    return True, ""


# --------------------------------------------------------------- degradações

def gaussian_blur(sigma: float) -> Callable[[Image.Image, str], Image.Image]:
    return lambda img, key: img if sigma <= 0 else img.filter(ImageFilter.GaussianBlur(sigma))


def occlusion(frac: float, seed: int) -> Callable[[Image.Image, str], Image.Image]:
    """Bloco cinza quadrado em posição aleatória, determinística por imagem."""
    def fn(img: Image.Image, key: str) -> Image.Image:
        if frac <= 0:
            return img
        rng = np.random.default_rng(seed + zlib.crc32(key.encode()))
        a = np.array(img).copy()
        h, w = a.shape[:2]
        side = int(round(frac * min(h, w)))
        y, x = rng.integers(0, h - side + 1), rng.integers(0, w - side + 1)
        a[y:y + side, x:x + side] = 127
        return Image.fromarray(a)
    return fn


# ---------------------------------------------------------------- execução

def crops_dir(cfg: dict, variant: str, dataset: str) -> Path:
    return path(cfg, "cache") / "crops" / variant / dataset


def run(cfg: dict, dataset: str, keys: list[str], variant: str | None = None,
        pre: Callable | None = None, post: Callable | None = None,
        prep: FacePreprocessor | None = None) -> pd.DataFrame:
    """Processa `keys` e salva os recortes. `pre` degrada a imagem bruta (antes da
    detecção); `post` degrada o recorte. Retorna o registro de qualidade."""
    variant = variant or variant_name(cfg)
    root, out = dataset_root(cfg, dataset), crops_dir(cfg, variant, dataset)
    qfile = out / "quality.csv"
    done = pd.read_csv(qfile) if qfile.exists() else pd.DataFrame(columns=["amostra"])
    done_set = set(done["amostra"])
    todo = [k for k in keys if k not in done_set]
    recs = []
    if todo:
        prep = prep or FacePreprocessor(cfg, get_device(cfg))
        for key in tqdm(todo, desc=f"{dataset}/{variant}"):
            try:
                img = Image.open(root / key).convert("RGB")
            except Exception as e:  # noqa: BLE001
                recs.append({"amostra": key, "detected": False, "prob": 0.0, "fta": True,
                             "method": f"erro_leitura: {e}"})
                continue
            if pre:
                img = pre(img, key)
            crop, rec = prep(img)
            if post:
                crop = post(crop, key)
            dst = out / (key.rsplit(".", 1)[0] + ".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst)
            recs.append({"amostra": key, **rec})
    q = pd.concat([done, pd.DataFrame(recs)], ignore_index=True) if recs else done
    out.mkdir(parents=True, exist_ok=True)
    q.to_csv(qfile, index=False)
    return q[q["amostra"].isin(set(keys))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfp_fp", "lfw"], required=True)
    ap.add_argument("--no-align", action="store_true", help="ablação: recorte só pela bbox")
    args = ap.parse_args()
    cfg = load_config()
    if args.no_align:
        cfg["preprocess"]["align"] = False
    keys = pair_images(load_pairs(cfg, args.dataset))
    q = run(cfg, args.dataset, keys)
    print(f"{len(q)} imagens | FTA = {q['fta'].mean():.4f} ({int(q['fta'].sum())}) | "
          f"métodos: {q['method'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
