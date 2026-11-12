"""CNN: Inception-ResNet-v1 (FaceNet) com pesos pré-treinados, usada congelada.

- Implementação e pesos: facenet-pytorch (https://github.com/timesler/facenet-pytorch)
- Pré-treinamento: VGGFace2 (padrão) ou CASIA-WebFace, conforme config.yaml
- Camadas congeladas: todas (sem fine-tuning); camadas treinadas: nenhuma
- Função de perda no pré-treino: softmax de classificação de identidade (detalhes no README)
- Saída usada: embedding de 512 dimensões normalizado em L2 (a camada de
  classificação não é usada)
"""
from __future__ import annotations

import io
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def build_model(cfg: dict, device: str) -> torch.nn.Module:
    from facenet_pytorch import InceptionResnetV1

    m = InceptionResnetV1(pretrained=cfg["model"]["pretrained"], classify=False)
    # A camada de classificação das identidades do pré-treino não é usada (só o
    # embedding); removê-la deixa nº de parâmetros e tamanho fiéis ao sistema.
    if hasattr(m, "logits"):
        del m.logits
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def to_tensor(imgs: list[Image.Image]) -> torch.Tensor:
    """Normalização de intensidade usada no treino do FaceNet: (x − 127.5) / 128."""
    a = np.stack([np.asarray(i.convert("RGB"), dtype=np.float32) for i in imgs])
    t = torch.from_numpy(a).permute(0, 3, 1, 2)
    return (t - 127.5) / 128.0


@torch.inference_mode()
def embed(model: torch.nn.Module, imgs: list[Image.Image], device: str) -> np.ndarray:
    e = model(to_tensor(imgs).to(device))
    return F.normalize(e, dim=1).cpu().numpy()


def model_info(model: torch.nn.Module) -> dict:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    n = sum(p.numel() for p in model.parameters())
    return {
        "architecture": "Inception-ResNet-v1 (FaceNet)",
        "params": int(n),
        "params_millions": round(n / 1e6, 2),
        "weights_size_mb": round(buf.getbuffer().nbytes / 2**20, 1),
        "embedding_dim": 512,
    }


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().split("\n"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
