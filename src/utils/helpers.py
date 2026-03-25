"""
General-purpose helpers.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_parameters(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M"


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "train.log")))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )


def get_device(cfg: dict) -> torch.device:
    preferred = cfg.get("project", {}).get("device", "cuda")
    if preferred == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)
