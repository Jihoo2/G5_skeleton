from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.json"


def load_config() -> dict:
    """project.json을 읽어 Python 딕셔너리로 반환합니다."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_device(requested: str):
    """auto이면 CUDA 사용 가능 여부에 따라 GPU 또는 CPU를 선택합니다."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return 0
    return "cpu"
