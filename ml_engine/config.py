from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
JOBS_ROOT = DATA_ROOT / "jobs"
WEIGHTS_ROOT = DATA_ROOT / "weights"

REALESRGAN_X2_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
    "RealESRGAN_x2plus.pth"
)
REALESRGAN_X2_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"


@dataclass(frozen=True)
class Preset:
    name: str
    max_minutes: int
    max_steps: int
    validation_every: int
    batch_size: int
    patch_size: int


PRESETS = {
    "quick": Preset("quick", 15, 1_000, 100, 2, 128),
    "balanced": Preset("balanced", 60, 5_000, 250, 2, 192),
    "quality": Preset("quality", 240, 20_000, 500, 2, 256),
}

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
