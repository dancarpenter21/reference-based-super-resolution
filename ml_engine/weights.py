from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from .config import REALESRGAN_X2_SHA256, REALESRGAN_X2_URL, WEIGHTS_ROOT


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_pretrained(
    destination: str | Path | None = None,
    url: str = REALESRGAN_X2_URL,
    expected_sha256: str | None = REALESRGAN_X2_SHA256,
) -> Path:
    destination = Path(destination or WEIGHTS_ROOT / "RealESRGAN_x2plus.pth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_sha256 or sha256(destination) == expected_sha256):
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    actual = sha256(temporary)
    if expected_sha256 and actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Pretrained weight checksum mismatch: expected {expected_sha256}, got {actual}")
    os.replace(temporary, destination)
    return destination
