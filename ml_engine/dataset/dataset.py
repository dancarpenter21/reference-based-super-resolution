from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml_engine.media import MediaInfo, normalize_reference_frame, probe


def measure_degradation(video_path: str | Path, max_samples: int = 120) -> dict[str, float]:
    """Estimate source blur/noise/JPEG-like loss for synthetic reference degradations."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), min(max_samples, total), dtype=int)
    sharpness: list[float] = []
    noise: list[float] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        noise.append(float(np.median(np.abs(gray.astype(np.float32) - smooth))))
    cap.release()
    return {
        "sharpness": float(np.median(sharpness)) if sharpness else 50.0,
        "noise": float(np.median(noise)) if noise else 2.0,
        "jpeg_quality": 55.0,
    }


def degrade(hr: np.ndarray, profile: dict[str, float], rng: random.Random) -> np.ndarray:
    h, w = hr.shape[:2]
    target = (round(w * 3 / 4), round(h * 3 / 4))
    sigma = rng.uniform(0.2, 1.4 if profile["sharpness"] < 100 else 0.8)
    blurred = cv2.GaussianBlur(hr, (0, 0), sigma)
    lr = cv2.resize(blurred, target, interpolation=rng.choice([cv2.INTER_AREA, cv2.INTER_CUBIC]))
    noise_std = min(8.0, max(0.5, profile["noise"] * rng.uniform(0.5, 1.5)))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_std, lr.shape)
    lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    quality = int(rng.uniform(max(35, profile["jpeg_quality"] - 12), min(90, profile["jpeg_quality"] + 12)))
    ok, encoded = cv2.imencode(".jpg", lr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else lr


class ReferenceVideoDataset(Dataset):
    def __init__(
        self,
        reference_path: str | Path,
        low_path: str | Path,
        patch_size: int = 192,
        length: int = 10_000,
        validation: bool = False,
        paired_anchors: list[dict] | None = None,
    ):
        self.reference_path = str(reference_path)
        self.reference_info: MediaInfo = probe(reference_path)
        self.patch_size = patch_size - patch_size % 12
        self.length = length
        self.validation = validation
        self.profile = measure_degradation(low_path)
        self.frame_count = self.reference_info.frame_count
        self.low_path = str(low_path)
        self.low_info = probe(low_path)
        self.paired_anchors = paired_anchors or []
        self.cap: cv2.VideoCapture | None = None
        self.low_cap: cv2.VideoCapture | None = None

    def __len__(self) -> int:
        return self.length

    def _capture(self) -> cv2.VideoCapture:
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.reference_path)
        return self.cap

    def _low_capture(self) -> cv2.VideoCapture:
        if self.low_cap is None:
            self.low_cap = cv2.VideoCapture(self.low_path)
        return self.low_cap

    @staticmethod
    def _read(cap: cv2.VideoCapture, frame_index: int, label: str) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode {label} frame {frame_index}")
        return frame

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(index if self.validation else index + random.randrange(1_000_000_000))
        # Last 10% is validation-only; adjacent timeline leakage is avoided.
        split = int(self.frame_count * 0.9)
        lo, hi = (split, self.frame_count) if self.validation else (0, max(1, split))
        use_paired = bool(self.paired_anchors) and rng.random() < 0.5
        if use_paired:
            anchor = self.paired_anchors[index % len(self.paired_anchors)]
            ref_index = round(anchor["reference_time"] * self.reference_info.fps)
            low_index = round(anchor["low_time"] * self.low_info.fps)
            hr = normalize_reference_frame(self._read(self._capture(), ref_index, "reference"), self.reference_info)
            paired_lr = self._read(self._low_capture(), low_index, "low")
            paired_lr = cv2.resize(paired_lr, (480, 360), interpolation=cv2.INTER_AREA)
        else:
            frame_index = rng.randrange(lo, max(lo + 1, hi))
            hr = normalize_reference_frame(
                self._read(self._capture(), frame_index, "reference"), self.reference_info
            )
        p = self.patch_size
        top = rng.randrange(0, hr.shape[0] - p + 1)
        left = rng.randrange(0, hr.shape[1] - p + 1)
        hr = hr[top : top + p, left : left + p]
        if use_paired:
            low_p = p * 3 // 4
            low_top, low_left = top * 3 // 4, left * 3 // 4
            lr = paired_lr[low_top : low_top + low_p, low_left : low_left + low_p]
        if not self.validation and rng.random() < 0.5:
            hr = np.ascontiguousarray(hr[:, ::-1])
            if use_paired:
                lr = np.ascontiguousarray(lr[:, ::-1])
        if not use_paired:
            lr = degrade(hr, self.profile, rng)
        return {
            "lr": torch.from_numpy(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255,
            "hr": torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255,
        }
