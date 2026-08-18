from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml_engine.config import PRESETS, Preset
from ml_engine.dataset.dataset import ReferenceVideoDataset
from ml_engine.models.generator import RRDBNet, load_weights, output_4_by_3

Progress = Callable[[float, str, dict | None], None]


def require_rocm() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required, but torch.cuda.is_available() is false")
    name = torch.cuda.get_device_name(0)
    if "Radeon" not in name and "AMD" not in name:
        raise RuntimeError(f"Expected an AMD ROCm GPU, found {name}")
    return torch.device("cuda:0")


def charbonnier(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + 1e-6).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    psnrs: list[float] = []
    for batch in loader:
        lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        pred = output_4_by_3(model, lr)
        mse = F.mse_loss(pred, hr).item()
        losses.append(F.l1_loss(pred, hr).item())
        psnrs.append(-10.0 * __import__("math").log10(max(mse, 1e-10)))
    return {"l1": sum(losses) / len(losses), "psnr": sum(psnrs) / len(psnrs)}


def train_model(
    reference_path: str | Path,
    low_path: str | Path,
    pretrained_path: str | Path,
    output_dir: str | Path,
    preset_name: str = "balanced",
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    device: torch.device | None = None,
    model_factory: Callable[[], nn.Module] = RRDBNet,
    paired_manifest: list[dict] | None = None,
) -> tuple[Path, dict]:
    preset: Preset = PRESETS[preset_name]
    device = device or require_rocm()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model_factory().to(device)
    load_weights(model, pretrained_path, strict=True)
    train_set = ReferenceVideoDataset(
        reference_path,
        low_path,
        preset.patch_size,
        preset.max_steps * preset.batch_size,
        paired_manifest=paired_manifest,
    )
    validation_pairs = (paired_manifest or [])[::10]
    val_set = ReferenceVideoDataset(
        reference_path, low_path, preset.patch_size, 12, validation=True,
        paired_manifest=validation_pairs,
    )
    train_loader = DataLoader(train_set, batch_size=preset.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.99), weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.monotonic()
    best_path = output_dir / "best.pth"
    latest_path = output_dir / "latest.pth"
    baseline = validate(model, val_loader, device)
    best_score = baseline["l1"]
    history: list[dict] = [{"step": 0, "baseline": True, **baseline}]
    patience = 0
    start_step = 0
    if latest_path.exists():
        resumed = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        start_step = int(resumed["step"])
        best_score = float(resumed["best_score"])
        baseline = resumed["baseline"]
        history = resumed["history"]
        patience = int(resumed["patience"])
    model.train()
    for step, batch in enumerate(train_loader, start_step + 1):
        if step > preset.max_steps or time.monotonic() - started > preset.max_minutes * 60:
            break
        if cancel and cancel():
            raise InterruptedError("Training cancelled")
        lr, hr = batch["lr"].to(device, non_blocking=True), batch["hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = output_4_by_3(model, lr)
            loss = charbonnier(pred, hr) + 0.15 * gradient_loss(pred, hr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % preset.validation_every == 0 or step == preset.max_steps:
            metrics = validate(model, val_loader, device)
            metrics.update({"step": step, "train_loss": float(loss.item())})
            history.append(metrics)
            if metrics["l1"] < best_score:
                best_score = metrics["l1"]
                torch.save({"params_ema": model.state_dict(), "metrics": metrics}, best_path)
                patience = 0
            else:
                patience += 1
            model.train()
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))
            torch.save(
                {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best_score": best_score,
                    "baseline": baseline, "history": history, "patience": patience,
                },
                latest_path,
            )
            if progress:
                progress(min(0.99, step / preset.max_steps), f"step {step}", metrics)
            if patience >= 5:
                break
    selected = "fine_tuned"
    if not best_path.exists():
        selected = "pretrained"
        best_path = Path(pretrained_path)
    return best_path, {
        "history": history,
        "baseline": baseline,
        "selected": selected,
        "elapsed_seconds": time.monotonic() - started,
        "verified_pair_count": len(paired_manifest or []),
    }
