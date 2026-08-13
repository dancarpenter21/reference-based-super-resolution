from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime


def now() -> str:
    return datetime.now(UTC).isoformat()


def probe_rocm() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable: torch.cuda.is_available() is false")
    device_count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    if "AMD" not in name.upper() and "RADEON" not in name.upper():
        raise RuntimeError(f"Expected an AMD ROCm GPU, found {name}")
    return {
        "device": "cuda:0",
        "device_count": device_count,
        "name": name,
        "hip_version": torch.version.hip,
        "torch_version": torch.__version__,
    }


class GPUDiagnostics:
    def __init__(self, probe: Callable[[], dict] = probe_rocm):
        self.probe = probe
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = {
            "state": "checking",
            "device": None,
            "device_count": None,
            "name": None,
            "hip_version": None,
            "torch_version": None,
            "error": None,
            "last_checked_at": None,
        }

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start_check(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._status["state"] = "checking"
            self._status["error"] = None
            self._thread = threading.Thread(target=self._check, name="refsr-gpu-check", daemon=True)
            self._thread.start()
            return True

    def _check(self) -> None:
        try:
            details = self.probe()
            status = {
                "state": "available",
                "device": details.get("device"),
                "device_count": details.get("device_count"),
                "name": details.get("name"),
                "hip_version": details.get("hip_version"),
                "torch_version": details.get("torch_version"),
                "error": None,
                "last_checked_at": now(),
            }
        except Exception as error:
            status = {
                "state": "unavailable",
                "device": None,
                "device_count": None,
                "name": None,
                "hip_version": None,
                "torch_version": None,
                "error": f"{type(error).__name__}: {error}",
                "last_checked_at": now(),
            }
        with self._lock:
            self._status = status


gpu_diagnostics = GPUDiagnostics()
