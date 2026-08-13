import threading

import pytest
import torch

from app.services.gpu_diagnostics import GPUDiagnostics, probe_rocm


def wait_for_check(diagnostics):
    diagnostics._thread.join(timeout=2)
    assert diagnostics._thread.is_alive() is False


def test_gpu_diagnostics_reports_available_device():
    diagnostics = GPUDiagnostics(lambda: {
        "device": "cuda:0", "device_count": 1, "name": "AMD Radeon RX 9070 XT",
        "hip_version": "7.2", "torch_version": "2.9.1",
    })
    assert diagnostics.start_check() is True
    wait_for_check(diagnostics)

    status = diagnostics.status()
    assert status["state"] == "available"
    assert status["name"] == "AMD Radeon RX 9070 XT"
    assert status["last_checked_at"]


def test_gpu_diagnostics_reports_probe_error():
    def fail():
        raise RuntimeError("driver missing")

    diagnostics = GPUDiagnostics(fail)
    diagnostics.start_check()
    wait_for_check(diagnostics)

    assert diagnostics.status()["state"] == "unavailable"
    assert diagnostics.status()["error"] == "RuntimeError: driver missing"


def test_concurrent_gpu_recheck_is_idempotent():
    release = threading.Event()
    diagnostics = GPUDiagnostics(lambda: release.wait(2) or {})

    assert diagnostics.start_check() is True
    assert diagnostics.start_check() is False
    release.set()
    wait_for_check(diagnostics)


def test_probe_rocm_rejects_unavailable_or_non_amd_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="is_available"):
        probe_rocm()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA Test GPU")
    with pytest.raises(RuntimeError, match="Expected an AMD"):
        probe_rocm()
