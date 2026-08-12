# Reference-Based Super-Resolution

A local, single-GPU application that learns a conservative video upscaler from an incomplete higher-quality reference and applies it to every frame of a complete lower-quality video.

The application first tries to prove that the reference overlaps the complete source. If it cannot establish enough geometrically consistent matches, it reports that result and safely falls back to unpaired adaptation: high-quality reference frames provide the targets, while blur, noise, and compression are calibrated from the complete source. It never silently treats unrelated frames as ground truth.

## Current workflow

1. Upload the complete low-resolution video and the incomplete high-resolution reference.
2. Probe both streams and search for reliable overlapping spans.
3. Fine-tune a pretrained RealESRGAN ×2 RRDB model without a GAN discriminator.
4. Upscale the complete stream to 640×480 (4:3), stabilize residual detail across frames, and restore the original audio.
5. Preview/download the MP4 and its JSON processing report.

The supplied fixtures exercise the fallback path: `ss-24-hi.mp4` is 720×480 at 24 fps with non-square pixels, while `ss-24-low.mp4` is 480×360 at 29.97 fps. Coarse visual and audio inspection did not establish a shared timeline.

## ROCm setup (RX 9070 XT / WSL2)

Requirements:

- Ubuntu 24.04 under WSL2
- ROCm 7.2 with `/dev/dxg` available
- AMD Radeon RX 9070 XT (`gfx1201`)
- Python 3.12, `uv`, Node.js 22, and `curl`

Install AMD's supported PyTorch 2.9.1 wheels and the project environment:

```bash
chmod +x scripts/install_rocm.sh
./scripts/install_rocm.sh
```

The script creates `.venv`, installs the ROCm 7.2 wheels published by AMD, applies AMD's required WSL runtime-library adjustment, and runs a GPU check. Validate again at any time with:

```bash
.venv/bin/python -m ml_engine.cli gpu-check
```

## Run the application

Backend (from the repository root):

```bash
.venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Jobs and generated artifacts live under `data/jobs/`; metadata lives in `data/jobs.sqlite3`. One durable worker consumes GPU jobs serially. Interrupted non-terminal jobs return to the queue when the backend restarts.

## CLI and diagnostics

Inspect media and alignment without training:

```bash
.venv/bin/python -m ml_engine.cli inspect test_resources/ss-24-low.mp4 test_resources/ss-24-hi.mp4
```

Run an end-to-end job without the web UI:

```bash
.venv/bin/python -m ml_engine.cli run \
  test_resources/ss-24-low.mp4 test_resources/ss-24-hi.mp4 \
  --preset balanced --output-dir data/cli-job
```

Presets cap fine-tuning at approximately 15 minutes (`quick`), 1 hour (`balanced`), or 4 hours (`quality`). Full-video inference takes additional time.

## Tests

```bash
.venv/bin/pytest
cd frontend && npm test -- --run
```

CPU tests cover media geometry, conservative alignment fallback, job persistence, APIs, and model dimensions. The ROCm check is intentionally separate so an environment problem cannot masquerade as an ML test failure.

The model implementation is compatible with the BSD-licensed Real-ESRGAN release weights and follows the Apache-licensed BasicSR RRDB layout. The pretrained asset is downloaded from the official v0.2.1 release and verified by SHA-256.
