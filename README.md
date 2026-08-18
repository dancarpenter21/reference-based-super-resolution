# Reference-Based Super-Resolution

A local, single-GPU application that learns a conservative video upscaler from an incomplete higher-quality reference and applies it to every frame of a complete lower-quality video.

The application proposes chronological overlap segments, then pauses for frame-match review before it starts GPU training. Different constant frame rates are handled with independent frame indices and per-segment timestamp mappings rather than one global offset. It never silently treats an automatic proposal as ground truth.

## Current workflow

1. Upload the complete low-resolution video and the incomplete high-resolution reference.
2. Choose **Find and match shared frames** (recommended) or **Skip matching · reference only**.
3. Guided jobs probe both streams, create lightweight navigation proxies, and propose same-order shared sections with gaps. Reference-only jobs skip this work and go directly to unpaired adaptation.
4. For guided jobs, review every proposed section in the browser. Confirm or adjust its exact start/end frame pairs, reject it, or add a missing section from the two playheads.
5. Fine-tune a pretrained RealESRGAN ×2 RRDB model without a GAN discriminator.
6. Upscale the complete stream to 640×480 (4:3), stabilize residual detail across frames, and restore the original audio.
7. Preview/download the MP4 and its JSON processing report, including the selected matching workflow.

Review jobs survive browser and backend restarts and do not occupy the GPU queue. The review screen provides exact ±1/±10-frame stepping, timecodes, overlay comparison, segment splitting/merging, and keyboard stepping (left/right for the complete source, down/up for the reference; hold Shift for ten frames).

The supplied fixtures include 720×480 references and a 480×360, 29.97 fps complete source. The matcher treats differing frame rates, intermittent omissions, and non-square reference pixels independently; the user remains the final authority on which proposed segments are safe for paired supervision.

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

CPU tests cover media geometry, unequal-frame-rate segment alignment, durable review state, review APIs, conservative fallback, and model dimensions. The ROCm check is intentionally separate so an environment problem cannot masquerade as an ML test failure.

The model implementation is compatible with the BSD-licensed Real-ESRGAN release weights and follows the Apache-licensed BasicSR RRDB layout. The pretrained asset is downloaded from the official v0.2.1 release and verified by SHA-256.
