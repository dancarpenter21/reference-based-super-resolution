# Reference-Based Super-Resolution

A local, single-GPU application that learns a conservative, video-specific upscaler from two overlapping versions of the same video: a high-resolution reference and a low-resolution supplement. Either input may contain footage the other does not.

The application builds a unified edit-sequence alignment, then pauses for frame-match review before it starts GPU training. Different constant frame rates are handled with independent frame indices and per-block mappings rather than one global offset. It never silently treats an automatic proposal as ground truth.

## Current workflow

1. Upload the low-resolution supplement and the high-resolution reference. Neither input is assumed to be complete.
2. Choose **Find and match shared frames** (recommended) or **Skip matching · reference only**.
3. Guided jobs probe both streams, create lightweight navigation proxies, and build an ordered map of shared and source-only footage. Reference-only jobs skip this work and go directly to unpaired adaptation.
4. For guided jobs, use the unified timeline to review automatic match drafts, jog either source independently, set exact Match In/Out frame pairs, and confirm contiguous shared segments.
5. Fine-tune a pretrained RealESRGAN ×2 RRDB model without a GAN discriminator.
6. Upscale the supplemental stream to 640×480 (4:3), stabilize residual detail across frames, and restore its original audio.
7. Preview/download the MP4 and its JSON processing report, including the selected matching workflow.

Review jobs survive browser and backend restarts and do not occupy the GPU queue. New reviews show the high-resolution reference above the low-resolution supplement on one edit-sequence axis: vertically aligned solid blocks are proposed or confirmed matches, while hatched difference blocks expose a reference-only intro, a supplemental-only tail, or unrelated footage between the same shared sections. Empty intervals are shown explicitly instead of stretching a shorter source to fill the gap. One shared transport scrubs or plays both navigation videos in sync. When only one source has footage at the playhead, the other preview displays an empty state.

The reference playback feed sits directly above the unified timeline and the supplemental feed directly below it, keeping both pictures and the shared playhead in one workspace. Click or drag directly across either timeline track to scrub both videos; there is no separate position slider. Each feed also has its own frame jog bar and nearby-frame search, so an automatic proposal that is a few frames off can be corrected without moving the other source.

Automatic analysis refines coarse samples against nearby full frames, then exposes every result as an editable Match In/Out draft. Position both feeds on the first corresponding images and mark Match In, then repeat for Match Out. Each mark is saved immediately; Apply changes the segment range, and Confirm approves that applied range for training. Taller labeled timeline bands distinguish automatic proposals, saved adjustments, applied user edits, confirmed matches, and unpaired footage. Two-sided difference blocks use the same controls to create new match segments. If either source drops frames or loses footage, applying or confirming rejects the duration mismatch: end the current segment before the discontinuity and create another after it instead of interpolating across the gap. Edits are revision-checked and preserve a complete, non-overlapping partition of both sources. Existing sparse-review jobs remain available in the legacy editor and can be explicitly rebuilt with the unified matcher.

Confirmed blocks produce a dense manifest containing every eligible supplemental frame; difference blocks never produce training pairs. Marking a block unpaired changes only that training relationship and never deletes source footage.

The current renderer follows the supplemental video's timeline. The coverage review now exposes high-resolution-only material instead of hiding it behind a “complete source” assumption, but assembling mutually exclusive ranges from both inputs into one ordered output requires an explicit edit/ordering plan and is not yet performed automatically.

The supplied fixtures include 720×480 references and a 480×360, 29.97 fps supplemental source. The matcher treats differing frame rates, intermittent omissions, and non-square reference pixels independently; the user remains the final authority on which proposed segments are safe for paired supervision.

Reference preprocessing samples the video to detect persistent near-black outer borders, applies one fixed active-picture crop, then performs sample-aspect correction and 4:3 normalization. A fixed video-level crop prevents border pixels from entering training patches without introducing per-frame framing jitter; videos without a persistent border are left unchanged.

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
