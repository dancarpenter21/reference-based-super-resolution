from __future__ import annotations

import argparse
import json

from .alignment import align_videos
from .media import probe
from .pipeline import run_pipeline
from .training.train import require_rocm


def main() -> None:
    parser = argparse.ArgumentParser(prog="refsr")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("low")
    inspect.add_argument("reference")
    run = commands.add_parser("run")
    run.add_argument("low")
    run.add_argument("reference")
    run.add_argument("--output-dir", default="data/cli-job")
    run.add_argument("--preset", choices=("quick", "balanced", "quality"), default="balanced")
    commands.add_parser("gpu-check")
    args = parser.parse_args()
    if args.command == "gpu-check":
        device = require_rocm()
        import torch
        print(json.dumps({"available": True, "device": str(device), "name": torch.cuda.get_device_name(0), "hip": torch.version.hip}))
    elif args.command == "inspect":
        low, ref = probe(args.low), probe(args.reference)
        print(json.dumps({"low": low.to_dict(), "reference": ref.to_dict(), "alignment": align_videos(args.low, args.reference, low, ref).to_dict()}, indent=2))
    else:
        report = run_pipeline(
            args.low, args.reference, args.output_dir, args.preset,
            update=lambda stage, p, message, metrics: print(f"[{stage}] {p:.1%} {message}"),
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
