"""Render the routed Markdown-bullet slider from a native RAW artifact."""
import argparse
import json
from pathlib import Path

import torch
from particle_sliders.inference import open_inference

from .bullet import INFERENCE


def main(argv=None):
    parser = argparse.ArgumentParser(prog="inkling2-bullet-render")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allocator-gib", type=float, default=24.)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    fraction = args.allocator_gib * 2**30 / torch.cuda.get_device_properties(device).total_memory
    if not 0 < fraction < 1:
        parser.error("allocator budget must be positive and smaller than total device memory")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.set_num_threads(4)
    with open_inference(args.raw, backend=INFERENCE, options={
        "model_dir": str(args.model_dir.resolve()), "device": args.device,
    }) as session:
        result = session.render(
            prompt=args.prompt, strength=args.strength, max_new_tokens=args.max_new_tokens,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
