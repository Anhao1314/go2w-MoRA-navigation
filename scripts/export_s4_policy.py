#!/usr/bin/env python3
"""Export one S4 actor from an RSL-RL checkpoint to ONNX (+ SHA-256 manifest).

S4 Sim2Sim evaluates the Isaac-trained policies under onnxruntime in a plain
MuJoCo environment (no torch needed at evaluation time). This script bridges
the two: it loads an RSL-RL checkpoint, exports the actor with the
zero-weight-change chain from ``locomotion.sim2sim.policy_export`` and prints
the manifest (checkpoint/ONNX SHA-256, forward-parity stats).

Presets pin the (num_obs, num_actions) pair so a wrong dimension fails fast
instead of silently loading a mismatched state dict:

- ``s0_flat``      48 → 12  M1 velocity policy (S0 of the layered group)
- ``s1_layered``   53 → 2   upper-level sequence policy (commands vx, wz)
- ``single_layer`` 53 → 12  end-to-end sequence policy

Usage::

    python scripts/export_s4_policy.py --preset s0_flat \
        --checkpoint logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02/model_1499.pt \
        --out logs/s4/actors/s0_flat.onnx
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.sim2sim.policy_export import SUPPORTED_PRESETS, export_actor_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "usage").splitlines()[0])
    parser.add_argument("--preset", choices=sorted(SUPPORTED_PRESETS), required=True,
                        help="which S4 actor to export (pins obs/action dims)")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path,
                        help="path to the RSL-RL checkpoint (model_*.pt)")
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="output ONNX path; a .manifest.json sidecar is written next to it")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset version (default: policy_export.DEFAULT_OPSET)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    num_obs, num_actions = SUPPORTED_PRESETS[args.preset]
    kwargs = {} if args.opset is None else {"opset_version": args.opset}
    manifest = export_actor_onnx(
        checkpoint=args.checkpoint,
        output_path=args.out,
        num_obs=num_obs,
        num_actions=num_actions,
        **kwargs,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
