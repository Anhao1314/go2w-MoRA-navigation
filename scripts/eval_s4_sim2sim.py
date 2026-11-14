#!/usr/bin/env python3
"""Run the frozen-seed S4 evaluation of an exported actor in MuJoCo.

Reuses ``locomotion.eval`` metrics (single source of truth, frozen protocol)
and writes an evidence bundle per run: ``evaluation.json`` (Isaac-compatible
schema with ``method: "mujoco"``), ``run_config.json`` (git + dependency +
actor provenance) and ``sha256_manifest.json``.

Usage::

    python scripts/eval_s4_sim2sim.py --task velocity --policy s0 \
        --label s0_flat
    python scripts/eval_s4_sim2sim.py --task sequence --policy layered \
        --label layered_B
    python scripts/eval_s4_sim2sim.py --task sequence --policy single \
        --label single_A
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.sim2sim.evaluate_sim2sim import (
    evaluate_sequence_group,
    evaluate_velocity,
    write_evidence_bundle,
)
from locomotion.sim2sim.go2_env import Sim2SimPolicy

ACTOR_DIR = PROJECT_ROOT / "logs" / "s4" / "actors"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "usage").splitlines()[0])
    parser.add_argument("--task", choices=["velocity", "sequence"], required=True)
    parser.add_argument("--policy", choices=["s0", "single", "layered"], required=True,
                        help="s0: 48-dim velocity policy; single: 53->12; layered: S1+S0 chain")
    parser.add_argument("--label", required=True,
                        help="checkpoint label recorded in the summary and report")
    parser.add_argument("--seeds", type=int, default=20,
                        help="number of frozen seeds (default 20 = E={0..19})")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="run directory (default: logs/s4/<task>_<policy>_<utcstamp>)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    s0 = ACTOR_DIR / "s0_flat.onnx"
    s1 = ACTOR_DIR / "s1_layered.onnx"
    single = ACTOR_DIR / "single_layer.onnx"
    if args.policy == "s0":
        policy = Sim2SimPolicy(mode="s0", s0_path=s0)
    elif args.policy == "single":
        policy = Sim2SimPolicy(mode="single", single_path=single)
    else:
        policy = Sim2SimPolicy(mode="layered", s0_path=s0, s1_path=s1)

    seeds = tuple(range(args.seeds))
    if args.task == "velocity":
        if args.policy != "s0":
            raise SystemExit("velocity task requires --policy s0")
        result = evaluate_velocity(policy, seeds=seeds, checkpoint_label=args.label)
        actor_files = {"s0_flat": s0}
    else:
        if args.policy == "s0":
            raise SystemExit("sequence task requires --policy layered or single")
        result = evaluate_sequence_group(
            policy, group_label=args.policy, checkpoint_label=args.label, seeds=seeds
        )
        actor_files = (
            {"s0_flat": s0, "s1_layered": s1}
            if args.policy == "layered"
            else {"single_layer": single}
        )

    out = args.out or PROJECT_ROOT / "logs" / "s4" / f"{args.task}_{args.policy}_{_utcstamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    actor_provenance = {}
    for name, path in actor_files.items():
        manifest_path = pathlib.Path(str(path) + ".manifest.json")
        actor_provenance[name] = {
            "onnx_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))["onnx"]["sha256"]
            if manifest_path.exists()
            else None,
            "path": str(path),
        }

    write_evidence_bundle(
        out,
        {
            "run_id": out.name,
            "started_at": result["summary"]["generated_at"],
            "task": result["summary"]["task"],
            "method": "mujoco",
            "policy_mode": args.policy,
            "label": args.label,
            "seeds": list(seeds),
            "engine": {"mujoco": __import__("mujoco").__version__,
                       "python": sys.version.split()[0],
                       "numpy": np.__version__},
            "actors": actor_provenance,
        },
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"bundle: {out}")
    return 0


def _utcstamp() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
