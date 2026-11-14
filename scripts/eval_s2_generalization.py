#!/usr/bin/env python3
"""Run the two eval-only protocol extensions on the MuJoCo transfer engine:

② S2 disturbance robustness (protocol §4, frozen gate no-fall ≥ 0.90):
   the M1 velocity policy with a 100 N / 0.5 s lateral push at t = 5 s.
③ Unseen-geometry generalization (P4): the L3-trained layered and
   single-layer policies evaluated on sequence geometries they never saw
   ((3 m, 90°), (2 m, 60°), (2 m, 120°)) — the rule layer adapts the
   commands; the end-to-end policy must re-learn them implicitly.

Both reuse locomotion.eval metrics (single source) and write evidence
bundles per run.  Protocol amendment note: S2/geometry runs execute on the
MuJoCo transfer engine, mirroring S4's zero-change principle; documented in
the S4 report changelog.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.sim2sim.evaluate_sim2sim import (
    evaluate_disturbance,
    evaluate_generalization,
    write_evidence_bundle,
)
from locomotion.sim2sim.go2_env import Sim2SimPolicy

ACTORS = PROJECT_ROOT / "logs" / "s4" / "actors"
import math

GEOMETRIES = {
    "trained_2m_90deg": (2.0, math.radians(90.0)),    # 训练几何（基线对照）
    "unseen_3m_90deg": (3.0, math.radians(90.0)),     # 更远
    "unseen_2m_60deg": (2.0, math.radians(60.0)),     # 更小转角
    "unseen_2m_120deg": (2.0, math.radians(120.0)),   # 更大转角
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "s4" / "s2_generalization")
    args = parser.parse_args(argv)

    seeds = tuple(range(args.seeds))
    s0 = ACTORS / "s0_flat.onnx"
    s1 = ACTORS / "s1_layered.onnx"
    single = ACTORS / "single_layer.onnx"
    actor_sha = {p.name: pathlib.Path(str(p) + ".manifest.json") and json.loads(
        pathlib.Path(str(p) + ".manifest.json").read_text(encoding="utf-8"))["onnx"]["sha256"]
        for p in (s0, s1, single) if pathlib.Path(str(p) + ".manifest.json").exists()
    }

    print("=== ② S2 disturbance robustness (s0_flat, 100 N / 0.5 s @ 5 s) ===", flush=True)
    policy_s0 = Sim2SimPolicy(mode="s0", s0_path=s0)
    s2 = evaluate_disturbance(policy_s0, seeds=seeds, checkpoint_label="s0_flat_seed0")
    print(json.dumps(s2["summary"], indent=2, ensure_ascii=False), flush=True)

    print("=== ③ unseen-geometry generalization ===", flush=True)
    gen_results = {}
    for group_label, policy, actor_key in (
        ("layered", Sim2SimPolicy(mode="layered", s0_path=s0, s1_path=s1), "layered"),
        ("single", Sim2SimPolicy(mode="single", single_path=single), "single"),
    ):
        print(f"--- {group_label} ---", flush=True)
        gen_results[group_label] = evaluate_generalization(
            policy, group_label=group_label,
            checkpoint_label=f"{group_label}_seed0",
            geometries=GEOMETRIES, seeds=seeds,
        )
        print(json.dumps(gen_results[group_label]["summary"], indent=2,
                         ensure_ascii=False), flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "s2_disturbance": s2,
        "generalization_layered": gen_results["layered"],
        "generalization_single": gen_results["single"],
    }
    (out / "evaluation.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    config = {
        "run_id": out.name,
        "suite": "S2-disturbance + unseen-geometry generalization",
        "method": "mujoco_transfer_engine",
        "protocol_note": "S2 replicated on the transfer engine (mirrors S4 zero-change); "
                         "documented as protocol amendment in the S4 report changelog",
        "seeds": list(seeds),
        "actors": actor_sha,
        "geometries": {k: list(v) for k, v in GEOMETRIES.items()},
    }
    write_evidence_bundle(out, config)
    print("bundle:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
