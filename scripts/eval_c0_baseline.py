#!/usr/bin/env python3
"""C0 scripted-PD baseline evaluation (RQ2): the protocol's lower bound.

Two suites on the same 20 frozen seeds and task definition as the RL evals:
1. flat velocity tracking (0.5 m/s): vx_rmse + falls — compared against the
   RL numbers (Isaac 0.0569, MuJoCo-transfer 0.0773, gate <= 0.15);
2. the S2 push scenario (100 N / 0.5 s at t = 5 s): no-fall rate — compared
   against the RL result (0/20 no-fall, gate >= 0.90 FAIL for untrained RL).

Evidence bundle per run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.eval.metrics import fall_rate, recovery_time, vx_rmse
from locomotion.sim2sim.c0_scripted import ScriptedTrotC0
from locomotion.sim2sim.evaluate_sim2sim import write_evidence_bundle
from locomotion.sim2sim.go2_env import Go2Sim2SimEnv


def run_c0_episode(env: Go2Sim2SimEnv, c0: ScriptedTrotC0) -> dict[str, Any]:
    obs = env.reset()
    time_s, cmd_vx, meas_vx = [], [], []
    while not env.over:
        command = env.command
        time_s.append(env.step_count * 0.02)
        cmd_vx.append(float(command[0]))
        meas_vx.append(float(obs[0]))
        env.step(c0.act(float(command[0]), float(command[2])))
        obs = env.observation()
    return {
        "seed": env.seed,
        "steps": env.step_count,
        "fall": bool(env.fell),
        "time_s": np.asarray(time_s, dtype=float),
        "commanded_vx": np.asarray(cmd_vx, dtype=float),
        "measured_vx": np.asarray(meas_vx, dtype=float),
    }


def evaluate_c0(
    seeds: tuple[int, ...],
    *,
    push_force_n: float,
    checkpoint_label: str,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = Go2Sim2SimEnv(
            "velocity", seed=int(seed),
            velocity_command=(0.5, 0.0),
            push_force_n=push_force_n, push_start_s=5.0, push_duration_s=0.5,
        )
        c0 = ScriptedTrotC0(env)
        episodes.append(run_c0_episode(env, c0))

    rmse = [
        vx_rmse(row["time_s"], row["commanded_vx"], row["measured_vx"])
        for row in episodes
    ]
    rmse_arr = np.asarray(rmse, dtype=float)
    no_fall = [not bool(row["fall"]) for row in episodes]
    recovery = [
        recovery_time(row["time_s"], row["commanded_vx"], row["measured_vx"],
                      disturbance_start_s=5.0)
        for row in episodes
    ]
    finite = [float(v) for v in recovery if v is not None]

    summary = {
        "episodes": len(episodes),
        "seeds": [int(s) for s in seeds],
        "vx_rmse_mean": float(np.mean(rmse_arr)),
        "vx_rmse_std": float(np.std(rmse_arr, ddof=0)),
        "fall_rate": fall_rate([bool(row["fall"]) for row in episodes]),
        "no_fall_rate": float(np.mean(no_fall)),
        "gate_pass": bool(np.mean(no_fall) >= 0.90),
        "recovery_time_mean_s": float(np.mean(finite)) if finite else None,
        "recovery_never_count": sum(1 for v in recovery if v is None),
        "checkpoint": checkpoint_label,
        "task": "S1-velocity + S2-push",
        "method": "mujoco_c0_scripted_pd",
        "push_force_n": push_force_n,
    }
    episode_rows = []
    for row, r in zip(episodes, rmse):
        episode_rows.append({
            "seed": row["seed"], "steps": row["steps"], "fall": row["fall"],
            "vx_rmse": float(r), "recovery_time_s": recovery[episodes.index(row)],
        })
    return {"summary": summary, "episodes": episode_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "usage").splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "s4" / "c0_baseline")
    args = parser.parse_args(argv)

    seeds = tuple(range(args.seeds))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=== C0 flat velocity tracking (0.5 m/s, 20 frozen seeds) ===", flush=True)
    flat = evaluate_c0(seeds, push_force_n=0.0, checkpoint_label="c0_scripted_trot")
    print(json.dumps({k: v for k, v in flat["summary"].items() if k != "results"},
                     indent=2, ensure_ascii=False), flush=True)

    print("=== C0 under the S2 push (100 N / 0.5 s @ 5 s) ===", flush=True)
    push = evaluate_c0(seeds, push_force_n=100.0, checkpoint_label="c0_scripted_trot_push")
    print(json.dumps({k: v for k, v in push["summary"].items() if k != "results"},
                     indent=2, ensure_ascii=False), flush=True)

    bundle = {"c0_flat": flat, "c0_push": push}
    (out / "evaluation.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    write_evidence_bundle(out, {
        "run_id": out.name,
        "suite": "C0 scripted-PD baseline (RQ2 lower bound)",
        "method": "mujoco_c0_scripted_pd",
        "seeds": list(seeds),
        "gait": {"type": "sine trot", "A_thigh_rad": 0.6, "A_calf_rad": 0.8,
                 "frequency_hz": 2.5, "in_phase_calf": True},
    })
    print("bundle:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
