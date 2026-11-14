#!/usr/bin/env python3
"""Sweep intermediate checkpoints to measure sample efficiency.

Success rate at the end of training says nothing about how much training was
needed.  This evaluates a list of checkpoints of one run on the frozen seeds
and reports success rate per iteration, which is the direct evidence for the
sample-efficiency hypothesis (H1 in docs/RESEARCH.md).

One simulator session is reused for every checkpoint; only the policy weights
are reloaded.

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/sweep_checkpoints.py \
        --run-dir logs/rsl_rl_layered/2026-10-23_16-55-02 \
        --layered --s0-checkpoint logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02/model_1499.pt \
        --out reports/training/sweep_layered.json --plot media/training/sweep_layered.png

Exit codes: 0 = done, 1 = failure.

Runs only inside the Isaac Lab environment (``.venv311``).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from locomotion.eval.seeds import EVAL_SEEDS  # noqa: E402

DEFAULT_CHECKPOINTS = ("100", "200", "300", "500", "700", "1000", "1499")


def _as_bool(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=bool)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-Sequence-v0")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--checkpoints",
        default=",".join(DEFAULT_CHECKPOINTS),
        help="iteration numbers (model_<n>.pt) to evaluate",
    )
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--layered", action="store_true")
    parser.add_argument("--s0-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--plot", type=pathlib.Path, default=None)
    parser.add_argument("--label", default=None, help="series label for the plot")
    args = parser.parse_args(argv)

    iterations = [int(text) for text in args.checkpoints.split(",") if text.strip()]
    checkpoints = []
    for iteration in iterations:
        path = args.run_dir / f"model_{iteration}.pt"
        if path.is_file():
            checkpoints.append((iteration, path))
        else:
            print(f"[sweep] missing {path}, skipped")
    if not checkpoints:
        print("[sweep] no checkpoints found")
        return 1

    seeds = list(EVAL_SEEDS[: args.episodes])

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import locomotion.isaac  # noqa: F401
    import locomotion.sequence  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    num_envs = len(seeds)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=num_envs, use_fabric=True)
    env_cfg.seed = 0
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped  # type: ignore[assignment]

    if args.layered:
        if args.s0_checkpoint is None:
            raise SystemExit("--layered requires --s0-checkpoint")
        from locomotion.sequence.layered_env import LayeredSequenceEnv, load_s0_actor

        s0_actor = load_s0_actor(args.s0_checkpoint, device="cuda:0")
        env = LayeredSequenceEnv(env, s0_actor, device="cuda:0")

    vec_env = RslRlVecEnvWrapper(env)
    if args.layered:
        vec_env.num_actions = int(gym.spaces.flatdim(env.action_space))

    runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    steps_max = int(getattr(inner, "max_episode_length"))

    results: list[dict[str, Any]] = []
    for iteration, checkpoint in checkpoints:
        started = time.time()
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device="cuda:0")

        inner._seq_success_log = set()  # type: ignore[attr-defined]
        active = np.ones(num_envs, dtype=bool)
        # reset must run inside the same inference context: a previous step
        # leaves inference tensors in the scene buffers, and updating those in
        # place outside InferenceMode raises RuntimeError.
        with torch.inference_mode():
            obs, _ = vec_env.reset()
            for _ in range(steps_max):
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                newly_done = _as_bool(dones) & active
                if newly_done.any():
                    active[newly_done] = False
                if not active.any():
                    break

        successes = len(getattr(inner, "_seq_success_log", set()))
        rate = successes / num_envs
        elapsed = time.time() - started
        results.append(
            {"iteration": iteration, "success_rate": rate, "episodes": num_envs}
        )
        print(
            f"[sweep] iter {iteration:>5}: success {rate * 100:5.1f}% "
            f"({successes}/{num_envs}) in {elapsed:.1f}s",
            flush=True,
        )

    vec_env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": args.run_dir.as_posix(),
        "layered": bool(args.layered),
        "episodes": num_envs,
        "results": results,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sweep] wrote {args.out}", flush=True)

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        label = args.label or ("layered B" if args.layered else "single layer A")
        xs = [row["iteration"] for row in results]
        ys = [row["success_rate"] * 100 for row in results]
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=110)
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        ax.set_xlabel("training iterations")
        ax.set_ylabel("success rate [%]")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Sample efficiency on L1 (frozen seeds)")
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot)
        plt.close(fig)
        print(f"[sweep] wrote {args.plot}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
