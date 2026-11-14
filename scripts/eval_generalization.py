#!/usr/bin/env python3
"""Generalisation test: evaluate trained policies on unseen task geometries.

Both groups were trained with left turns (+90 deg) over 2 m legs only.  This
script re-evaluates the frozen checkpoints on geometries they have never seen
(right turn, longer legs, both) by patching the task constants at runtime --
the MDP reads them per step, so no retraining and no code change is needed.

Configs (per group):
    baseline      turn +90, leg 2.0 m   (seen in training)
    right-turn    turn -90, leg 2.0 m   (unseen direction)
    long-leg      turn +90, leg 3.0 m   (unseen length)
    right+long    turn -90, leg 3.0 m   (both unseen)

The raw environment is built ONCE per process and shared by both groups (the
layered group adds a wrapper on top).  Creating a second env after
``vec_env.close()`` in the same Kit session stalled for ~35 min on 2026-10-27
(the original single-layer timeout): ``ManagerBasedEnv.close()`` tears down
the scene and clears the sim context, and a subsequent ``gym.make`` hangs on
the destroyed world.  Closing is therefore deferred to the very end.

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/eval_generalization.py \
        --single-checkpoint <A model.pt> --layered-checkpoint <B model.pt> \
        --s0-checkpoint <S0 model.pt> --episodes 20 --out <json> \
        [--groups layered,single]

Results are written incrementally (``"partial": true``) after every config so
a killed run still leaves its completed evaluations on disk.

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
from typing import Any, cast

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from locomotion.eval.seeds import EVAL_SEEDS  # noqa: E402

CONFIGS = (
    ("baseline(+90deg,2m)", 1.5707963267948966, 2.0),
    ("right-turn(-90deg,2m)", -1.5707963267948966, 2.0),
    ("long-leg(+90deg,3m)", 1.5707963267948966, 3.0),
    ("right+long(-90deg,3m)", -1.5707963267948966, 3.0),
)

VALID_GROUPS = ("layered", "single")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-Sequence-v0")
    parser.add_argument("--single-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--layered-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--s0-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument(
        "--groups",
        default="layered,single",
        help="comma-separated subset of {layered, single} (default: both)",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    groups = [name.strip() for name in args.groups.split(",") if name.strip()]
    unknown = [name for name in groups if name not in VALID_GROUPS]
    if unknown:
        parser.error(f"unknown groups: {unknown}; valid: {VALID_GROUPS}")
    seeds = list(EVAL_SEEDS[: args.episodes])

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import locomotion.isaac  # noqa: F401
    import locomotion.sequence as seq  # noqa: F401
    from locomotion.sequence import mdp as seq_mdp
    from locomotion.sequence.layered_env import LayeredSequenceEnv, load_s0_actor
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    device = "cuda:0"

    results: list[dict[str, Any]] = []

    def write_out(partial: bool) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"results": results}
        if partial:
            payload["partial"] = True
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def evaluate(runner: OnPolicyRunner, inner: Any, vec_env: Any, steps_max: int) -> float:
        num_envs = vec_env.num_envs
        inner._seq_success_log = set()
        active = np.ones(num_envs, dtype=bool)
        with torch.inference_mode():
            policy = runner.get_inference_policy(device=device)
            obs, _ = vec_env.reset()
            for step in range(steps_max):
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                dones_np = (
                    dones.cpu().numpy().astype(bool)
                    if torch.is_tensor(dones)
                    else np.asarray(dones, dtype=bool)
                )
                active &= ~dones_np
                if not active.any():
                    break
                if (step + 1) % 250 == 0:
                    print(
                        f"[gen]   step {step + 1}/{steps_max},"
                        f" active {int(active.sum())}/{num_envs}",
                        flush=True,
                    )
        return len(getattr(inner, "_seq_success_log", set())) / num_envs

    num_envs = len(seeds)

    # one raw env for the whole session (see module docstring: a second
    # gym.make after close() stalls the Kit session)
    build_started = time.time()
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=num_envs, use_fabric=True)
    env_cfg.seed = 0
    raw_env = gym.make(args.task, cfg=env_cfg)
    inner = cast(Any, raw_env.unwrapped)
    steps_max = int(inner.max_episode_length)
    print(
        f"[gen] raw env ready in {time.time() - build_started:.1f}s"
        f" (steps_max={steps_max})",
        flush=True,
    )

    vec_env: Any = None
    for group in groups:
        group_started = time.time()
        # shared env: data buffers may hold inference tensors from earlier
        # in-inference-mode resets, so every later reset (including the one
        # inside the wrapper constructor) must also run inside inference_mode
        with torch.inference_mode():
            if group == "layered":
                s0_actor = load_s0_actor(args.s0_checkpoint, device=device)
                wrapped: Any = LayeredSequenceEnv(raw_env, s0_actor, device=device)
            else:
                wrapped = raw_env
            vec_env = RslRlVecEnvWrapper(wrapped)
            if group == "layered":
                vec_env.num_actions = int(gym.spaces.flatdim(wrapped.action_space))
            runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device=device)
            checkpoint = args.layered_checkpoint if group == "layered" else args.single_checkpoint
            runner.load(str(checkpoint))
        print(f"[gen] {group}: runner ready in {time.time() - group_started:.1f}s", flush=True)

        for label, turn, distance in CONFIGS:
            # runtime patch: the MDP reads these constants on every step, so a
            # reset under the patched values fully re-geometries the task
            seq_mdp.TURN_RADIANS = turn
            seq_mdp.FORWARD_DISTANCE_M = distance
            seen = "seen" if label.startswith("baseline") else "UNSEEN"
            started = time.time()
            rate = evaluate(runner, inner, vec_env, steps_max)
            results.append(
                {
                    "group": group,
                    "config": label,
                    "seen": seen,
                    "success_rate": round(rate, 3),
                    "episodes": num_envs,
                    "eval_time_s": round(time.time() - started, 1),
                }
            )
            write_out(partial=True)
            print(
                f"[gen] {group:>7} | {label:<20} ({seen}) ->"
                f" success {rate * 100:5.1f}% ({results[-1]['eval_time_s']}s)",
                flush=True,
            )

    if vec_env is not None:
        vec_env.close()
    write_out(partial=False)
    print(f"[gen] wrote {args.out}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
