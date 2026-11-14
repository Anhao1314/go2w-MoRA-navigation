"""Frozen-seed S4 evaluation, reusing ``locomotion.eval.metrics`` directly.

The frozen S4 constraints (``locomotion/sim2sim/README.md``) forbid a second
metric implementation on the MuJoCo side, so every number reported here is
computed by ``locomotion.eval.metrics`` / ``locomotion.eval.seeds`` — the same
functions the Isaac-side evaluator calls.  This module only collects arrays
from the MuJoCo rollout and feeds them in:

- velocity task: per-step ``(time_s, cmd_vx, meas_vx, cmd_wz, meas_wz)``, then
  ``vx_rmse`` / ``wz_rmse`` / ``fall_rate`` (thresholds: vx ≤ 0.15, wz ≤ 0.2);
- sequence task: per-episode ``(no fall, terminal within ±0.5 m of P2)``, then
  ``success_rate`` (threshold ≥ 0.90), plus stage-reach and fall rates.

Summary JSONs mirror ``scripts/eval_policy.py``'s schema (keys and shapes), so
Isaac-side and MuJoCo-side results are directly comparable; ``method`` records
which engine produced the file.  Each evaluation run also writes an evidence
bundle (``run_config.json`` with git/dependency/actor provenance +
``sha256_manifest.json``) following the repository convention.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import pathlib
import subprocess
import time as _time
from typing import Any

import numpy as np

from ..eval.metrics import fall_rate, recovery_time, success_rate, vx_rmse, wz_rmse
from ..eval.seeds import EVAL_SEEDS
from .go2_env import Go2Sim2SimEnv, Sim2SimPolicy

EVIDENCE_SCHEMA_VERSION = "locomotion-run-evidence-v1"
SEQUENCE_TARGET_TOLERANCE_M = 0.5  # frozen S4 terminal-position window


def _git_info(root: pathlib.Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        return {"available": True, "commit": commit, "dirty": bool(status), "dirty_files": len(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False, "commit": None, "branch": None, "dirty": None, "dirty_files": None}


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence_bundle(run_dir: pathlib.Path, config: dict[str, object]) -> None:
    """Write ``run_config.json`` + ``sha256_manifest.json`` over ``run_dir``.

    Follows the ``locomotion-run-evidence-v1`` convention of
    ``scripts/run_evidence.py`` (finalize phase): every file in the bundle is
    hashed, the manifest excludes itself.
    """
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **config,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "phase": "finalized",
        "finalized_at": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    files: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            files[path.relative_to(run_dir).as_posix()] = _sha256_file(path)
    manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "phase": "finalize",
        "generated_at": run_config["finalized_at"],
        "run_dir": str(run_dir),
        "file_count": len(files),
        "files": files,
    }
    (run_dir / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _heading(env: Go2Sim2SimEnv) -> float:
    rotation = env.data.xmat[env._base_body_id].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


# --- velocity task -----------------------------------------------------------
def run_velocity_episode(
    env: Go2Sim2SimEnv, policy: Sim2SimPolicy
) -> dict[str, Any]:
    """One frozen-seed episode; arrays are fed into the shared metrics later."""
    obs = env.reset()
    time_s, cmd_vx, cmd_wz, meas_vx, meas_wz = [], [], [], [], []
    while not env.over:
        command = env.command
        time_s.append(env.step_count * 0.02)
        cmd_vx.append(float(command[0]))
        cmd_wz.append(float(command[2]))
        meas_vx.append(float(obs[0]))
        meas_wz.append(float(obs[5]))
        env.step(policy.act(obs))
        obs = env.observation()
    t = np.asarray(time_s, dtype=float)
    return {
        "seed": env.seed,
        "steps": env.step_count,
        "fall": bool(env.fell),
        "time_s": t,
        "commanded_vx": np.asarray(cmd_vx, dtype=float),
        "measured_vx": np.asarray(meas_vx, dtype=float),
        "commanded_wz": np.asarray(cmd_wz, dtype=float),
        "measured_wz": np.asarray(meas_wz, dtype=float),
    }


def evaluate_velocity(
    policy: Sim2SimPolicy,
    *,
    seeds: tuple[int, ...] = EVAL_SEEDS,
    checkpoint_label: str,
) -> dict[str, Any]:
    """Velocity-task summary mirroring ``eval_policy.py``'s JSON schema."""
    started = _time.time()
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = Go2Sim2SimEnv("velocity", seed=int(seed))
        result = run_velocity_episode(env, policy)
        episodes.append(result)

    vx_values, wz_values = [], []
    for row in episodes:
        vx_values.append(vx_rmse(row["time_s"], row["commanded_vx"], row["measured_vx"]))
        wz_values.append(wz_rmse(row["time_s"], row["commanded_wz"], row["measured_wz"]))
    vx = np.asarray(vx_values, dtype=float)
    wz = np.asarray(wz_values, dtype=float)

    episode_rows = []
    for row, vr, wr in zip(episodes, vx_values, wz_values):
        if row["time_s"].size and row["time_s"][-1] > 0.6:
            post = row["time_s"] > 0.5
            mean_cmd = float(np.mean(row["commanded_vx"][post]))
            mean_meas = float(np.mean(row["measured_vx"][post]))
        else:
            mean_cmd = mean_meas = None
        episode_rows.append(
            {
                "seed": row["seed"],
                "steps": row["steps"],
                "fall": row["fall"],
                "vx_rmse": float(vr),
                "wz_rmse": float(wr),
                "mean_cmd_vx": mean_cmd,
                "mean_vx": mean_meas,
            }
        )

    summary = {
        "episodes": len(episodes),
        "seeds": [int(s) for s in seeds],
        "vx_rmse_mean": float(np.mean(vx)),
        "vx_rmse_std": float(np.std(vx, ddof=0)),
        "wz_rmse_mean": float(np.mean(wz)),
        "wz_rmse_std": float(np.std(wz, ddof=0)),
        "fall_rate": fall_rate([bool(row["fall"]) for row in episodes]),
        "checkpoint": checkpoint_label,
        "task": "Go2Loco-S1-Flat-Blind-v0",
        "method": "mujoco",
        "wall_time_s": round(_time.time() - started, 3),
        "generated_at": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
    }
    return {"summary": summary, "episodes": episode_rows}


# --- sequence task -----------------------------------------------------------
def run_sequence_episode(
    env: Go2Sim2SimEnv, policy: Sim2SimPolicy
) -> dict[str, Any]:
    obs = env.reset()
    max_stage = 0
    while not env.over:
        env.step(policy.act(obs))
        obs = env.observation()
        max_stage = max(max_stage, env.stage)
        if env.over:
            break
    _, p2 = env._stage.targets()
    terminal = env.base_position[0:2]
    within_target = bool(float(np.linalg.norm(terminal - np.asarray(p2))) <= SEQUENCE_TARGET_TOLERANCE_M)
    return {
        "seed": env.seed,
        "steps": env.step_count,
        "success": bool(env.completed),
        "fall": bool(env.fell),
        "max_stage": int(max_stage),
        "terminal_pos": [float(terminal[0]), float(terminal[1])],
        "target_p2": [float(p2[0]), float(p2[1])],
        "within_target": within_target,
    }




# --- ② S2 disturbance robustness (protocol amendment: MuJoCo transfer engine) ---
def run_disturbance_episode(
    env: Go2Sim2SimEnv, policy: Sim2SimPolicy
) -> dict[str, Any]:
    """Velocity episode with the frozen S2 push protocol (100 N / 0.5 s at 5 s)."""
    obs = env.reset()
    time_s, cmd_vx, meas_vx = [], [], []
    while not env.over:
        command = env.command
        time_s.append(env.step_count * 0.02)
        cmd_vx.append(float(command[0]))
        meas_vx.append(float(obs[0]))
        env.step(policy.act(obs))
        obs = env.observation()
    t = np.asarray(time_s, dtype=float)
    return {
        "seed": env.seed,
        "steps": env.step_count,
        "fall": bool(env.fell),
        "time_s": t,
        "commanded_vx": np.asarray(cmd_vx, dtype=float),
        "measured_vx": np.asarray(meas_vx, dtype=float),
    }


def evaluate_disturbance(
    policy: Sim2SimPolicy,
    *,
    seeds: tuple[int, ...] = EVAL_SEEDS,
    checkpoint_label: str,
    push_force_n: float = 100.0,
) -> dict[str, Any]:
    """S2 disturbance robustness: no-fall rate vs the frozen >= 0.90 gate.

    Recovery time uses the shared metrics implementation (single source).
    """
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = Go2Sim2SimEnv(
            "velocity", seed=int(seed),
            push_force_n=push_force_n, push_start_s=5.0, push_duration_s=0.5,
        )
        episodes.append(run_disturbance_episode(env, policy))

    no_fall = [not bool(row["fall"]) for row in episodes]
    recovery = []
    for row in episodes:
        rt = recovery_time(
            row["time_s"], row["commanded_vx"], row["measured_vx"],
            disturbance_start_s=5.0,
        )
        recovery.append(rt)
    finite = [float(v) for v in recovery if v is not None]

    summary = {
        "episodes": len(episodes),
        "seeds": [int(s) for s in seeds],
        "no_fall_rate": float(np.mean(no_fall)),
        "gate_no_fall_min": 0.90,
        "gate_pass": bool(np.mean(no_fall) >= 0.90),
        "recovery_time_mean_s": float(np.mean(finite)) if finite else None,
        "recovery_time_max_s": float(np.max(finite)) if finite else None,
        "recovery_never_count": sum(1 for v in recovery if v is None),
        "fall_rate": fall_rate([bool(row["fall"]) for row in episodes]),
        "checkpoint": checkpoint_label,
        "task": "S2-disturbance-velocity",
        "method": "mujoco",
        "push": {"force_n": push_force_n, "duration_s": 0.5, "start_s": 5.0},
        "engine": "mujoco_transfer_engine",
    }
    episode_rows = []
    for row in episodes:
        rt = recovery_time(
            row["time_s"], row["commanded_vx"], row["measured_vx"],
            disturbance_start_s=5.0,
        )
        episode_rows.append({
            "seed": row["seed"],
            "steps": row["steps"],
            "fall": row["fall"],
            "recovery_time_s": rt,
        })
    return {"summary": summary, "episodes": episode_rows}


# --- ③ unseen-task generalization (P4) ---
def run_generalization_episode(
    env: Go2Sim2SimEnv, policy: Sim2SimPolicy
) -> dict[str, Any]:
    obs = env.reset()
    max_stage = 0
    while not env.over:
        env.step(policy.act(obs))
        obs = env.observation()
        max_stage = max(max_stage, env.stage)
        if env.over:
            break
    return {
        "seed": env.seed,
        "steps": env.step_count,
        "success": bool(env.completed),
        "fall": bool(env.fell),
        "max_stage": int(max_stage),
    }


def evaluate_generalization(
    policy: Sim2SimPolicy,
    *,
    group_label: str,
    checkpoint_label: str,
    geometries: dict[str, tuple[float, float]],
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> dict[str, Any]:
    """Unseen-geometry sequence evaluation: success per variant, layered vs single."""
    per_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_name, geometry in geometries.items():
        rows = []
        for seed in seeds:
            env = Go2Sim2SimEnv(
                "sequence", seed=int(seed), sequence_geometry=geometry,
            )
            rows.append(run_generalization_episode(env, policy))
        per_variant[variant_name] = rows

    variant_summary = {}
    all_rows = []
    for variant_name, rows in per_variant.items():
        successes = sum(1 for r in rows if r["success"])
        falls = sum(1 for r in rows if r["fall"])
        finished = [int(r["steps"]) for r in rows if r["success"]]
        variant_summary[variant_name] = {
            "success_rate": successes / len(rows),
            "fall_rate": falls / len(rows),
            "mean_steps_to_finish": float(np.mean(finished)) if finished else None,
        }
        for r in rows:
            all_rows.append({"variant": variant_name, **r})

    total = sum(len(rows) for rows in per_variant.values())
    total_success = sum(sum(1 for r in rows if r["success"]) for rows in per_variant.values())
    summary = {
        "episodes_total": total,
        "success_rate_overall": total_success / max(1, total),
        "per_variant": variant_summary,
        "checkpoint": checkpoint_label,
        "task": "Go2Loco-Sequence-unseen-geometry",
        "method": "mujoco",
        "group": group_label,
        "engine": "mujoco_transfer_engine",
    }
    return {"summary": summary, "episodes": all_rows}


def evaluate_sequence_group(
    policy: Sim2SimPolicy,
    *,
    group_label: str,
    checkpoint_label: str,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> dict[str, Any]:
    """Sequence-task summary; success uses the shared ``success_rate``."""
    started = _time.time()
    episodes = []
    for seed in seeds:
        env = Go2Sim2SimEnv("sequence", seed=int(seed))
        episodes.append(run_sequence_episode(env, policy))

    no_fall = [not bool(row["fall"]) for row in episodes]
    within = [bool(row["within_target"]) for row in episodes]
    s4_rate = success_rate(no_fall, within)
    stage_reach = {
        str(stage): float(np.mean([row["max_stage"] >= stage + 1 for row in episodes]))
        for stage in range(4)
    }
    finished_steps = [int(row["steps"]) for row in episodes if row["success"]]
    summary = {
        "episodes": len(episodes),
        "seeds": [int(s) for s in seeds],
        "success_rate": s4_rate,
        "stage_reach_rate": stage_reach,
        "mean_steps_to_finish": float(np.mean(finished_steps)) if finished_steps else None,
        "fall_rate": fall_rate([bool(row["fall"]) for row in episodes]),
        "episode_length_steps": Go2Sim2SimEnv("sequence", seed=0).episode_steps,
        "checkpoint": checkpoint_label,
        "task": "Go2Loco-Sequence-v0",
        "method": "mujoco",
        "group": group_label,
        "wall_time_s": round(_time.time() - started, 3),
        "generated_at": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
    }
    return {"summary": summary, "episodes": episodes}
