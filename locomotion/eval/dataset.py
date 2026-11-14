"""Trajectory-dataset writer for ``go2-sequence-v1`` (docs/DATASET.md v1.1).

Pure numpy/json -- no Isaac imports -- so CI can unit-test the schema.

Layout (one episode = one compressed npz)::

    <root>/episodes/<group>_eval<seed:02d>.npz
    <root>/manifest.json      # schema + merged episode metadata + stats + splits
    <root>/checksums.sha256

``manifest.json`` merges across collection invocations (one invocation per
condition): entries are upserted by ``episode_id``, then totals, per-feature
statistics, machine-readable splits and checksums are recomputed.  Failure
episodes are kept and flagged, never filtered (robomimic-style "mask, don't
delete" -- the splits provide the filters).

Boundary semantics (RLDS conventions): ``obs[t] --action[t]--> obs[t+1]``;
``reward[t]`` is triggered by ``action[t]``; ``success``/``fall`` are
terminal, ``timeout`` is truncated.  ``foot_contact`` is derived from
``foot_contact_force`` at :data:`CONTACT_THRESHOLD_N`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

SCHEMA_VERSION = "go2-sequence-v1.1"
CONTACT_THRESHOLD_N = 1.0  # same convention as the feet_air_time reward
FPS = 50

# per-step features: name -> (dtype, per-env shape suffix, reduced in stats)
STEP_FEATURES: dict[str, tuple[str, tuple[int, ...]]] = {
    "obs": ("float32", (53,)),
    "action": ("float32", ()),  # 2 (layered) or 12 (single), recorded per episode
    "cmd": ("float32", (3,)),
    "joint_target": ("float32", (12,)),
    "joint_torque": ("float32", (12,)),
    "base_pos": ("float32", (3,)),
    "base_quat": ("float32", (4,)),
    "base_lin_vel_b": ("float32", (3,)),
    "base_ang_vel_b": ("float32", (3,)),
    "joint_pos": ("float32", (12,)),
    "joint_vel": ("float32", (12,)),
    "foot_contact_force": ("float32", (4,)),
    "stage": ("int8", ()),
    "reward": ("float32", ()),
    "body_pos": ("float32", (13, 3)),
    "done": ("bool_", ()),
}
DERIVED_FEATURES: dict[str, tuple[str, tuple[int, ...]]] = {
    "terminated": ("bool_", ()),
    "truncated": ("bool_", ()),
    "foot_contact": ("bool_", (4,)),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EpisodeRecorder:
    """Accumulates per-step ``[num_envs, ...]`` arrays; writes one npz per episode."""

    def __init__(self, root: Path, group: str, run_meta: dict[str, Any]) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.group = str(group)
        self.run_meta = dict(run_meta)
        self._buffers: dict[str, list[np.ndarray]] = {}
        self.entries: list[dict[str, Any]] = []

    def add(self, **arrays: np.ndarray) -> None:
        """Append one step for the given feature keys (``[num_envs, ...]`` each).

        Called up to three times per env step (pre-state, action, post-state);
        every feature key must be appended exactly once per step so the buffer
        lists stay in lockstep.
        """
        for name, value in arrays.items():
            if name not in STEP_FEATURES:
                raise ValueError(f"unknown feature {name!r}; known: {sorted(STEP_FEATURES)}")
            self._buffers.setdefault(name, []).append(np.asarray(value))

    def finalize(
        self,
        env_index: int,
        *,
        eval_seed: int,
        done_step: int,
        outcome: str,
        final_stage: int,
    ) -> Path | None:
        """Write env ``env_index``'s episode (steps ``0 .. done_step-1``) to npz."""
        if outcome not in {"success", "fall", "timeout"}:
            raise ValueError(f"unknown outcome {outcome!r}")
        length = int(done_step)
        if length <= 0:
            return None
        lengths = {name: len(chunks) for name, chunks in self._buffers.items()}
        if lengths and len(set(lengths.values())) != 1:
            raise RuntimeError(f"buffer lockstep broken: {lengths}")
        recorded = max(lengths.values(), default=0)
        if length > recorded:
            raise RuntimeError(f"done_step {length} exceeds recorded steps {recorded}")

        data: dict[str, np.ndarray] = {}
        for name, chunks in self._buffers.items():
            dtype, _ = STEP_FEATURES[name]
            data[name] = np.stack([chunk[env_index] for chunk in chunks[:length]]).astype(dtype)

        terminal = outcome != "timeout"
        data["terminated"] = np.zeros(length, dtype=bool)
        data["terminated"][-1] = terminal
        data["truncated"] = ~data["terminated"]
        if "foot_contact_force" in data:
            data["foot_contact"] = data["foot_contact_force"] > CONTACT_THRESHOLD_N

        stage = data["stage"].astype(int)
        stage_enter_steps = [
            int(np.argmax(stage == value)) if bool((stage == value).any()) else -1
            for value in sorted(set(stage.tolist()))
        ]
        success = outcome == "success"
        episode_id = f"{self.group}_eval{eval_seed:02d}"
        path = self.episodes_dir / f"{episode_id}.npz"
        entry: dict[str, Any] = {
            "episode_id": episode_id,
            "group": self.group,
            "eval_seed": int(eval_seed),
            "steps": length,
            "outcome": outcome,
            "success": success,
            "final_stage": int(final_stage),
            "stage_enter_steps": stage_enter_steps,
            "fail_step": None if success else length - 1,
            "checkpoint": self.run_meta.get("checkpoint"),
            "checkpoint_sha256": self.run_meta.get("checkpoint_sha256"),
            "interface": self.run_meta.get("interface"),
            "layered": bool(self.run_meta.get("layered", False)),
            "config": self.run_meta.get("config", {}),
            "git_commit": self.run_meta.get("git_commit"),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "file": f"episodes/{path.name}",
        }
        payload = cast("dict[str, Any]", {**data, "meta": np.asarray(json.dumps(entry, ensure_ascii=False))})
        np.savez_compressed(path, **payload)
        entry["sha256"] = sha256_file(path)
        entry["bytes"] = path.stat().st_size
        self.entries.append(entry)
        return path

    def flush(self) -> dict[str, Any]:
        """Merge this run's entries into ``manifest.json`` and rewrite checksums."""
        return write_manifest(self.root, self.entries, self.run_meta)


def write_manifest(
    root: Path,
    new_entries: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    """Upsert entries into ``<root>/manifest.json`` and recompute derived blocks."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    merged: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in previous.get("episodes", []):
            merged[entry["episode_id"]] = entry
    for entry in new_entries:
        merged[entry["episode_id"]] = entry
    episodes = sorted(merged.values(), key=lambda item: item["episode_id"])

    def condition(predicate: Callable[[dict[str, Any]], bool]) -> list[str]:
        return [item["episode_id"] for item in episodes if predicate(item)]

    splits = {
        "all": [item["episode_id"] for item in episodes],
        "success": condition(lambda item: item["success"]),
        "failure": condition(lambda item: not item["success"]),
        "layered": condition(lambda item: item["layered"]),
        "single": condition(lambda item: not item["layered"]),
        "generalization": condition(lambda item: bool(item.get("config", {}).get("patched"))),
    }
    features, stats = _features_and_stats(root, episodes)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "robot": "go2",
        "source": "isaaclab_sim",
        "task": run_meta.get("task"),
        "joint_names": run_meta.get("joint_names"),
        "body_names": run_meta.get("body_names"),
        "foot_names": run_meta.get("foot_names"),
        "obs_layout": run_meta.get("obs_layout"),
        "config_defaults": run_meta.get("config", {}),
        "total_episodes": len(episodes),
        "total_steps": sum(int(item["steps"]) for item in episodes),
        "splits": splits,
        "features": features,
        "stats": stats,
        "episodes": episodes,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_checksums(root)
    _write_card(root, manifest)
    return manifest


def _features_and_stats(
    root: Path, episodes: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    features: dict[str, Any] = {}
    accumulators: dict[str, list[float]] = {}
    for entry in episodes:
        with np.load(root / entry["file"], allow_pickle=False) as data:
            if not features:
                for name in STEP_FEATURES:
                    if name in data:
                        dtype, suffix = STEP_FEATURES[name]
                        features[name] = {"dtype": dtype, "shape": list(suffix)}
                for name, (dtype, suffix) in DERIVED_FEATURES.items():
                    if name in data:
                        features[name] = {"dtype": dtype, "shape": list(suffix), "derived": True}
            for name in features:
                values = np.asarray(data[name], dtype=np.float64)
                flat = values.reshape(-1)
                bucket = accumulators.setdefault(
                    name, [0.0, 0.0, float("inf"), float("-inf"), 0]
                )
                bucket[0] += float(flat.sum())
                bucket[1] += float((flat**2).sum())
                bucket[2] = min(bucket[2], float(flat.min()))
                bucket[3] = max(bucket[3], float(flat.max()))
                bucket[4] += int(flat.size)
    stats: dict[str, Any] = {}
    for name, (total, total_sq, minimum, maximum, count) in accumulators.items():
        if count == 0:
            continue
        mean = total / count
        variance = max(total_sq / count - mean**2, 0.0)
        stats[name] = {
            "mean": round(mean, 6),
            "std": round(variance**0.5, 6),
            "min": round(minimum, 6),
            "max": round(maximum, 6),
            "count": count,
        }
    stats["_scope"] = "all episodes, all steps"
    return features, stats


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted((root / "episodes").glob("*.npz")):
        lines.append(f"{sha256_file(path)}  episodes/{path.name}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_card(root: Path, manifest: dict[str, Any]) -> None:
    """Create or refresh the dataset card; the outcome table sits between markers."""
    root = Path(root)
    card_path = root / "README.md"
    counts: dict[tuple[str, str], int] = {}
    for entry in manifest["episodes"]:
        key = (entry["group"], entry["outcome"])
        counts[key] = counts.get(key, 0) + 1
    groups = sorted({key[0] for key in counts})
    rows = ["| group | success | fall | timeout | total |", "| --- | --- | --- | --- | --- |"]
    for group in groups:
        values = [counts.get((group, outcome), 0) for outcome in ("success", "fall", "timeout")]
        rows.append(f"| {group} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")
    table = "\n".join(rows)
    begin, end = "<!-- outcome-distribution:begin -->", "<!-- outcome-distribution:end -->"
    block = f"{begin}\n\n{table}\n\n{end}"

    if card_path.is_file():
        text = card_path.read_text(encoding="utf-8")
        if begin in text and end in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            card_path.write_text(head + block + tail, encoding="utf-8")
            return
    card_path.write_text(
        "\n".join(
            [
                "---",
                'pretty_name: "Go2 Long-Horizon Sequence Trajectories (go2-sequence-v1)"',
                "license: other",
                "task_categories:",
                "- robotics",
                "tags:",
                "- reinforcement-learning",
                "- quadruped",
                "- simulation",
                "- hierarchical-rl",
                "size_categories:",
                "- n<1K",
                "---",
                "",
                f"# go2-sequence-v1（Go2 长程运动任务轨迹数据集）",
                "",
                f"Schema `{SCHEMA_VERSION}` · {manifest['total_episodes']} episodes ·"
                f" {manifest['total_steps']} steps · {FPS} Hz · Isaac Lab 仿真自产（无真机）。",
                "字段、边界语义与采集命令见 [docs/DATASET.md](../../../docs/DATASET.md)；",
                "失败轨迹保留并标注（`outcome`/`fail_step`/`stage_enter_steps`），"
                "用 `manifest.splits` 筛选而非删除。",
                "",
                "## 结局分布",
                "",
                block,
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
