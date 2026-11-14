#!/usr/bin/env python3
"""Evaluate a fine-tuned SmolVLA policy in the SO-101 tabletop MuJoCo env.

Uses the OFFICIAL lerobot inference stack: PreTrainedConfig.from_pretrained +
make_pre_post_processors (device move, observation renaming, task tokenizer,
dataset-stats normalization) + PreTrainedPolicy.select_action.  The env
provides raw observations; every transformation is the official one — no
hand-written preprocessing here.

Usage (WSL vla-env):
    python vla/evaluate.py \
        --checkpoint logs/vla/smolvla_run1/checkpoints/020000/pretrained_model \
        --episodes 10 [--video]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "usage").splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True,
                        help="pretrained_model dir from lerobot-train")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000,
                        help="first eval seed (env randomization)")
    parser.add_argument("--video", action="store_true", help="record the first episode as MP4")
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "vla" / "eval")
    parser.add_argument("--max-steps", type=int, default=3600, help="per-episode cap at 30 fps")
    args = parser.parse_args(argv)

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from lerobot.configs.policies import PreTrainedConfig  # pyright: ignore[reportMissingImports]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]
    from lerobot.policies import PreTrainedPolicy, get_policy_class, make_pre_post_processors  # pyright: ignore[reportMissingImports]

    from vla.env import TASK_INSTRUCTION, make_so101_env

    ckpt = pathlib.Path(args.checkpoint)
    policy_config = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_config.pretrained_path = str(ckpt)
    device = "cuda"

    # normalization stats come from the TEACHER dataset the policy was fine-tuned on
    teacher_dataset = LeRobotDataset("lerobot/svla_so101_pickplace")
    stats = teacher_dataset.meta.stats

    # the rollout rename maps the ENV's camera keys to the policy's expected names
    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(ckpt),
        dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(str(ckpt), config=policy_config)
    policy.to(device)
    policy.eval()

    env = make_so101_env()
    results: list[dict[str, object]] = []
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset(seed=[args.seed])
    task = env.get_attr("task_description")[0] or TASK_INSTRUCTION

    for episode in range(args.episodes):
        seed = args.seed + episode
        obs, info = env.reset(seed=[seed])
        policy.reset()

        episode_frames: list[np.ndarray] = []
        steps = 0
        done = False
        success = False
        cubes = 0
        while not done and steps < args.max_steps:
            batch: dict[str, object] = {
                # CHW float [0,1] — the LeRobot image convention the pipeline expects
                "observation.images.front": torch.as_tensor(
                    np.asarray(obs["observation.images.front"])
                ).permute(0, 3, 1, 2).float() / 255.0,
                "observation.images.wrist": torch.as_tensor(
                    np.asarray(obs["observation.images.wrist"])
                ).permute(0, 3, 1, 2).float() / 255.0,
                "observation.state": torch.as_tensor(
                    np.asarray(obs["observation.state"], dtype=np.float32)
                ),
                "task": task,
            }
            processed = preprocessor(batch)

            with torch.inference_mode():
                action = policy.select_action(processed)
            if action.ndim == 1:  # select_action may return unbatched (6,)
                action = action.unsqueeze(0)

            action = postprocessor(action)
            action_np = action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)
            obs, reward, terminated, truncated, info = env.step(action_np)  # batched (1, 6)
            steps += 1
            done = bool(terminated or truncated)
            success = bool(info.get("is_success", False))
            cubes = int(info.get("cubes_in_box", 0))
            if episode == 0 and args.video:
                episode_frames.append(np.asarray(obs["observation.images.front"]).copy())

        results.append({
            "episode": episode,
            "seed": seed,
            "steps": steps,
            "success": success,
            "cubes_in_box": cubes,
        })
        print(f"[eval] episode {episode} seed {seed}: "
              f"{'SUCCESS' if success else 'incomplete'} ({cubes}/4 cubes, {steps} steps)",
              flush=True)

        if episode == 0 and args.video and episode_frames:
            import imageio  # pyright: ignore[reportMissingImports]

            video_path = out / "eval_vla_episode0.mp4"
            writer: Any = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
            with writer:
                for frame in episode_frames:
                    writer.append_data(frame)
            print(f"      video: {video_path}", flush=True)

    successes = sum(1 for r in results if r["success"])
    summary = {
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(1, args.episodes),
        "checkpoint": str(ckpt),
        "task": task,
        "method": "mujoco_envhub",
        "results": results,
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
