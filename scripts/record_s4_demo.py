#!/usr/bin/env python3
"""Record S4 demo videos: the layered vs single-layer policies running the
L3 sequence in MuJoCo (the Sim2Sim side of the A/B comparison).

Outputs (media/training/):
- s4_layered_sequence.mp4 / .gif   layered policy, expect 4-stage success
- s4_single_fail.mp4 / .gif        single-layer policy, expect a fall at the turn

The loop is byte-for-byte the S4 evaluation loop (Go2Sim2SimEnv +
Sim2SimPolicy.act) with a per-step offscreen render (track camera) added.
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


def record(policy, seed: int, out_mp4: pathlib.Path, out_gif: pathlib.Path,
           max_steps: int = 1200, fps: int = 30) -> dict[str, object]:
    import mujoco
    import imageio

    from locomotion.sim2sim.go2_env import Go2Sim2SimEnv

    env = Go2Sim2SimEnv("sequence", seed=seed)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    obs = env.reset()

    frames: list[np.ndarray] = []
    last_stage = 0
    steps = 0
    while not env.over and steps < max_steps:
        env.step(policy.act(obs))
        obs = env.observation()
        steps += 1
        if env.stage != last_stage:
            print(f"  stage {last_stage} -> {env.stage} at step {steps}")
            last_stage = env.stage
        renderer.update_scene(env.data, camera="track")
        frames.append(renderer.render().copy())
        if env.over:
            break

    writer: Any = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    with writer:
        for frame in frames:
            writer.append_data(frame)
    import PIL.Image

    pil_frames = [PIL.Image.fromarray(f) for f in frames]
    pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:],
                       duration=1000 // fps, loop=0, optimize=True)

    renderer.close()
    return {
        "steps": steps,
        "fell": bool(env.fell),
        "completed": bool(env.completed),
        "stage_reached": int(env.stage),
        "frames": len(frames),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "usage").splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "media" / "training")
    args = parser.parse_args(argv)

    from locomotion.sim2sim.go2_env import Sim2SimPolicy

    actors = PROJECT_ROOT / "logs" / "s4" / "actors"

    layered = Sim2SimPolicy(
        mode="layered",
        s0_path=actors / "s0_flat.onnx",
        s1_path=actors / "s1_layered.onnx",
    )
    print("[1/2] layered policy — L3 sequence in MuJoCo")
    layered_stats = record(layered, args.seed,
                           args.out / "s4_layered_sequence.mp4",
                           args.out / "s4_layered_sequence.gif")
    print("   ", layered_stats)

    single = Sim2SimPolicy(
        mode="single",
        single_path=actors / "single_layer.onnx",
    )
    print("[2/2] single-layer policy — same task, same seed")
    single_stats = record(single, args.seed,
                          args.out / "s4_single_fail.mp4",
                          args.out / "s4_single_fail.gif")
    print("   ", single_stats)

    summary = {"layered": layered_stats, "single": single_stats, "seed": args.seed}
    (args.out / "s4_demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("summary:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
