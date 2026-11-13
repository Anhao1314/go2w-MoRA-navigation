#!/usr/bin/env python3
"""V1 smoke test: SO-101 EnvHub env + scripted expert picking all four cubes.

Runs inside the WSL2 ``vla-env`` (lerobot + mujoco + EGL).  Verifies:

1. the EnvHub env downloads and steps (batched LocalVectorEnv);
2. both cameras render (MUJOCO_GL=egl) at 480x640;
3. ground-truth geometry resolves (four cubes + box + EE frame);
4. the scripted expert places all four cubes in the box (official counter);
5. a demo MP4 of the episode is written to ``logs/vla/demo_expert.mp4``.

Usage (WSL): MUJOCO_GL=egl python scripts/vla_smoke.py [--video] [--max-steps N]
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", action="store_true", help="record an MP4 of the episode")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--out", type=pathlib.Path, default=PROJECT_ROOT / "logs" / "vla")
    args = parser.parse_args(argv)

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    print("[1/5] creating the SO-101 tabletop env (EnvHub download on first run)...")
    from vla.env import SO101TabletopSession, make_so101_raw_env

    env, module = make_so101_raw_env()
    session = SO101TabletopSession(env)
    print("      env:", type(env).__name__, "| EE frame:", session._ee_site)

    obs = session.reset(seed=0)
    print("[2/5] camera rendering check...")
    from vla.env import camera_frames_from

    for name, frame in camera_frames_from(obs).items():
        print(f"      {name}: shape={frame.shape} dtype={frame.dtype}")

    print("[3/5] ground-truth geometry check...")
    print("      tip_offset_z:", round(session._tip_offset_z, 4), "m")
    for cube in session._body_ids:
        if cube.startswith("red") or "box" in cube:
            print(f"      {cube}:", tuple(round(v, 3) for v in session._body_pos(cube)))

    print("[4/5] scripted expert: picking all four cubes...")
    from vla.expert import PickPlacePhaseMachine
    from vla.env import CUBE_BODIES

    machines = {cube: PickPlacePhaseMachine() for cube in CUBE_BODIES}
    queue = list(CUBE_BODIES)
    current = queue[0]
    video_frames = []
    phase_log = []
    steps = 0
    obs = session.reset(seed=0)
    while steps < args.max_steps and queue:
        machine = machines[current]
        command = machine.update(session.expert_observation(obs, current))
        if command.done:
            placed = session.cubes_in_box()
            print(f"      {current}: DONE (cubes in box: {placed}/{session.total_cubes()}) at step {steps}")
            queue.pop(0)
            current = queue[0] if queue else None
            if current:
                machines[current].reset()
            continue
        action = session.expert_action(command)
        obs, reward, terminated, truncated, info = session.step(action)
        steps += 1
        if args.video:
            video_frames.append(session.front_frame_rgb(obs).copy())
        if terminated or truncated:
            break

    in_box = session.cubes_in_box()
    success = in_box == session.total_cubes()
    print(f"[4/5] result: cubes_in_box={in_box}/{session.total_cubes()} steps={steps}")

    print("[5/5] writing demo MP4...")
    out = args.out / "demo_expert.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    with imageio.get_writer(out, fps=30, codec="libx264", quality=8) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    print("      video:", out)

    report = {"cubes_in_box": in_box, "total": session.total_cubes(),
              "success": success, "steps": steps, "env": "lerobot/so101-tabletop-mujoco"}
    (args.out / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("smoke:", json.dumps(report))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
