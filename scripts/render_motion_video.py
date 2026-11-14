#!/usr/bin/env python3
"""Render a robot-motion video from a recorded trajectory (pure CPU).

Why not render the simulator?  Isaac Sim crashes on this machine when
``enable_cameras`` forces the viewport's Hydra engine to initialise in
headless mode (access violation inside omni.kit.widget.viewport).  The crash
is in the rendering front end, not in our camera config, and no camera
configuration avoids it.

So the motion is recorded numerically instead: ``eval_policy.py
--record-trajectory`` saves the world position of every body at every step,
and this script draws the skeleton as a 3-D animation.  It shows exactly what
matters for the research -- where the robot went, which stage it was in, and
whether it finished.

Usage::

    python scripts/render_motion_video.py \
        --trajectory logs/.../trajectory.npz \
        --model locomotion/guardrail/go2_model.json \
        --out media/training/robot_motion.mp4

Exit codes: 0 = written, 1 = failure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

STAGE_NAMES = ("forward", "turn +90", "forward", "stop")
STAGE_COLORS = ("#4c8dff", "#f5a524", "#3ddc97", "#ff5c5c")


def load_edges(model_path: pathlib.Path) -> list[tuple[str, str]]:
    """Body-tree edges as (parent_name, child_name) from the exported model.

    Names, not indices: the trajectory stores bodies in the simulator's order,
    which need not match the model file's order.
    """
    data = json.loads(model_path.read_text(encoding="utf-8"))
    edges: list[tuple[str, str]] = []
    for joint in data["joints"]:
        parent = joint.get("body0") or ""
        child = joint.get("body1") or ""
        if parent and child and parent != child:
            edges.append((parent, child))
    return edges


def render(
    trajectory: pathlib.Path,
    model_path: pathlib.Path,
    out: pathlib.Path,
    *,
    fps: int = 25,
    stride: int = 4,
    view: str = "follow",
) -> pathlib.Path:
    data = np.load(trajectory, allow_pickle=False)
    body_pos = data["body_pos"]  # (steps, bodies, 3)
    stages = data["stage"]
    names = [str(name) for name in data["body_names"]]
    dt = float(data["dt"])
    steps = body_pos.shape[0]

    # map model-file names onto the trajectory's body order
    index = {name: i for i, name in enumerate(names)}
    edges = [
        (index[parent], index[child])
        for parent, child in load_edges(model_path)
        if parent in index and child in index
    ]

    # leg chains, highlighted separately so the gait is readable
    leg_chains: list[list[int]] = []
    for prefix in ("FL", "FR", "RL", "RR"):
        chain = [index.get(f"{prefix}_{part}") for part in ("hip", "thigh", "calf", "foot")]
        chain = [i for i in chain if i is not None]
        if len(chain) >= 3:
            leg_chains.append(chain)

    frames = list(range(0, steps, max(1, stride)))

    plt.rcParams.update({"figure.facecolor": "#0f1115", "axes.facecolor": "#0f1115"})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f1115")

    xy = body_pos[:, :, :2]
    span = float(np.max(xy.max(axis=(0, 1)) - xy.min(axis=(0, 1))) + 1.0)

    def style_axes(center: np.ndarray) -> None:
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(0.0, max(1.2, span * 0.4))
        ax.set_xlabel("x [m]", color="#8b93a7", fontsize=8)
        ax.set_ylabel("y [m]", color="#8b93a7", fontsize=8)
        ax.tick_params(colors="#8b93a7", labelsize=7)
        getattr(ax.xaxis, "pane").set_facecolor("#141821")
        getattr(ax.yaxis, "pane").set_facecolor("#141821")
        getattr(ax.zaxis, "pane").set_facecolor("#141821")
        ax.grid(color="#262b36", alpha=0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1
    )
    try:
        for frame in frames:
            ax.clear()
            position = body_pos[frame]
            stage = int(stages[frame])
            # fixed extents: a per-frame follow camera makes the motion unreadable
            style_axes(body_pos[:, :, :2].reshape(-1, 2).mean(axis=0))

            # spine/tree edges
            for parent, child in edges:
                ax.plot(
                    [position[parent, 0], position[child, 0]],
                    [position[parent, 1], position[child, 1]],
                    [position[parent, 2], position[child, 2]],
                    color="#3a4254", linewidth=1.2,
                )
            # legs, coloured by leg
            for chain, color in zip(leg_chains, ("#4c8dff", "#3ddc97", "#f5a524", "#ff5c5c")):
                pts = position[chain]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.4, marker="o", markersize=3)
            # ground projection of the trajectory so far
            trail = body_pos[: frame + 1, :, :2].mean(axis=1)
            ax.plot(trail[:, 0], trail[:, 1], np.zeros(len(trail)) + 0.005, color="#8b93a7", linewidth=1.0, alpha=0.7)

            ax.set_title(
                f"Go2 command sequence  |  t={frame * dt:5.2f}s  |  "
                f"stage {stage + 1}/4: {STAGE_NAMES[min(stage, 3)]}",
                color=STAGE_COLORS[min(stage, 3)], fontsize=13, loc="left",
            )
            ax.view_init(elev=28, azim=-125)
            fig.canvas.draw()
            frame_rgb = np.asarray(getattr(fig.canvas, "buffer_rgba")())[:, :, :3]
            writer.append_data(frame_rgb)
    finally:
        writer.close()
        plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=PROJECT_ROOT / "locomotion" / "guardrail" / "go2_model.json",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--stride", type=int, default=4, help="simulation steps per video frame (at 50 Hz)"
    )
    args = parser.parse_args(argv)

    if not args.trajectory.is_file():
        print(f"[motion] trajectory not found: {args.trajectory}")
        return 1
    out = render(args.trajectory, args.model, args.out, fps=args.fps, stride=args.stride)
    size_mb = out.stat().st_size / 1e6
    print(f"[motion] wrote {out} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
