#!/usr/bin/env python3
"""Generate the S4 Sim2Sim comparison figure from the frozen-protocol results.

Data source: logs/vla/../s4 evaluation (evaluate_sequence_group +
evaluate_velocity results on E={0..19}) — the numbers are transcribed from
the committed evaluation JSONs and cited in the README/S4 report.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "media" / "training" / "s4_sim2sim_comparison.png"

# Frozen-protocol results (Isaac: go2_loco_sequence evaluation.json;
# MuJoCo: logs/s4/sequence_* evaluation.json — see S4 report for paths)
SUCESS_RATE = {
    "layered": {"Isaac": 1.00, "MuJoCo": 1.00},
    "single": {"Isaac": 0.00, "MuJoCo": 0.00},
}
VX_RMSE = {"Isaac": 0.0569, "MuJoCo": 0.0773}
VX_RMSE_STD = {"MuJoCo": 0.0285}
GATE_SUCCESS = 0.90
GATE_VX = 0.15

C_ISAAC = "#3b6fd4"
C_MUJOCO = "#d4772b"
C_GATE = "#c23b3b"


def main() -> int:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fig.patch.set_facecolor("white")

    # Panel A: sequence success rate
    labels = ["分层 B\n(Isaac)", "分层 B\n(MuJoCo)", "单层 A\n(Isaac)", "单层 A\n(MuJoCo)"]
    values = [1.00, 1.00, 0.00, 0.00]
    colors = [C_ISAAC, C_MUJOCO, "#b9c0cc", "#b9c0cc"]
    bars = ax1.bar(labels, values, color=colors, width=0.62, edgecolor="#39404d")
    ax1.axhline(GATE_SUCCESS, color=C_GATE, linestyle="--", linewidth=1.2)
    ax1.text(3.45, GATE_SUCCESS + 0.02, "冻结阈值 ≥ 0.90", color=C_GATE, fontsize=9,
             ha="right")
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("四阶段序列成功率", fontsize=10)
    ax1.set_title("S4 序列成功率：分层 vs 单层（20 冻结种子）", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel B: vx tracking RMSE
    bars2 = ax2.bar(["Isaac\n(训练引擎)", "MuJoCo\n(迁移引擎)"], [VX_RMSE["Isaac"], VX_RMSE["MuJoCo"]],
                    color=[C_ISAAC, C_MUJOCO], width=0.5, edgecolor="#39404d")
    ax2.errorbar(1, VX_RMSE["MuJoCo"], yerr=VX_RMSE_STD["MuJoCo"], fmt="none",
                 ecolor="#39404d", capsize=5, linewidth=1.2)
    ax2.axhline(GATE_VX, color=C_GATE, linestyle="--", linewidth=1.2)
    ax2.text(1.42, GATE_VX + 0.004, "冻结阈值 ≤ 0.15", color=C_GATE, fontsize=9, ha="right")
    for bar, value in zip(bars2, [VX_RMSE["Isaac"], VX_RMSE["MuJoCo"]]):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 0.11)
    ax2.set_ylabel("vx 跟踪 RMSE [m/s]", fontsize=10)
    ax2.set_title("S4 速度跟踪精度（M1 低层策略）", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("S4 Sim2Sim：零策略改动跨引擎迁移（20 冻结种子，协议 v1.0）", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("chart:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
