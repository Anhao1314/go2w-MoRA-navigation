"""rclpy-free core of the Go2 ROS2 online bridge (sim-only closed loop).

This module holds everything that must be testable WITHOUT ROS 2 (no rclpy,
no message packages — matching the repo's CI, which runs on plain Python):

- the topic/protocol constants and ``protocol.json`` validation;
- payload dataclasses and canonical-JSON builders for every published topic;
- the command gate: clamp to the frozen ±1 ranges, deadman zeroing after
  ``DEADMAN_S`` without fresh commands.

The rclpy layer (``sim_node.py`` / ``probe.py``) imports ROS packages lazily
inside functions and converts these payloads into real messages; nothing here
may import rclpy, mujoco, or onnxruntime, so the whole bridge stays testable
on a Windows host without ROS.

Authority statement (see ``protocol.json``): this is a SIMULATION-ONLY closed
loop. Frames are sim ground truth (``mujoco_world`` → ``base``); no map/odom
frames, no odometry/EKF claims, no hardware, and no raw motor-command
publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "go2_ros2_bridge_v0.1"
FRAME_WORLD = "mujoco_world"
FRAME_BASE = "base"
CONTROL_HZ = 50
STEP_DT_S = 0.02
COMMAND_VX_RANGE = (-1.0, 1.0)
COMMAND_WZ_RANGE = (-1.0, 1.0)
DEADMAN_S = 0.5  # stale /cmd_vel older than this zeroes the command

TOPICS: dict[str, str] = {
    "clock": "/clock",
    "joint_states": "/go2/joint_states",
    "base_pose": "/go2/base_pose",
    "tf": "/tf",
    "session_info": "/go2/session_info",
    "cmd_vel": "/go2/cmd_vel",  # subscription
}
PUBLISHED_TOPICS: tuple[str, ...] = (
    "clock",
    "joint_states",
    "base_pose",
    "tf",
    "session_info",
)

#: Canonical joint order as published (the menagerie MJCF's own per-leg order —
#: the shared model artifact; the Isaac per-joint-type remap lives in the
#: policy layer only and must not leak into the ROS interface).
JOINT_ORDER: tuple[str, ...] = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


class BridgeError(RuntimeError):
    """Fail-closed error for protocol/env/dependency violations."""


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON for String payloads (g1-repo convention)."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(repo_root: pathlib.Path) -> dict[str, Any]:
    """Load and validate ``ros2/go2_bridge/protocol.json`` (fail-closed)."""
    path = pathlib.Path(repo_root) / "ros2" / "go2_bridge" / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA:
        raise BridgeError(f"protocol schema mismatch: {protocol.get('schema')!r}")
    if protocol.get("topics") != TOPICS:
        raise BridgeError("protocol topic table does not match the bridge TOPICS")
    return protocol


def clamp_command(vx: float, wz: float) -> tuple[float, float]:
    """Clamp a raw command into the frozen training ranges."""
    return (
        max(COMMAND_VX_RANGE[0], min(COMMAND_VX_RANGE[1], float(vx))),
        max(COMMAND_WZ_RANGE[0], min(COMMAND_WZ_RANGE[1], float(wz))),
    )


class CommandGate:
    """Latest /cmd_vel with clamp and deadman zeroing (pure, clock-injected)."""

    def __init__(self, *, deadman_s: float = DEADMAN_S) -> None:
        self._deadman_s = deadman_s
        self._last: tuple[float, float] | None = None
        self._last_ns: int | None = None
        self.accepted: int = 0

    def update(self, vx: float, wz: float, now_ns: int) -> tuple[float, float]:
        """Accept a fresh command; returns the clamped pair."""
        self._last = clamp_command(vx, wz)
        self._last_ns = int(now_ns)
        self.accepted += 1
        return self._last

    def current(self, now_ns: int) -> tuple[float, float]:
        """Command currently in effect (zeroed when the deadman expires)."""
        if (
            self._last is None
            or self._last_ns is None
            or (now_ns - self._last_ns) > int(self._deadman_s * 1_000_000_000)
        ):
            return (0.0, 0.0)
        return self._last


def quat_wxyz_to_xyzw(quat_wxyz) -> tuple[float, float, float, float]:
    """MuJoCo (w, x, y, z) → ROS geometry_msgs (x, y, z, w)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("degenerate quaternion")
    return (x / norm, y / norm, z / norm, w / norm)


@dataclass(frozen=True)
class LiveSample:
    """One control step projected for ROS publication (all fields plain)."""

    time_ns: int
    episode_id: int
    step: int
    joint_names: tuple[str, ...]
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    base_position_m: tuple[float, float, float]
    base_quaternion_xyzw: tuple[float, float, float, float]
    command_applied: tuple[float, float]

    def __post_init__(self) -> None:
        if not (len(self.joint_names) == len(self.joint_position_rad) == len(self.joint_velocity_rad_s)):
            raise ValueError("joint name/position/velocity length mismatch")

    def joint_state_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "JOINT_STATE",
            "episode_id": self.episode_id,
            "step": self.step,
            "stamp_ns": self.time_ns,
            "frame_id": "",  # no one physical frame is shared by articulated joints
            "joint_names": list(self.joint_names),
            "position": list(self.joint_position_rad),
            "velocity": list(self.joint_velocity_rad_s),
            "effort": [],  # not measured in this sim setup — never fabricated
            "command_applied": list(self.command_applied),
        }

    def pose_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "BASE_POSE",
            "episode_id": self.episode_id,
            "step": self.step,
            "stamp_ns": self.time_ns,
            "frame_id": FRAME_WORLD,
            "child_frame_id": FRAME_BASE,
            "position": list(self.base_position_m),
            "orientation_xyzw": list(self.base_quaternion_xyzw),
        }

    def tf_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "TF",
            "stamp_ns": self.time_ns,
            "frame_id": FRAME_WORLD,
            "child_frame_id": FRAME_BASE,
            "translation": list(self.base_position_m),
            "rotation_xyzw": list(self.base_quaternion_xyzw),
        }


@dataclass
class SessionState:
    """Accumulating session bookkeeping published on ``session_info``."""

    episode_id: int = 0
    reset_count: int = 0
    fall_count: int = 0
    steps_total: int = 0
    commands_accepted: int = 0
    command_deadman_zeroed: bool = False
    command_applied: tuple[float, float] = (0.0, 0.0)
    extra: dict[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "SESSION_INFO",
            "episode_id": self.episode_id,
            "reset_count": self.reset_count,
            "fall_count": self.fall_count,
            "steps_total": self.steps_total,
            "commands_accepted": self.commands_accepted,
            "command_deadman_zeroed": self.command_deadman_zeroed,
            "command_applied": list(self.command_applied),
            "authority": "SIM_ONLY_CLOSED_LOOP_NO_HARDWARE",
            **self.extra,
        }
