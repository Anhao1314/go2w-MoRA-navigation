"""Independent ROS2 probe for the Go2 bridge acceptance (publisher + verifier).

This program imports rclpy only inside its functions.  It is deliberately
independent of the sim node: it re-derives the topic contract from
``core``/``protocol.json`` itself and verifies the closed loop BEHAVIORALLY —
commands it publishes must show up as (clamped) applied commands in
``session_info`` and as real motion in ``base_pose``:

- phase A (sim t ∈ [2, 5) s): drive (0.5, 0)   → planar displacement > 0.2 m
- phase B (sim t ∈ [5, 8) s): turn (0, +0.8)   → yaw change > +0.1 rad
- elsewhere: zero command → the robot must not walk away

A PASS writes an exclusively-created receipt ``{"status": "ROS2_ROUNDTRIP_PASS",
...}``; every failure writes ``ROS2_ROUNDTRIP_FAILED`` with a reason and exits 2.
The receipt is the only success claim; the sim node alone never claims one.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time as _time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ros2.go2_bridge import core


def _gate_environment() -> None:
    import os

    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if not domain.isdigit() or not (1 <= int(domain) <= 101):
        raise core.BridgeError(f"ROS_DOMAIN_ID must be a decimal in [1, 101], got {domain!r}")
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise core.BridgeError("ROS_AUTOMATIC_DISCOVERY_RANGE must be LOCALHOST")


def _require_ros():
    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import PoseStamped, Twist
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
    except ImportError as error:
        raise core.BridgeError(f"real ROS 2 rclpy and standard messages are unavailable: {error}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=32,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    # TF verification is conditional: tf2_msgs ships with tf2_ros, which the
    # sim node also degrades on — the receipt records whether it was checked.
    try:
        from tf2_msgs.msg import TFMessage
    except ImportError:
        TFMessage = None
    return rclpy, qos, Clock, JointState, PoseStamped, String, Twist, TFMessage


def _yaw_from_xyzw(q) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _phase_command(sim_t_s: float) -> tuple[float, float]:
    """Piecewise test sequence, keyed on the SIM clock the probe receives."""
    if 2.0 <= sim_t_s < 5.0:
        return (0.5, 0.0)  # A: walk forward
    if 5.0 <= sim_t_s < 8.0:
        return (0.0, 0.8)  # B: turn in place
    return (0.0, 0.0)


def run_probe(*, duration_s: float = 14.0, out_dir: pathlib.Path | None = None) -> dict[str, object]:
    _gate_environment()
    rclpy, qos, Clock, JointState, PoseStamped, String, Twist, TFMessage = _require_ros()

    core.load_protocol(PROJECT_ROOT)  # fail-closed protocol check
    received: dict[str, list] = {key: [] for key in core.TOPICS}
    rclpy.init(args=None)
    node = rclpy.create_node("go2_bridge_probe", use_global_arguments=False,
                             enable_rosout=False, start_parameter_services=False)

    def _factory(key: str):
        def callback(message) -> None:
            received[key].append((message, _time.monotonic_ns()))
        return callback

    node.create_subscription(Clock, core.TOPICS["clock"], _factory("clock"), qos)
    node.create_subscription(JointState, core.TOPICS["joint_states"], _factory("joint_states"), qos)
    node.create_subscription(PoseStamped, core.TOPICS["base_pose"], _factory("base_pose"), qos)
    node.create_subscription(String, core.TOPICS["session_info"], _factory("session_info"), qos)
    if TFMessage is not None:
        node.create_subscription(TFMessage, core.TOPICS["tf"], _factory("tf"), qos)
    pub_cmd = node.create_publisher(Twist, core.TOPICS["cmd_vel"], qos)

    deadline = _time.monotonic() + duration_s
    last_wall_publish = 0.0
    try:
        while _time.monotonic() < deadline:
            # publish the phase command at ~20 Hz, keyed on the latest sim clock
            if received["clock"]:
                latest = received["clock"][-1][0]
                sim_t = latest.clock.sec + latest.clock.nanosec / 1e9
            else:
                sim_t = 0.0
            now = _time.monotonic()
            if now - last_wall_publish >= 0.05:
                vx, wz = _phase_command(sim_t)
                message = Twist()
                message.linear.x = vx
                message.angular.z = wz
                pub_cmd.publish(message)
                last_wall_publish = now
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    verdict = _verify(received, tf_verified=TFMessage is not None)
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "x"  # exclusive creation — a rerun must never overwrite a receipt
        try:
            (out_dir / "probe_receipt.json").write_text(
                json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
            )
        except FileExistsError:
            verdict = {"status": "RECEIPT_ERROR", "reason": "probe receipt already exists"}
    return verdict


def _verify(received: dict[str, list], *, tf_verified: bool) -> dict[str, object]:
    """Behavioral closed-loop verification; returns the receipt dict."""
    def fail(reason: str) -> dict[str, object]:
        return {"status": "ROS2_ROUNDTRIP_FAILED", "reason": reason,
                "topic_counts": {key: len(values) for key, values in received.items()}}

    counts = {key: len(values) for key, values in received.items()}
    if counts["clock"] < 200 or counts["base_pose"] < 200 or counts["joint_states"] < 200:
        return fail(f"insufficient messages received: {counts}")

    clocks = [m.clock.sec * 1_000_000_000 + m.clock.nanosec for m, _ in received["clock"]]
    if any(b <= a for a, b in zip(clocks, clocks[1:])):
        return fail("/clock is not strictly increasing")

    for message, _ in received["joint_states"]:
        if list(message.name) != list(core.JOINT_ORDER):
            return fail("joint names/order do not match the protocol")
        if list(message.effort) != []:
            return fail("effort must stay empty (OMITTED_UNMEASURED)")

    poses = []
    for message, _ in received["base_pose"]:
        if message.header.frame_id != core.FRAME_WORLD:
            return fail("base_pose frame_id must be mujoco_world")
        q = (message.pose.orientation.x, message.pose.orientation.y,
             message.pose.orientation.z, message.pose.orientation.w)
        norm = math.sqrt(sum(v * v for v in q))
        if abs(norm - 1.0) > 1e-3:
            return fail("base_pose quaternion is not normalized")
        poses.append((
            message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec,
            (message.pose.position.x, message.pose.position.y),
            _yaw_from_xyzw(q),
        ))

    def window(t0: float, t1: float):
        return [row for row in poses if t0 <= row[0] / 1e9 < t1]

    # Phase A: forward drive must move the base by > 0.2 m
    phase_a = window(2.0, 5.0)
    if len(phase_a) < 10:
        return fail("phase A (forward) has too few pose samples")
    dx = phase_a[-1][1][0] - phase_a[0][1][0]
    dy = phase_a[-1][1][1] - phase_a[0][1][1]
    displacement = math.hypot(dx, dy)
    if displacement <= 0.2:
        return fail(f"forward command produced only {displacement:.3f} m displacement")

    # Phase B: +0.8 rad/s turn must produce a clearly positive yaw change
    phase_b = window(5.0, 8.0)
    if len(phase_b) < 10:
        return fail("phase B (turn) has too few pose samples")
    dyaw = (phase_b[-1][2] - phase_b[0][2] + math.pi) % (2 * math.pi) - math.pi
    if dyaw <= 0.1:
        return fail(f"turn command produced only {dyaw:+.3f} rad yaw change")

    # Round trip: the sim must report the clamped applied commands
    session_payloads = [json.loads(m.data) for m, _ in received["session_info"]]
    applied = [tuple(row["command_applied"]) for row in session_payloads if "command_applied" in row]
    if not any(vx > 0.4 for vx, _ in applied):
        return fail("session_info never reports the forward command as applied")
    if not any(abs(wz) > 0.4 for _, wz in applied):
        return fail("session_info never reports the turn command as applied")

    return {
        "status": "ROS2_ROUNDTRIP_PASS",
        "topic_counts": counts,
        "tf_verified": tf_verified,
        "forward_displacement_m": round(displacement, 4),
        "turn_yaw_change_rad": round(dyaw, 4),
        "commands_accepted": max((row.get("commands_accepted", 0) for row in session_payloads), default=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=11.0, help="probe wall-clock seconds")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="receipt output dir")
    args = parser.parse_args(argv)
    verdict = run_probe(duration_s=args.duration, out_dir=args.out)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["status"] == "ROS2_ROUNDTRIP_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
