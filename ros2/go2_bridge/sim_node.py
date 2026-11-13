"""rclpy side of the Go2 online bridge: a 50 Hz closed-loop sim node.

Importing this module does NOT import rclpy, mujoco, or onnxruntime — ROS and
sim dependencies are function-local (the g1-repo bridge pattern), so the core
and its tests run on hosts without ROS 2.

``run_node`` is the only entry that touches ROS:

- fail-closed env gating (``ROS_DOMAIN_ID`` in [1, 101], discovery LOCALHOST);
- publishers: /clock, /go2/joint_states, /go2/base_pose, /go2/session_info and
  a sim-ground-truth TF (mujoco_world → base; gracefully skipped when
  ``tf2_ros`` is unavailable, recorded in the status payload);
- subscription: /go2/cmd_vel → :class:`core.CommandGate` (clamp + deadman);
- loop: step ``Go2Sim2SimEnv`` (velocity task) + ``Sim2SimPolicy`` at 50 Hz,
  auto-reset on fall/timeout, session bookkeeping, wall-clock pacing.

The node returns ``PUBLISHED_NOT_SUBSCRIBER_VERIFIED`` — as in the g1 bridge,
the publisher alone never claims delivery; the independent probe does that.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time as _time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ros2.go2_bridge import core

SCHEMA = core.SCHEMA
NODE_NAME = "go2_sim_bridge"


def preflight(repo_root: pathlib.Path) -> dict[str, object]:
    """No-ROS validation: protocol, actor, env imports discoverable."""
    protocol = core.load_protocol(repo_root)
    actor = repo_root / protocol["policy"]["actor"]
    if not actor.exists():
        raise core.BridgeError(f"policy actor missing: {actor}")
    import importlib.util

    report = {
        "schema": core.SCHEMA,
        "protocol_version_ok": True,
        "actor_path": str(actor),
        "actor_sha256": core.sha256_file(actor),
        "mujoco_discoverable": importlib.util.find_spec("mujoco") is not None,
        "onnxruntime_discoverable": importlib.util.find_spec("onnxruntime") is not None,
        "rclpy_discoverable": importlib.util.find_spec("rclpy") is not None,
    }
    return report


def _gate_environment() -> None:
    import os

    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if not domain.isdigit() or not (1 <= int(domain) <= 101):
        raise core.BridgeError(f"ROS_DOMAIN_ID must be a decimal in [1, 101], got {domain!r}")
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise core.BridgeError("ROS_AUTOMATIC_DISCOVERY_RANGE must be LOCALHOST")


def _require_ros():
    """Lazy ROS import; raises a structured BridgeError when unavailable."""
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
    return rclpy, qos, Clock, JointState, PoseStamped, String, Twist


def run_node(
    *,
    actor_path: pathlib.Path,
    max_steps: int = 3000,
    out_dir: pathlib.Path | None = None,
) -> dict[str, object]:
    """Run the live bridge node; blocks until max_steps or KeyboardInterrupt."""
    _gate_environment()
    rclpy, qos, Clock, JointState, PoseStamped, String, Twist = _require_ros()

    from locomotion.sim2sim.go2_env import Go2Sim2SimEnv, Sim2SimPolicy

    protocol = core.load_protocol(PROJECT_ROOT)
    policy = Sim2SimPolicy(mode="s0", s0_path=actor_path)
    gate = core.CommandGate()

    episode_id = 0
    session = core.SessionState()
    session.extra = {
        "actor_sha256": core.sha256_file(actor_path),
        "git_commit": _git_commit(),
        "control_hz": core.CONTROL_HZ,
    }

    rclpy.init(args=None)
    node = rclpy.create_node(
        NODE_NAME,
        use_global_arguments=False,
        enable_rosout=False,
        start_parameter_services=False,
        enable_logger_service=False,
    )
    try:
        pub_clock = node.create_publisher(Clock, core.TOPICS["clock"], qos)
        pub_joints = node.create_publisher(JointState, core.TOPICS["joint_states"], qos)
        pub_pose = node.create_publisher(PoseStamped, core.TOPICS["base_pose"], qos)
        pub_session = node.create_publisher(String, core.TOPICS["session_info"], qos)
    except Exception:
        node.destroy_node()
        rclpy.shutdown()
        raise

    tf_available = True
    tf_broadcaster = None
    try:
        from tf2_ros import TransformBroadcaster  # optional; degrade gracefully

        tf_broadcaster = TransformBroadcaster(node, qos=qos)
    except ImportError:
        tf_available = False
        session.extra["tf"] = "SKIPPED_TF2_ROS_UNAVAILABLE"

    env = Go2Sim2SimEnv("velocity", seed=0, episode_steps=10_000_000)

    def on_cmd_vel(message: Twist) -> None:
        now_ns = _time.monotonic_ns()
        gate.update(message.linear.x, message.angular.z, now_ns)

    node.create_subscription(Twist, core.TOPICS["cmd_vel"], on_cmd_vel, qos)

    def _stamp(message, time_ns: int) -> None:
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(int(time_ns), 1_000_000_000)

    def _publish_sample(sample: core.LiveSample, clock_msg: Clock, joint_msg: JointState,
                        pose_msg: PoseStamped, session_msg: String, sim_time_ns: int) -> None:
        clock_msg.clock.sec, clock_msg.clock.nanosec = divmod(sim_time_ns, 1_000_000_000)
        pub_clock.publish(clock_msg)

        joint = sample.joint_state_payload()
        joint_msg.header.stamp.sec, joint_msg.header.stamp.nanosec = divmod(sample.time_ns, 1_000_000_000)
        joint_msg.header.frame_id = joint["frame_id"]
        joint_msg.name = list(joint["joint_names"])
        joint_msg.position = list(joint["position"])
        joint_msg.velocity = list(joint["velocity"])
        joint_msg.effort = []
        pub_joints.publish(joint_msg)

        pose = sample.pose_payload()
        pose_msg.header.stamp.sec, pose_msg.header.stamp.nanosec = divmod(sample.time_ns, 1_000_000_000)
        pose_msg.header.frame_id = pose["frame_id"]
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pose["position"]
        (pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
         pose_msg.pose.orientation.z, pose_msg.pose.orientation.w) = pose["orientation_xyzw"]
        pub_pose.publish(pose_msg)

        if tf_broadcaster is not None:
            from tf2_ros import TransformStamped  # function-local, like the rest

            tf = TransformStamped()
            tf.header.stamp.sec, tf.header.stamp.nanosec = divmod(sample.time_ns, 1_000_000_000)
            tf.header.frame_id = core.FRAME_WORLD
            tf.child_frame_id = core.FRAME_BASE
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = sample.base_position_m
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = sample.base_quaternion_xyzw
            tf_broadcaster.sendTransform(tf)

        if sample.step % 50 == 0:  # session info at ~1 Hz (and on resets)
            session.steps_total = env.step_count
            session.commands_accepted = gate.accepted
            session.command_applied = sample.command_applied
            session.command_deadman_zeroed = gate.current(sample.time_ns) == (0.0, 0.0) and gate.accepted > 0
            session_msg.data = core.canonical_json(session.payload())
            pub_session.publish(session_msg)

    started_ns = _time.monotonic_ns()
    obs = env.reset()
    session.episode_id = 0
    transcript: list[dict[str, object]] = []
    step = 0
    clock_msg, joint_msg, pose_msg, session_msg = Clock(), JointState(), PoseStamped(), String()
    try:
        while step < max_steps:
            vx, wz = gate.current(_time.monotonic_ns())
            env.set_external_command(vx, wz)
            env.step(policy.act(obs))
            obs = env.observation()
            step += 1

            if env.over:
                if env.fell:
                    session.fall_count += 1
                session.reset_count += 1
                episode_id += 1
                session.episode_id = episode_id
                obs = env.reset()

            sim_time_ns = int(env.step_count * core.STEP_DT_S * 1_000_000_000)
            sample = core.LiveSample(
                time_ns=sim_time_ns,
                episode_id=session.episode_id,
                step=env.step_count,
                joint_names=core.JOINT_ORDER,
                joint_position_rad=tuple(float(v) for v in env.data.qpos[env.isaac_qpos_adr]),
                joint_velocity_rad_s=tuple(float(v) for v in env.data.qvel[env.isaac_dof_adr]),
                base_position_m=tuple(float(v) for v in env.base_position),
                base_quaternion_xyzw=core.quat_wxyz_to_xyzw(env.data.qpos[3:7]),
                command_applied=(float(env.command[0]), float(env.command[2])),
            )
            _publish_sample(sample, clock_msg, joint_msg, pose_msg, session_msg, sim_time_ns)

            rclpy.spin_once(node, timeout_sec=0.0)
            # wall-clock pacing to the 50 Hz control rate
            target = started_ns + int(step * core.STEP_DT_S * 1_000_000_000)
            remaining_s = (target - _time.monotonic_ns()) / 1e9
            if remaining_s > 0:
                _time.sleep(min(remaining_s, 0.05))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "node_status.json").write_text(
            json.dumps({"schema": SCHEMA, "ros2": {"status": "PUBLISHED_NOT_SUBSCRIBER_VERIFIED"},
                        "steps": step, "session": session.payload()}, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "schema": SCHEMA,
        "ros2": {"status": "PUBLISHED_NOT_SUBSCRIBER_VERIFIED", "tf_available": tf_available},
        "steps": step,
        "session": session.payload(),
    }


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--actor", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "s4" / "actors" / "s0_flat.onnx")
    parser.add_argument("--steps", type=int, default=3000, help="control steps to run (50 Hz)")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="status/receipt output dir")
    args = parser.parse_args(argv)
    status = run_node(actor_path=args.actor, max_steps=args.steps, out_dir=args.out)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
