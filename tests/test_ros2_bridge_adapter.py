"""MOCK_CONTRACT_ONLY tests for the bridge adapter layer (no real ROS 2).

Follows the sibling-repo bridge doctrine: these tests inject in-memory ROS
module doubles and therefore never demonstrate actual DDS delivery — that is
the container acceptance's job (``ros2/go2_bridge/run_r1_acceptance.sh`` and
the ``ros2-live-r1`` workflow).  What they DO pin:

- the ROS import contract (lazy imports, fail-closed without rclpy);
- node/publisher construction arguments and QoS values;
- the publish allowlist (no motor/torque topics ever);
- the closed-loop wiring: a published /go2/cmd_vel reaches the sim as an
  applied command and shows up in the session payload (real MuJoCo, fake ROS);
- the probe's behavioral verifier (_verify) on synthetic traffic: PASS on a
  consistent forward/turn capture, FAIL on tampering.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

from ros2.go2_bridge import core

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTOR = REPO_ROOT / "logs" / "s4" / "actors" / "s0_flat.onnx"


class _Stamp:
    sec = 0
    nanosec = 0


class _Header:
    def __init__(self):
        self.stamp = _Stamp()
        self.frame_id = ""


def _pose_stamped():
    message = types.SimpleNamespace(header=_Header())
    message.pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return message


def _twist():
    return types.SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                 angular=SimpleNamespace(x=0.0, y=0.0, z=0.0))


def _install_mock_ros(capture: list, subscription_topics: list):
    """Inject in-memory ROS modules whose constructors assert the contract."""
    rclpy = types.ModuleType("rclpy")
    qos_mod = types.ModuleType("rclpy.qos")

    class HistoryPolicy:
        KEEP_LAST = "KEEP_LAST"

    class ReliabilityPolicy:
        RELIABLE = "RELIABLE"

    class DurabilityPolicy:
        VOLATILE = "VOLATILE"

    class QoSProfile:
        def __init__(self, **kwargs):
            assert kwargs["history"] == "KEEP_LAST"
            assert kwargs["depth"] == 32
            assert kwargs["reliability"] == "RELIABLE"
            assert kwargs["durability"] == "VOLATILE"

    qos_mod.HistoryPolicy = HistoryPolicy
    qos_mod.ReliabilityPolicy = ReliabilityPolicy
    qos_mod.DurabilityPolicy = DurabilityPolicy
    qos_mod.QoSProfile = QoSProfile
    rclpy.qos = qos_mod

    geometry = types.ModuleType("geometry_msgs")
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.PoseStamped = _pose_stamped
    geometry_msg.Twist = _twist
    geometry.msg = geometry_msg

    rosgraph = types.ModuleType("rosgraph_msgs")
    rosgraph_msg = types.ModuleType("rosgraph_msgs.msg")

    class Clock:
        def __init__(self):
            self.clock = SimpleNamespace(sec=0, nanosec=0)

    rosgraph_msg.Clock = Clock
    rosgraph.msg = rosgraph_msg

    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")

    class JointState:
        def __init__(self):
            self.header = _Header()
            self.name = []
            self.position = []
            self.velocity = []
            self.effort = []

    sensor_msg.JointState = JointState
    sensor.msg = sensor_msg

    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")

    class String:
        def __init__(self):
            self.data = ""

    std_msg.String = String
    std.msg = std_msg

    class _FakePublisher:
        def __init__(self, topic):
            self.topic = topic
            self.subscription_count = 1

        def publish(self, message):
            capture.append((self.topic, message))

        def get_subscription_count(self):
            return self.subscription_count

    class _FakeNode:
        def __init__(self):
            self.subscriptions = subscription_topics

        def create_publisher(self, message_type, topic, qos):
            return _FakePublisher(topic)

        def create_subscription(self, message_type, topic, callback, qos):
            self.subscriptions.append(topic)
            return SimpleNamespace(topic=topic)

        def destroy_node(self):
            return True

    def create_node(name, **kwargs):
        assert kwargs.get("use_global_arguments") is False
        assert kwargs.get("enable_rosout") is False
        assert kwargs.get("start_parameter_services") is False
        assert kwargs.get("enable_logger_service") is False
        return _FakeNode()

    def spin_once(node, timeout_sec=0.0):
        return None

    def shutdown():
        return None

    def init(args=None):
        return None

    rclpy.create_node = create_node
    rclpy.spin_once = spin_once
    rclpy.shutdown = shutdown
    rclpy.init = init

    for name, module in {
        "rclpy": rclpy,
        "rclpy.qos": qos_mod,
        "geometry_msgs": geometry,
        "geometry_msgs.msg": geometry_msg,
        "rosgraph_msgs": rosgraph,
        "rosgraph_msgs.msg": rosgraph_msg,
        "sensor_msgs": sensor,
        "sensor_msgs.msg": sensor_msg,
        "std_msgs": std,
        "std_msgs.msg": std_msg,
    }.items():
        sys.modules[name] = module


def _remove_mock_ros():
    for name in ("rclpy", "rclpy.qos", "geometry_msgs", "geometry_msgs.msg",
                 "rosgraph_msgs", "rosgraph_msgs.msg", "sensor_msgs",
                 "sensor_msgs.msg", "std_msgs", "std_msgs.msg"):
        sys.modules.pop(name, None)


@unittest.skipUnless(ACTOR.exists(), "exported s0_flat actor not present (run scripts/export_s4_policy.py)")
class TestSimNodeAdapter(unittest.TestCase):
    """Real MuJoCo + policy, fake ROS: pins the closed-loop wiring."""

    def setUp(self):
        self.capture: list = []
        self.subscriptions: list = []
        _install_mock_ros(self.capture, self.subscriptions)
        self._old_env = dict(__import__("os").environ)
        __import__("os").environ["ROS_DOMAIN_ID"] = "77"
        __import__("os").environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

    def tearDown(self):
        _remove_mock_ros()
        __import__("os").environ.clear()
        __import__("os").environ.update(self._old_env)

    def test_environment_gate_blocks_bad_domain(self):
        from ros2.go2_bridge import sim_node

        __import__("os").environ["ROS_DOMAIN_ID"] = "200"
        with self.assertRaises(core.BridgeError):
            sim_node.run_node(actor_path=ACTOR, max_steps=5)
        __import__("os").environ["ROS_DOMAIN_ID"] = "77"

    def test_closed_loop_publishes_allowlist_and_applies_commands(self):
        from ros2.go2_bridge import sim_node

        status = sim_node.run_node(actor_path=ACTOR, max_steps=120)
        self.assertEqual(status["ros2"]["status"], "PUBLISHED_NOT_SUBSCRIBER_VERIFIED")
        self.assertFalse(status["ros2"]["tf_available"])  # mock lacks tf2_ros -> degrade

        topics = {topic for topic, _ in self.capture}
        for expected in ("/clock", "/go2/joint_states", "/go2/base_pose", "/go2/session_info"):
            self.assertIn(expected, topics)
        for topic in topics:
            self.assertNotIn("motor", topic)
            self.assertNotIn("torque", topic)
        self.assertIn("/go2/cmd_vel", self.subscriptions)

        session_rows = [
            json.loads(message.data)
            for topic, message in self.capture
            if topic == "/go2/session_info"
        ]
        self.assertTrue(session_rows)
        self.assertTrue(all(row["authority"] == "SIM_ONLY_CLOSED_LOOP_NO_HARDWARE" for row in session_rows))

        # deadman keeps the idle session at zero applied command
        applied = [tuple(row["command_applied"]) for row in session_rows]
        self.assertTrue(all(vx == 0.0 and wz == 0.0 for vx, wz in applied))


class TestProbeVerifier(unittest.TestCase):
    """The behavioral verifier on synthetic traffic (pure, no ROS)."""

    def _traffic(self, *, forward_displacement=0.9, turn_yaw=1.2):
        received = {key: [] for key in core.TOPICS}
        for step in range(600):  # 12 s at 50 Hz
            sim_t = step * 0.02
            clock = types.SimpleNamespace(clock=SimpleNamespace(sec=int(sim_t), nanosec=int((sim_t % 1.0) * 1e9)))
            received["clock"].append((clock, step))

            if step % 50 == 0 and len(received["session_info"]) < 12:
                vx, wz = (0.5, 0.0) if 2.0 <= sim_t < 5.0 else ((0.0, 0.8) if 5.0 <= sim_t < 8.0 else (0.0, 0.0))
                payload = core.SessionState(commands_accepted=7).payload()
                payload["command_applied"] = [vx, wz]
                message = types.SimpleNamespace(data=core.canonical_json(payload))
                received["session_info"].append((message, step))

            joint = types.SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(sec=int(sim_t), nanosec=0), frame_id=""),
                name=list(core.JOINT_ORDER),
                position=[0.0] * 12,
                velocity=[0.0] * 12,
                effort=[],
            )
            received["joint_states"].append((joint, step))

            # trajectory: walks forward during [2,5), turns during [5,8)
            if sim_t < 2.0:
                x, y, yaw = 0.0, 0.0, 0.0
            elif sim_t < 5.0:
                x, y, yaw = 0.5 * (sim_t - 2.0) * forward_displacement / 0.9, 0.0, 0.0
            else:
                x = 0.5 * forward_displacement / 0.9
                yaw = 0.8 * (sim_t - 5.0) * turn_yaw / 1.2
            yaw_half = yaw / 2.0
            quat = (0.0, 0.0, math_sin(yaw_half), math_cos(yaw_half))
            pose = types.SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=int(sim_t), nanosec=int((sim_t % 1.0) * 1e9)),
                    frame_id=core.FRAME_WORLD,
                ),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=y, z=0.33),
                    orientation=SimpleNamespace(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                ),
            )
            received["base_pose"].append((pose, step))
        return received

    def test_verifier_passes_consistent_traffic(self):
        from ros2.go2_bridge import probe

        verdict = probe._verify(self._traffic(), tf_verified=True)
        self.assertEqual(verdict["status"], "ROS2_ROUNDTRIP_PASS", verdict)
        self.assertGreater(verdict["forward_displacement_m"], 0.2)
        self.assertGreater(verdict["turn_yaw_change_rad"], 0.1)

    def test_verifier_fails_when_turn_does_not_happen(self):
        from ros2.go2_bridge import probe

        verdict = probe._verify(self._traffic(turn_yaw=0.05), tf_verified=True)
        self.assertEqual(verdict["status"], "ROS2_ROUNDTRIP_FAILED")
        self.assertIn("turn", verdict["reason"])

    def test_verifier_fails_without_forward_motion(self):
        from ros2.go2_bridge import probe

        verdict = probe._verify(self._traffic(forward_displacement=0.05), tf_verified=True)
        self.assertEqual(verdict["status"], "ROS2_ROUNDTRIP_FAILED")
        self.assertIn("displacement", verdict["reason"])

    def test_verifier_fails_on_bad_joint_names(self):
        from ros2.go2_bridge import probe

        traffic = self._traffic()
        first = traffic["joint_states"][0][0]
        first.name = ["wrong"] * 12
        verdict = probe._verify(traffic, tf_verified=True)
        self.assertEqual(verdict["status"], "ROS2_ROUNDTRIP_FAILED")
        self.assertIn("names", verdict["reason"])


def math_sin(x):
    import math

    return math.sin(x)


def math_cos(x):
    import math

    return math.cos(x)


if __name__ == "__main__":
    unittest.main()
