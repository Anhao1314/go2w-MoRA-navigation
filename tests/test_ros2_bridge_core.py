"""Contract tests for the rclpy-free bridge core (no ROS 2 required).

Pins the pieces the ROS2 acceptance depends on: the topic/protocol tables, the
command gate (clamping + deadman zeroing), the payload structures (frames,
empty effort, canonical JSON), and the quaternion convention.  The joint order
is additionally cross-checked against the committed MuJoCo model so the ROS
interface cannot drift from the simulator artifact.
"""

from __future__ import annotations

import json
import pathlib
import unittest

from ros2.go2_bridge import core

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestProtocol(unittest.TestCase):
    def test_protocol_json_matches_core_tables(self):
        protocol = core.load_protocol(REPO_ROOT)
        self.assertEqual(protocol["schema"], core.SCHEMA)
        self.assertEqual(protocol["topics"], core.TOPICS)
        self.assertEqual(protocol["joint_order"], list(core.JOINT_ORDER))
        self.assertEqual(protocol["authority"], "SIM_ONLY_CLOSED_LOOP_NO_HARDWARE")

    def test_joint_order_matches_committed_model(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "models/go2/go2_flat_scene.xml"))
        model_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                              model.actuator_trnid[i, 0])
            for i in range(model.nu)
        ]
        self.assertEqual(model_names, list(core.JOINT_ORDER))

    def test_published_topics_exclude_raw_motor_commands(self):
        for topic in core.PUBLISHED_TOPICS:
            self.assertNotIn("motor", topic)
            self.assertNotIn("torque", topic)
        self.assertIn("/go2/cmd_vel", core.TOPICS.values())  # the one subscription


class TestCommandGate(unittest.TestCase):
    def test_clamps_to_frozen_ranges(self):
        gate = core.CommandGate()
        applied = gate.update(2.0, -5.0, now_ns=0)
        self.assertEqual(applied, (1.0, -1.0))

    def test_deadman_zeroes_stale_command(self):
        gate = core.CommandGate(deadman_s=0.5)
        gate.update(0.5, 0.5, now_ns=0)
        self.assertEqual(gate.current(int(0.4e9)), (0.5, 0.5))
        self.assertEqual(gate.current(int(0.6e9)), (0.0, 0.0))  # stale -> zero
        self.assertEqual(gate.current(int(10.0e9)), (0.0, 0.0))

    def test_fresh_command_beats_deadman(self):
        gate = core.CommandGate(deadman_s=0.5)
        gate.update(0.5, 0.5, now_ns=0)
        gate.update(0.2, -0.2, now_ns=int(0.6e9))
        self.assertEqual(gate.current(int(0.7e9)), (0.2, -0.2))

    def test_never_updated_gate_is_zero(self):
        self.assertEqual(core.CommandGate().current(0), (0.0, 0.0))


class TestPayloads(unittest.TestCase):
    def _sample(self) -> core.LiveSample:
        names = list(core.JOINT_ORDER)
        return core.LiveSample(
            time_ns=2_000_000_000,
            episode_id=1,
            step=100,
            joint_names=tuple(names),
            joint_position_rad=tuple(0.1 * i for i in range(12)),
            joint_velocity_rad_s=tuple(0.0 for _ in range(12)),
            base_position_m=(1.0, 2.0, 0.33),
            base_quaternion_xyzw=(0.0, 0.0, 0.7071, 0.7071),
            command_applied=(0.5, 0.8),
        )

    def test_quaternion_wxyz_to_xyzw(self):
        # 90° yaw in MuJoCo wxyz = (cos45, 0, 0, sin45)
        q = core.quat_wxyz_to_xyzw((0.7071067811865476, 0.0, 0.0, 0.7071067811865475))
        self.assertAlmostEqual(q[0], 0.0, places=12)
        self.assertAlmostEqual(q[3], 0.7071067811865476, places=12)
        with self.assertRaises(ValueError):
            core.quat_wxyz_to_xyzw((0.0, 0.0, 0.0, 0.0))

    def test_joint_state_payload_contract(self):
        payload = self._sample().joint_state_payload()
        self.assertEqual(payload["frame_id"], "")  # no fabricated shared frame
        self.assertEqual(payload["effort"], [])  # never fabricated
        self.assertEqual(payload["joint_names"], list(core.JOINT_ORDER))
        self.assertEqual(payload["command_applied"], [0.5, 0.8])

    def test_pose_and_tf_payload_frames(self):
        sample = self._sample()
        pose = sample.pose_payload()
        self.assertEqual(pose["frame_id"], "mujoco_world")
        self.assertEqual(pose["child_frame_id"], "base")
        tf = sample.tf_payload()
        self.assertEqual((tf["frame_id"], tf["child_frame_id"]), ("mujoco_world", "base"))
        self.assertEqual(tf["rotation_xyzw"], list(sample.base_quaternion_xyzw))

    def test_live_sample_rejects_mismatched_joint_lengths(self):
        with self.assertRaises(ValueError):
            core.LiveSample(
                time_ns=0, episode_id=0, step=0,
                joint_names=core.JOINT_ORDER,
                joint_position_rad=(0.0,) * 11,
                joint_velocity_rad_s=(0.0,) * 12,
                base_position_m=(0.0, 0.0, 0.0),
                base_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                command_applied=(0.0, 0.0),
            )

    def test_session_payload_authority(self):
        session = core.SessionState(fall_count=1, reset_count=2)
        payload = session.payload()
        self.assertEqual(payload["authority"], "SIM_ONLY_CLOSED_LOOP_NO_HARDWARE")
        self.assertEqual(payload["fall_count"], 1)

    def test_canonical_json_is_deterministic(self):
        first = core.canonical_json({"b": 1, "a": [2, 3]})
        second = core.canonical_json({"a": [2, 3], "b": 1})
        self.assertEqual(first, second)
        json.loads(first)  # must stay valid JSON


if __name__ == "__main__":
    unittest.main()
