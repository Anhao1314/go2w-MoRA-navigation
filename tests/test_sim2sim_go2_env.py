"""Pin the MuJoCo S4 rollout environment against the Isaac contract.

These tests need only mujoco + numpy (CI-safe, no Isaac/GPU): the model is the
committed ``models/go2/go2_flat_scene.xml`` and the policy stubs are zeros.
They pin the pieces the S4 result depends on:

- joint order: the Isaac grouping (hips, then thighs, then calves) is what the
  policy sees, regardless of the MJCF's per-leg layout;
- the DC-motor saturation boundaries (stall/cont/no-load speeds);
- observation dimensions and reset pose ranges;
- seed determinism (same seed → identical trajectory);
- the MVP.md "single-joint/stand" smoke test: zero action holds the stance
  without a base-contact fall.
"""

from __future__ import annotations

import unittest

import numpy as np

from locomotion.sim2sim import go2_env as g2e
from locomotion.sim2sim.go2_env import (
    DEFAULT_JOINT_POS_ISAAC,
    ISAAC_JOINT_NAMES,
    NUM_OBS_S0,
    NUM_OBS_SEQUENCE,
    Go2Sim2SimEnv,
    dc_motor_saturation,
)


class TestContractConstants(unittest.TestCase):
    def test_isaac_joint_order(self):
        # grouped by joint type, FL/FR/RL/RR within each group
        self.assertEqual(
            ISAAC_JOINT_NAMES,
            (
                "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            ),
        )

    def test_default_pose_matches_isaac_init(self):
        np.testing.assert_allclose(
            DEFAULT_JOINT_POS_ISAAC,
            [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
        )


class TestDCMotorSaturation(unittest.TestCase):
    def test_zero_velocity_full_torque_band(self):
        self.assertAlmostEqual(dc_motor_saturation(23.5, 0.0), 23.5)
        self.assertAlmostEqual(dc_motor_saturation(-23.5, 0.0), -23.5)
        # desired torque beyond the continuous limit is clipped
        self.assertAlmostEqual(dc_motor_saturation(40.0, 0.0), 23.5)
        self.assertAlmostEqual(dc_motor_saturation(-40.0, 0.0), -23.5)

    def test_no_load_speed_blocks_driving_torque(self):
        # at +30 rad/s no positive torque may pass, braking torque still can
        self.assertEqual(dc_motor_saturation(10.0, 30.0), 0.0)
        self.assertAlmostEqual(dc_motor_saturation(-10.0, 30.0), -10.0)
        # mirrored at -30 rad/s
        self.assertEqual(dc_motor_saturation(-10.0, -30.0), 0.0)
        self.assertAlmostEqual(dc_motor_saturation(10.0, -30.0), 10.0)

    def test_half_speed_halves_driving_torque(self):
        self.assertAlmostEqual(dc_motor_saturation(20.0, 15.0), 11.75)


class TestObservationLayout(unittest.TestCase):
    def test_obs_dims(self):
        velocity = Go2Sim2SimEnv("velocity", seed=0)
        sequence = Go2Sim2SimEnv("sequence", seed=0)
        self.assertEqual(velocity.observation().shape, (NUM_OBS_S0,))
        self.assertEqual(sequence.observation().shape, (NUM_OBS_SEQUENCE,))
        with self.assertRaises(ValueError):
            Go2Sim2SimEnv("bogus")

    def test_joint_order_remap(self):
        import mujoco

        env = Go2Sim2SimEnv("velocity", seed=0)
        # place every joint at a distinctive angle, in Isaac order
        probe = np.linspace(-0.4, 0.4, 12)
        env.data.qpos[env.isaac_qpos_adr] = probe
        mujoco.mj_forward(env.model, env.data)
        obs = env.observation()
        np.testing.assert_allclose(obs[12:24], probe - DEFAULT_JOINT_POS_ISAAC, atol=1e-12)

    def test_reset_pose_ranges(self):
        for seed in range(5):
            env = Go2Sim2SimEnv("velocity", seed=seed)
            x, y, z = env.base_position
            self.assertLessEqual(abs(x), 0.5)
            self.assertLessEqual(abs(y), 0.5)
            self.assertAlmostEqual(z, 0.4)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        def rollout(seed: int) -> list[float]:
            env = Go2Sim2SimEnv("velocity", seed=seed, velocity_command=(0.5, 0.0))
            rng = np.random.default_rng(123)
            trace = []
            for _ in range(100):
                obs = env.observation()
                trace.append(float(obs[0]))  # body-frame vx
                env.step(rng.normal(scale=0.1, size=12))
            return trace

        self.assertEqual(rollout(7), rollout(7))
        self.assertNotEqual(rollout(7), rollout(8))


class TestBodyFrameVelocityConvention(unittest.TestCase):
    """Empirical contract: world readings + manual Rᵀ (mj_objectVelocity local

    mode is mis-rotated for free-joint bodies — pinned here so a MuJoCo upgrade
    cannot silently flip the policy's velocity axes again).
    """

    def test_body_frame_velocities_follow_rotation_matrix(self):
        import mujoco

        env = Go2Sim2SimEnv("velocity", seed=0)
        rotation = env.data.xmat[env._base_body_id].reshape(3, 3)

        env.data.qvel[:] = 0
        env.data.qvel[0] = 1.0  # world +x translation
        mujoco.mj_forward(env.model, env.data)
        linear, angular = env._base_velocities_body_frame()
        np.testing.assert_allclose(linear, rotation.T @ np.array([1.0, 0.0, 0.0]), atol=1e-9)
        np.testing.assert_allclose(angular, [0.0, 0.0, 0.0], atol=1e-9)

        env.data.qvel[:] = 0
        env.data.qvel[5] = 1.0  # spin about the (yaw-aligned) vertical axis
        mujoco.mj_forward(env.model, env.data)
        linear, angular = env._base_velocities_body_frame()
        np.testing.assert_allclose(angular, rotation.T @ np.array([0.0, 0.0, 1.0]), atol=1e-9)
        # the world-frame reading is object-centered slightly off the body
        # origin, so a pure spin leaves a small residual linear term
        np.testing.assert_allclose(linear, [0.0, 0.0, 0.0], atol=0.05)


class TestS4BaselineConfig(unittest.TestCase):
    """The adopted engine-matching config (protocol TODO #2) must be applied."""

    def test_passive_params_zeroed_and_foot_solref_stiff(self):
        import mujoco as mj

        env = Go2Sim2SimEnv("velocity", seed=0)
        dofs = env.isaac_dof_adr
        np.testing.assert_allclose(env.model.dof_damping[dofs], 0.0)
        np.testing.assert_allclose(env.model.dof_frictionloss[dofs], 0.0)
        np.testing.assert_allclose(env.model.dof_armature[dofs], 0.0)
        # upstream XML stays unmodified on disk (config is load-time only)
        raw = mj.MjModel.from_xml_path(str(g2e.DEFAULT_MODEL_PATH))
        self.assertGreater(float(raw.dof_damping[dofs].max()), 0.0)
        for gid in env._foot_geom_ids:
            np.testing.assert_allclose(env.model.geom_solref[gid], (0.005, 1.0))


class TestExternalCommandInjection(unittest.TestCase):
    """ROS2 bridge entry point: live commands override, determinism preserved."""

    def test_external_command_applied_and_clamped(self):
        env = Go2Sim2SimEnv("velocity", seed=0)
        env.set_external_command(2.0, -3.0)  # beyond the frozen ±1 ranges
        command = env.command
        self.assertEqual(command[0], 1.0)  # vx clamped to +1
        self.assertEqual(command[1], 0.0)
        self.assertEqual(command[2], -1.0)  # wz clamped to -1
        # survives a resample boundary
        for _ in range(500):
            env.step(np.zeros(12))
        self.assertEqual(env.command[0], 1.0)
        self.assertEqual(env.command[2], -1.0)

    def test_no_external_command_keeps_sampled_stream(self):
        env = Go2Sim2SimEnv("velocity", seed=3)
        obs_a = env.reset()
        obs_b = Go2Sim2SimEnv("velocity", seed=3).reset()
        np.testing.assert_array_equal(obs_a, obs_b)  # unchanged default behavior

    def test_episode_steps_override(self):
        env = Go2Sim2SimEnv("velocity", seed=0, episode_steps=37)
        self.assertEqual(env.episode_steps, 37)
        for _ in range(37):
            env.step(np.zeros(12))
        self.assertTrue(env.over)


class TestStandingSmoke(unittest.TestCase):
    """MVP.md §2.2: stand test before any full episode."""

    def test_zero_action_does_not_fall_and_settles(self):
        # Explicit PD (kp=25) has a static gravity error of ~tau/kp, so a pure
        # PD hold settles into a crouch (same on the Isaac side); the learned
        # policy is what holds the nominal stance.  This pins: no base-contact
        # fall, bounded sag, bounded drift.
        env = Go2Sim2SimEnv("velocity", seed=0, velocity_command=(0.0, 0.0))
        zeros = np.zeros(12)
        for _ in range(250):  # 5 s at 50 Hz
            env.step(zeros)
            if env.over:
                break
        self.assertFalse(env.fell)
        self.assertGreater(env.base_position[2], 0.20)
        self.assertLess(abs(env.base_position[0]), 0.30)
        self.assertLess(abs(env.base_position[1]), 0.30)

    @unittest.skipUnless(
        (g2e.REPO_ROOT / "logs" / "s4" / "actors" / "s0_flat.onnx").exists(),
        "exported s0_flat actor not present (run scripts/export_s4_policy.py)",
    )
    def test_real_policy_holds_stance(self):
        from locomotion.sim2sim.go2_env import Sim2SimPolicy

        policy = Sim2SimPolicy(mode="s0", s0_path=g2e.REPO_ROOT / "logs" / "s4" / "actors" / "s0_flat.onnx")
        env = Go2Sim2SimEnv("velocity", seed=0, velocity_command=(0.0, 0.0))
        obs = env.reset()
        start_xy = env.base_position[0:2].copy()
        for _ in range(250):  # 5 s
            env.step(policy.act(obs))
            obs = env.observation()
            if env.over:
                break
        self.assertFalse(env.fell)
        self.assertGreater(env.base_position[2], 0.25)
        np.testing.assert_allclose(env.base_position[0:2], start_xy, atol=0.30)


if __name__ == "__main__":
    unittest.main()
