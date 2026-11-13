"""Pin the stage-logic port against the Isaac-side mdp source of truth.

The S4 sequence task is driven on the MuJoCo side by
``locomotion.sequence.stage_logic`` (pure numpy).  These tests keep the port
honest: constants are asserted equal to ``locomotion.sequence.mdp`` (torch is
available in CI), and every stage transition / command formula is pinned with
hand-computed geometry from the frozen L3 setup (start at origin, heading 0:
P1 = (2, 0), turned heading +90°, P2 = (2, 2)).
"""

from __future__ import annotations

import math
import unittest

from locomotion.sequence import mdp as isaaclab_mdp
from locomotion.sequence import stage_logic

machine_factory = stage_logic.SequenceStageMachine


class TestConstantsMatchIsaacSource(unittest.TestCase):
    """The numpy port may not drift from the Isaac mdp constants."""

    def test_constants_are_identical(self):
        pairs = [
            ("FORWARD_DISTANCE_M",),
            ("TURN_RADIANS",),
            ("POSITION_TOLERANCE_M",),
            ("HEADING_TOLERANCE_RAD",),
            ("STOP_SPEED_TOLERANCE_MPS",),
            ("STOP_DRIFT_TOLERANCE_M",),
            ("STOP_HOLD_STEPS",),
            ("NUM_STAGES",),
        ]
        for (name,) in pairs:
            self.assertEqual(getattr(stage_logic, name), getattr(isaaclab_mdp, name), name)


class TestStageTransitions(unittest.TestCase):
    """Hand-computed walk-through of the L3 ladder from origin, heading 0."""

    def setUp(self):
        self.machine = machine_factory()
        self.machine.reset(0.0, 0.0, 0.0)

    def test_stage0_command_tracks_p1(self):
        # on the line to P1 = (2, 0), heading aligned: full speed, no turn
        update = self.machine.update(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(update.stage, 0)
        self.assertAlmostEqual(update.command[0], 0.6, places=12)
        self.assertAlmostEqual(update.command[1], 0.0, places=12)
        self.assertAlmostEqual(update.remaining, 1.0, places=12)
        self.assertEqual(update.onehot, (1.0, 0.0, 0.0, 0.0))

        # heading offset 0.1 rad: wz = 2 * wrap(0 - 0.1) = -0.2,
        # vx = 0.6 * clip(1 - 0.8 * 0.1, ...) = 0.552
        update = self.machine.update(1.0, 0.0, 0.1, 0.0)
        self.assertAlmostEqual(update.command[0], 0.552, places=12)
        self.assertAlmostEqual(update.command[1], -0.2, places=12)

    def test_stage0_advances_at_p1(self):
        # 0.05 m from P1 is inside the 0.30 m tolerance
        update = self.machine.update(2.05, 0.0, 0.0, 0.0)
        self.assertTrue(update.advanced)
        self.assertEqual(update.stage, 1)

        # 0.31 m short of P1 must not advance
        other = machine_factory()
        other.reset(0.0, 0.0, 0.0)
        update = other.update(1.69, 0.0, 0.0, 0.0)
        self.assertFalse(update.advanced)
        self.assertEqual(update.stage, 0)

    def test_stage1_turns_in_place_toward_plus_90(self):
        self.machine.update(2.05, 0.0, 0.0, 0.0)  # -> stage 1
        update = self.machine.update(2.05, 0.0, 0.0, 0.0)
        self.assertEqual(update.stage, 1)
        self.assertEqual(update.command[0], 0.0)
        # wz = clip(2 * wrap(pi/2 - 0), -1, 1) = +1
        self.assertAlmostEqual(update.command[1], 1.0, places=12)
        self.assertAlmostEqual(update.remaining, math.pi / 2, places=12)
        self.assertEqual(update.onehot, (0.0, 1.0, 0.0, 0.0))

    def test_stage1_advances_when_heading_within_tolerance(self):
        self.machine.update(2.05, 0.0, 0.0, 0.0)  # -> stage 1
        # 0.17 rad of heading error is inside the 0.175 rad tolerance
        update = self.machine.update(2.05, 0.0, math.pi / 2 - 0.17, 0.0)
        self.assertTrue(update.advanced)
        self.assertEqual(update.stage, 2)

    def test_stage2_targets_p2(self):
        self.machine.update(2.05, 0.0, 0.0, 0.0)  # -> stage 1
        self.machine.update(2.05, 0.0, math.pi / 2, 0.0)  # -> stage 2
        # standing 0.05 m past P1 along +x: P2 = (2, 2) is behind-right, so the
        # desired heading is atan2(2, -0.05) = pi/2 + eps -> small corrections
        walk_error = math.atan2(2.0, -0.05) - math.pi / 2
        update = self.machine.update(2.05, 0.0, math.pi / 2, 0.0)
        self.assertEqual(update.stage, 2)
        self.assertAlmostEqual(update.command[0], 0.6 * (1.0 - 0.8 * abs(walk_error)), places=12)
        self.assertAlmostEqual(update.command[1], 2.0 * walk_error, places=12)
        # remaining distance to P2 = (2, 2) from (2.05, 0)
        self.assertAlmostEqual(update.remaining, math.hypot(-0.05, 2.0), places=12)

    def test_stage3_hold_requires_50_clean_steps(self):
        self.machine.update(2.05, 0.0, 0.0, 0.0)  # -> 1
        self.machine.update(2.05, 0.0, math.pi / 2, 0.0)  # -> 2
        self.machine.update(2.05, 1.95, math.pi / 2, 0.0)  # -> 3, anchor here
        update = self.machine.update(2.05, 1.95, math.pi / 2, 0.0)
        self.assertEqual(update.stage, 3)
        self.assertEqual(update.command, (0.0, 0.0))
        self.assertEqual(update.onehot, (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(update.extra["stop_steps"], 1.0)

        for step in range(2, 51):
            update = self.machine.update(2.05, 1.95, math.pi / 2, 0.0)
            self.assertEqual(update.extra["stop_steps"], float(step))
        self.assertTrue(update.completed)
        self.assertTrue(self.machine.completed)

    def test_stage3_drift_violation_resets_hold_timer(self):
        self.machine.update(2.05, 0.0, 0.0, 0.0)  # -> 1
        self.machine.update(2.05, 0.0, math.pi / 2, 0.0)  # -> 2
        self.machine.update(2.05, 1.95, math.pi / 2, 0.0)  # -> 3
        for _ in range(30):
            self.machine.update(2.05, 1.95, math.pi / 2, 0.0)
        # drift 0.2 m > 0.15 m tolerance: timer resets, no completion
        update = self.machine.update(2.05, 2.15, math.pi / 2, 0.0)
        self.assertEqual(update.extra["stop_steps"], 0.0)
        self.assertFalse(update.completed)
        # a clean hold from here still needs the full 50 steps
        for step in range(1, 51):
            update = self.machine.update(2.05, 1.95, math.pi / 2, 0.0)
        self.assertTrue(update.completed)

    def test_stage3_reanchor_on_entry(self):
        # drift is judged from where stage 3 BEGAN, not from reset origin
        self.machine.update(2.05, 0.0, 0.0, 0.0)  # -> 1
        self.machine.update(2.05, 0.0, math.pi / 2, 0.0)  # -> 2
        self.machine.update(1.9, 2.0, math.pi / 2, 0.0)  # -> 3, anchor (1.9, 2.0)
        update = self.machine.update(2.0, 2.0, math.pi / 2, 0.0)
        # 0.1 m from the anchor: inside tolerance, timer accumulates
        self.assertEqual(update.extra["stop_steps"], 1.0)


if __name__ == "__main__":
    unittest.main()
