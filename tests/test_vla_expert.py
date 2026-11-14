"""Pick-place phase machine + IK smoke tests (no SO-101 assets needed).

The phase machine is pure logic: tests drive it with a synthetic perfect
executor (the end effector teleports to each commanded waypoint) and assert
the full ladder, the failure/retry paths, and determinism.  The IK smoke test
builds a minimal 3-joint MJCF inline so it runs anywhere mujoco is installed.
"""

from __future__ import annotations

import unittest

from vla.expert import (
    Geometry,
    ExpertObservation,
    PickPlacePhase,
    PickPlacePhaseMachine,
)

CUBE = (0.15, -0.10, 0.02)
BOX = (-0.12, 0.14, 0.01)
HOME = (0.0, 0.0, 0.20)


class TestPhaseLadder(unittest.TestCase):
    """A perfect executor walks the whole ladder to DONE."""

    def setUp(self):
        self.machine = PickPlacePhaseMachine()
        self.grasped = False
        self.ee = HOME

    def _step(self):
        gripper_open = self.machine.phase in (
            PickPlacePhase.APPROACH_ABOVE_CUBE,
            PickPlacePhase.DESCEND_TO_CUBE,
            PickPlacePhase.OPEN_GRIPPER,
            PickPlacePhase.RETREAT,
        )
        # the sim registers the grasp once the gripper closes on the cube
        if self.machine.phase is PickPlacePhase.LIFT and not self.grasped:
            self.grasped = True  # contact registers right after CLOSE dwell
        # jaw ratio model: fully open on approach, closing during CLOSE_GRIPPER
        ratio = 0.1 if self.machine.phase in (PickPlacePhase.CLOSE_GRIPPER, PickPlacePhase.LIFT,
                                              PickPlacePhase.TRANSPORT_ABOVE_BOX,
                                              PickPlacePhase.DESCEND_TO_BOX) else 0.9
        command = self.machine.update(
            ExpertObservation(
                ee_pos=self.ee,
                cube_pos=CUBE,
                box_pos=BOX,
                gripper_open=gripper_open,
                cube_grasped=self.grasped,
                gripper_ratio=ratio,
            )
        )
        # perfect executor: teleport the EE to the commanded waypoint
        self.ee = command.target_ee_pos
        return command

    def test_full_ladder_reaches_done(self):
        phases_seen = []
        for _ in range(400):
            command = self._step()
            phases_seen.append(command.phase)
            if command.done:
                break
        self.assertTrue(command.done, f"stuck at {command.phase}")
        for phase in list(PickPlacePhase)[:-1]:
            self.assertIn(phase, phases_seen, phase.value)

    def test_ladder_order_is_respected(self):
        order = [p for p in list(PickPlacePhase) if p is not PickPlacePhase.DONE]
        index_seen = []
        for _ in range(400):
            command = self._step()
            index_seen.append(order.index(command.phase) if command.phase in order else 99)
            if command.done:
                break
        self.assertEqual(index_seen, sorted(index_seen), "phases must never regress")


class TestFailurePaths(unittest.TestCase):
    def test_lost_cube_during_transport_retries_after_grace(self):
        machine = PickPlacePhaseMachine()
        ee = HOME
        # drive to TRANSPORT with a grasped cube (perfect executor, shortened)
        grasped = True
        for _ in range(400):
            command = machine.update(
                ExpertObservation(ee, CUBE, BOX, gripper_open=False, cube_grasped=grasped)
            )
            ee = command.target_ee_pos
            if command.phase is PickPlacePhase.LIFT:
                grasped = True
            if command.phase is PickPlacePhase.TRANSPORT_ABOVE_BOX:
                break
        self.assertIs(machine.phase, PickPlacePhase.TRANSPORT_ABOVE_BOX)

        # drop the cube for longer than the grace period
        for _ in range(Geometry().lost_cube_grace_steps + 1):
            machine.update(ExpertObservation(ee, CUBE, BOX, False, cube_grasped=False))
        self.assertIs(machine.phase, PickPlacePhase.APPROACH_ABOVE_CUBE)

    def test_lost_cube_within_grace_does_not_reset(self):
        machine = PickPlacePhaseMachine()
        ee = HOME
        grasped = True
        for _ in range(400):
            command = machine.update(
                ExpertObservation(ee, CUBE, BOX, False, cube_grasped=grasped)
            )
            ee = command.target_ee_pos
            if command.phase is PickPlacePhase.TRANSPORT_ABOVE_BOX:
                break
        for _ in range(Geometry().lost_cube_grace_steps):
            machine.update(ExpertObservation(ee, CUBE, BOX, False, cube_grasped=False))
        self.assertIs(machine.phase, PickPlacePhase.TRANSPORT_ABOVE_BOX)

    def test_drift_while_closing_restarts_approach(self):
        machine = PickPlacePhaseMachine()
        ee = HOME
        for _ in range(400):
            command = machine.update(
                ExpertObservation(ee, CUBE, BOX, True, cube_grasped=False)
            )
            ee = command.target_ee_pos
            if command.phase is PickPlacePhase.CLOSE_GRIPPER:
                break
        # EE drifts away from the cube mid-close
        machine.update(
            ExpertObservation((ee[0] + 0.10, ee[1], ee[2]), CUBE, BOX, False, False)
        )
        self.assertIs(machine.phase, PickPlacePhase.APPROACH_ABOVE_CUBE)


class TestDeterminism(unittest.TestCase):
    def test_same_observations_same_commands(self):
        def run():
            machine = PickPlacePhaseMachine()
            ee = HOME
            commands = []
            grasped = False
            for _ in range(400):
                command = machine.update(
                    ExpertObservation(ee, CUBE, BOX, True, cube_grasped=grasped)
                )
                commands.append((command.target_ee_pos, command.gripper_open, command.phase))
                ee = command.target_ee_pos
                if machine.phase is PickPlacePhase.LIFT:
                    grasped = True
                if command.done:
                    break
            return commands

        self.assertEqual(run(), run())


class TestIKSmoke(unittest.TestCase):
    """DLS IK converges on a tiny inline 3-joint arm."""

    MJCF = """
<mujoco model="ik_smoke">
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      <body pos="0.1 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
        <body pos="0.1 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.01"/>
          <site name="tip" pos="0.08 0 0" size="0.005"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

    def test_ik_reaches_reachable_target(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string(self.MJCF)
        data = mujoco.MjData(model)
        from vla.expert import EndEffectorIK

        ik = EndEffectorIK(model, data, site_name="tip", joint_ids=(0, 1, 2))
        # reachable point: radius within [0, 0.28], above the base plane
        data.qpos[:] = (0.0, 0.5, -1.0)  # start away from the target pose
        target = (0.10, 0.10, 0.05)
        solutions = ik.solve(target)
        mujoco.mj_forward(model, data)
        reached = data.site_xpos[ik.frame_id]
        error = sum((reached[i] - target[i]) ** 2 for i in range(3)) ** 0.5
        self.assertLess(error, 5e-3)
        self.assertEqual(len(solutions), 3)


if __name__ == "__main__":
    unittest.main()
