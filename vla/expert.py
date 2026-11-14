"""Scripted pick-and-place expert for the SO-101 tabletop task.

Two layers, deliberately separated:

1. :class:`PickPlacePhaseMachine` — **pure logic, no simulator**: given a
   ground-truth geometry snapshot (cube/box/end-effector positions, gripper
   state) it returns the next end-effector waypoint and gripper command,
   advancing through a fixed phase ladder.  Every threshold is a named
   constant and every transition is unit-tested; this is the part a reviewer
   can audit without running MuJoCo.

2. :class:`EndEffectorIK` — converts an end-effector waypoint into SO-101
   joint targets with damped-least-squares IK on the MuJoCo model.

The expert never touches lerobot; the env wrapper feeds it ground-truth
positions read from the MuJoCo data (documented in ``env.py`` — the VLA
itself sees only cameras + joints, the expert is privileged by design).

Phase ladder::

    APPROACH_ABOVE_CUBE → DESCEND_TO_CUBE → CLOSE_GRIPPER → LIFT
      → TRANSPORT_ABOVE_BOX → DESCEND_TO_BOX → OPEN_GRIPPER → RETREAT → DONE

Failure handling: a cube lost during carry phases triggers a retry (with a
grace period for contact registration), and drifting away from the cube while
the gripper closes restarts the approach.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Tunable task geometry (SO-101 tabletop scale, metres)."""

    hover_height: float = 0.06      # standoff above cube / box during approach
    grasp_depth: float = 0.005      # descent past the cube centre for the grasp
    position_tolerance: float = 0.008   # waypoint reached when EE is this close
    grasp_tolerance: float = 0.010      # |ee - cube| must hold while gripper closes
    settle_steps: int = 8               # control steps to dwell at each waypoint
    lost_cube_grace_steps: int = 5      # contact-registration grace before retry


CARRY_PHASES = frozenset({"LIFT", "TRANSPORT_ABOVE_BOX", "DESCEND_TO_BOX"})

# Gripper motor is slower than the arm: closing/opening completes on the
# MEASURED jaw ratio, not on a fixed dwell (8 steps only moved it 0.92→0.61).
GRIPPER_CLOSED_RATIO = 0.25
GRIPPER_OPEN_RATIO = 0.75
GRIPPER_MAX_STEPS = 90  # fail-safe: advance anyway if the motor never reaches the target


class PickPlacePhase(enum.Enum):
    APPROACH_ABOVE_CUBE = "APPROACH_ABOVE_CUBE"
    DESCEND_TO_CUBE = "DESCEND_TO_CUBE"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    TRANSPORT_ABOVE_BOX = "TRANSPORT_ABOVE_BOX"
    DESCEND_TO_BOX = "DESCEND_TO_BOX"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    RETREAT = "RETREAT"
    DONE = "DONE"


@dataclass(frozen=True)
class ExpertObservation:
    """Ground-truth geometry snapshot fed to the machine by the env wrapper."""

    ee_pos: tuple[float, float, float]
    cube_pos: tuple[float, float, float]
    box_pos: tuple[float, float, float]
    gripper_open: bool
    cube_grasped: bool  # from the sim: cube follows the gripper
    gripper_ratio: float = 0.5  # jaw opening ratio: 1 = fully open, 0 = fully closed


@dataclass(frozen=True)
class ExpertCommand:
    """Next waypoint + gripper command emitted by the machine."""

    target_ee_pos: tuple[float, float, float]
    gripper_open: bool
    phase: PickPlacePhase
    steps_in_phase: int

    @property
    def done(self) -> bool:
        return self.phase is PickPlacePhase.DONE


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


_LADDER = (
    PickPlacePhase.APPROACH_ABOVE_CUBE,
    PickPlacePhase.DESCEND_TO_CUBE,
    PickPlacePhase.CLOSE_GRIPPER,
    PickPlacePhase.LIFT,
    PickPlacePhase.TRANSPORT_ABOVE_BOX,
    PickPlacePhase.DESCEND_TO_BOX,
    PickPlacePhase.OPEN_GRIPPER,
    PickPlacePhase.RETREAT,
    PickPlacePhase.DONE,
)


class PickPlacePhaseMachine:
    """Deterministic phase ladder; the same observations always produce the
    same command sequence, which is what makes the dataset reproducible."""

    def __init__(self, geometry: Geometry | None = None) -> None:
        self.geo = geometry or Geometry()
        self.phase = PickPlacePhase.APPROACH_ABOVE_CUBE
        self.steps_in_phase = 0
        self.lost_steps = 0

    def reset(self) -> None:
        self.phase = PickPlacePhase.APPROACH_ABOVE_CUBE
        self.steps_in_phase = 0
        self.lost_steps = 0

    def _advance(self) -> None:
        index = _LADDER.index(self.phase)
        self.phase = _LADDER[min(index + 1, len(_LADDER) - 1)]
        self.steps_in_phase = 0
        self.lost_steps = 0

    def _waypoint(self, obs: ExpertObservation) -> tuple[tuple[float, float, float], bool]:
        geo = self.geo
        cube_above = (obs.cube_pos[0], obs.cube_pos[1], obs.cube_pos[2] + geo.hover_height)
        grasp = (obs.cube_pos[0], obs.cube_pos[1], obs.cube_pos[2] - geo.grasp_depth)
        box_above = (obs.box_pos[0], obs.box_pos[1], obs.box_pos[2] + geo.hover_height)
        box_inside = (obs.box_pos[0], obs.box_pos[1], obs.box_pos[2] + geo.grasp_depth)
        table = {
            PickPlacePhase.APPROACH_ABOVE_CUBE: (cube_above, True),
            PickPlacePhase.DESCEND_TO_CUBE: (grasp, True),
            PickPlacePhase.CLOSE_GRIPPER: (grasp, False),
            PickPlacePhase.LIFT: (cube_above, False),
            PickPlacePhase.TRANSPORT_ABOVE_BOX: (box_above, False),
            PickPlacePhase.DESCEND_TO_BOX: (box_inside, False),
            PickPlacePhase.OPEN_GRIPPER: (box_inside, True),
            PickPlacePhase.RETREAT: (box_above, True),
            PickPlacePhase.DONE: (obs.ee_pos, False),
        }
        return table[self.phase]

    def update(self, obs: ExpertObservation) -> ExpertCommand:
        if self.phase is PickPlacePhase.DONE:
            return ExpertCommand(obs.ee_pos, False, self.phase, self.steps_in_phase)

        # lost cube during carry: allow a contact-registration grace, then retry
        if self.phase.value in CARRY_PHASES and not obs.cube_grasped:
            self.lost_steps += 1
            if self.lost_steps > self.geo.lost_cube_grace_steps:
                self.reset()
                return self.update(obs)
        else:
            self.lost_steps = 0

        waypoint, gripper_open = self._waypoint(obs)
        self.steps_in_phase += 1
        reached = _dist(obs.ee_pos, waypoint) <= self.geo.position_tolerance

        if self.phase is PickPlacePhase.CLOSE_GRIPPER:
            if _dist(obs.ee_pos, obs.cube_pos) > self.geo.grasp_tolerance:
                self.reset()  # drifted while closing: restart the approach
                return self.update(obs)
            if obs.gripper_ratio <= GRIPPER_CLOSED_RATIO or self.steps_in_phase >= GRIPPER_MAX_STEPS:
                self._advance()
                return self.update(obs)
            return ExpertCommand(waypoint, False, self.phase, self.steps_in_phase)

        if self.phase is PickPlacePhase.OPEN_GRIPPER:
            if obs.gripper_ratio >= GRIPPER_OPEN_RATIO or self.steps_in_phase >= GRIPPER_MAX_STEPS:
                self._advance()
                return self.update(obs)
            return ExpertCommand(waypoint, True, self.phase, self.steps_in_phase)

        # position-gated phases: reach the waypoint and dwell, then advance
        if reached and self.steps_in_phase >= self.geo.settle_steps:
            self._advance()
            return self.update(obs)
        return ExpertCommand(waypoint, gripper_open, self.phase, self.steps_in_phase)


class EndEffectorIK:
    """Damped-least-squares IK: joint targets that move the EE to a waypoint.

    Operates directly on a MuJoCo ``model``/``data`` pair; the caller steps the
    simulation with the returned joint targets (position actuators).  Solves
    per step from the live state, so drift/self-collision is handled by the
    damping rather than by planning.
    """

    def __init__(self, model, data, *, joint_ids, site_name: str | None = None,
                 body_name: str | None = None, damping: float = 0.01,
                 step_scale: float = 1.0, max_iters: int = 100) -> None:
        import mujoco

        import numpy as np

        if (site_name is None) == (body_name is None):
            raise ValueError("exactly one of site_name / body_name is required")
        self._mujoco = mujoco
        self.np = np
        self.model = model
        self.data = data
        self.frame_kind = "site" if site_name is not None else "body"
        self.frame_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE if site_name is not None else mujoco.mjtObj.mjOBJ_BODY,
            site_name if site_name is not None else body_name,
        )
        if self.frame_id < 0:
            raise ValueError(f"frame {site_name or body_name!r} not found in model")
        self.joint_ids = tuple(joint_ids)
        self.damping = damping
        self.step_scale = step_scale
        self.max_iters = max_iters

    def _frame_position(self):
        if self.frame_kind == "site":
            return self.data.site_xpos[self.frame_id].copy()
        return self.data.xpos[self.frame_id].copy()

    def _frame_jacobian(self, jacp, jacr):
        mujoco = self._mujoco
        if self.frame_kind == "site":
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.frame_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.frame_id)

    def solve(self, target_pos, target_z_axis=None) -> list[float]:
        """Iterate DLS until the frame reaches ``target_pos`` (within 2 mm).

        ``target_z_axis`` (unit vector, optional) adds an orientation constraint:
        the frame's local z-axis is pulled toward it (e.g. (0, 0, -1) for a
        top-down grasp).  Position error and axis-alignment error are stacked
        into one DLS system; with fewer joints than constraints the damping
        picks the least-squares compromise, which is what we want — position
        stays prioritized because its error rows come first and orientation
        error is scaled below 1.
        """
        np = self.np
        mujoco = self._mujoco
        target = np.asarray(target_pos, dtype=float)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        desired = None
        if target_z_axis is not None:
            desired = np.asarray(target_z_axis, dtype=float)
            desired = desired / float(np.linalg.norm(desired))
        for _ in range(self.max_iters):
            mujoco.mj_forward(self.model, self.data)
            current = self._frame_position()
            error_pos = target - current
            if desired is None:
                error = error_pos
            else:
                z_cur = self._frame_z_axis()
                error_axis = np.cross(z_cur, desired) * 0.5  # rotation axis × sin(angle)
                error = np.concatenate([error_pos, error_axis])
                if float(np.linalg.norm(error_pos)) <= 2e-3 and float(
                    np.linalg.norm(error_axis)
                ) <= 2e-2:
                    break
            self._frame_jacobian(jacp, jacr)
            columns_p = [jacp[:, self.model.jnt_dofadr[j]] for j in self.joint_ids]
            jacobian = np.stack(columns_p, axis=1)  # 3 x n_joints
            if desired is not None:
                columns_r = [jacr[:, self.model.jnt_dofadr[j]] for j in self.joint_ids]
                rot = np.stack(columns_r, axis=1) * 0.5
                jacobian = np.vstack([jacobian, rot])
            JT = jacobian.T
            delta = JT @ np.linalg.solve(
                jacobian @ JT + self.damping * np.eye(jacobian.shape[0]), error
            )
            for k, j in enumerate(self.joint_ids):
                adr = self.model.jnt_qposadr[j]
                self.data.qpos[adr] += self.step_scale * float(delta[k])
        return [float(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self.joint_ids]

    def _frame_z_axis(self):
        if self.frame_kind == "site":
            return self.data.site_xmat[self.frame_id].reshape(3, 3)[:, 2].copy()
        return self.data.xmat[self.frame_id].reshape(3, 3)[:, 2].copy()
