"""Pure-numpy port of the L3 sequence stage machine for the MuJoCo side.

This is the S4 counterpart of ``locomotion/sequence/mdp/__init__.py`` (which
is torch/isaaclab-bound): the same constants, the same closed-loop stage
command and the same advance conditions, expressed against plain floats so a
MuJoCo rollout can drive the task without Isaac.  A unit test pins the
constants of both modules together so they cannot drift apart.

Known deliberate difference (documented, see ``locomotion/sim2sim/README.md``):
on the Isaac side both ``stage_reward`` and ``sequence_complete`` call
``_advance`` within one environment step, so the stop-hold counter accrues
twice per control step (the effective hold is ~0.5 s, not the nominal
``STOP_HOLD_STEPS = 50`` steps / 1 s).  This port counts once per control
step, i.e. it implements the *intended* 1 s hold and is therefore slightly
stricter than the Isaac-side task as evaluated.  The S4 report carries this
difference as a recorded caveat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Mirrors locomotion/sequence/mdp/__init__.py — kept in sync by a unit test.
FORWARD_DISTANCE_M = 2.0
TURN_RADIANS = 1.5707963267948966  # +90 degrees
POSITION_TOLERANCE_M = 0.30
HEADING_TOLERANCE_RAD = 0.175  # 10 degrees
STOP_SPEED_TOLERANCE_MPS = 0.10
STOP_DRIFT_TOLERANCE_M = 0.15
STOP_HOLD_STEPS = 50  # 1 s at 50 Hz
NUM_STAGES = 4


def wrap(angle: float) -> float:
    """Wrap to (-pi, pi], same semantics as the torch-side ``_wrap``."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class StageUpdate:
    """One control-step snapshot of the stage machine (obs + bookkeeping)."""

    stage: int
    command: tuple[float, float]  # closed-loop (vx, wz) for the current stage
    onehot: tuple[float, float, float, float]
    remaining: float  # distance (0/2), |heading error| (1), speed (3)
    heading_error: float  # wrapped error toward the +90 deg heading
    advanced: bool = False
    completed: bool = False
    extra: dict[str, float] = field(default_factory=dict)


class SequenceStageMachine:
    """Per-episode stage state; update() once per control step (50 Hz)."""

    def __init__(
        self,
        *,
        forward_distance_m: float = FORWARD_DISTANCE_M,
        turn_radians: float = TURN_RADIANS,
    ) -> None:
        # geometry overrides for the unseen-task generalization eval (P4);
        # defaults are the frozen L3 training geometry
        self._forward_distance_m = float(forward_distance_m)
        self._turn_radians = float(turn_radians)
        self._stage = 0
        self._start_x = 0.0
        self._start_y = 0.0
        self._heading0 = 0.0
        self._stop_steps = 0.0
        self._stop_anchor_x = 0.0
        self._stop_anchor_y = 0.0
        self._completed = False

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def completed(self) -> bool:
        return self._completed

    def reset(self, start_x: float, start_y: float, heading0: float) -> None:
        """Event-term equivalent of ``mdp.reset_stage`` (mode='reset')."""
        self._stage = 0
        self._start_x = start_x
        self._start_y = start_y
        self._heading0 = heading0
        self._stop_steps = 0.0
        self._stop_anchor_x = start_x
        self._stop_anchor_y = start_y
        self._completed = False

    def _targets(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(P1, P2) in world coordinates, anchored on the initial heading."""
        forward1 = (math.cos(self._heading0), math.sin(self._heading0))
        forward2 = (math.cos(self._heading0 + self._turn_radians), math.sin(self._heading0 + self._turn_radians))
        p1 = (
            self._start_x + self._forward_distance_m * forward1[0],
            self._start_y + self._forward_distance_m * forward1[1],
        )
        p2 = (p1[0] + self._forward_distance_m * forward2[0], p1[1] + self._forward_distance_m * forward2[1])
        return p1, p2

    def targets(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Public accessor for (P1, P2) — the S4 evaluator needs P2 as the
        commanded-integral target of the terminal-position success check."""
        return self._targets()

    def _heading_error(self, heading: float) -> float:
        return wrap(self._heading0 + self._turn_radians - heading)

    def _command(self, pos_x: float, pos_y: float, heading: float) -> tuple[float, float]:
        """``mdp.stage_command`` for a single env: closed-loop (vx, wz)."""
        stage = self._stage
        if stage >= 3:
            return (0.0, 0.0)

        if stage == 1:
            turn_error = self._heading_error(heading)
            return (0.0, max(-1.0, min(1.0, 2.0 * turn_error)))

        p1, p2 = self._targets()
        target = p2 if stage == 2 else p1
        delta_x = target[0] - pos_x
        delta_y = target[1] - pos_y
        desired_heading = math.atan2(delta_y, delta_x)
        walk_error = wrap(desired_heading - heading)
        walk_speed = 0.6 * max(0.15, min(1.0, 1.0 - 0.8 * abs(walk_error)))
        walk_turn = max(-1.0, min(1.0, 2.0 * walk_error))
        return (walk_speed, walk_turn)

    def _remaining(self, pos_x: float, pos_y: float, heading: float, speed: float) -> float:
        """``mdp._remaining``: distance (0/2), |heading error| (1), speed (3)."""
        p1, p2 = self._targets()
        if self._stage == 1:
            return abs(self._heading_error(heading))
        if self._stage == 3:
            return speed
        target = p2 if self._stage == 2 else p1
        return math.hypot(target[0] - pos_x, target[1] - pos_y)

    def update(
        self, pos_x: float, pos_y: float, heading: float, speed: float
    ) -> StageUpdate:
        """Advance the machine and return this step's command/observation terms.

        ``speed`` is the planar base speed (world frame), matching
        ``norm(root_lin_vel_w[:, :2])`` on the Isaac side.
        """
        p1, p2 = self._targets()
        heading_error = self._heading_error(heading)
        abs_heading_error = abs(heading_error)

        # hold timer only accumulates during the stop stage; judged on drift
        drift = math.hypot(pos_x - self._stop_anchor_x, pos_y - self._stop_anchor_y)
        if self._stage == 3 and drift <= STOP_DRIFT_TOLERANCE_M:
            self._stop_steps += 1.0
        else:
            self._stop_steps = 0.0

        advanced = False
        if self._stage == 0 and math.hypot(p1[0] - pos_x, p1[1] - pos_y) <= POSITION_TOLERANCE_M:
            advanced = True
        elif self._stage == 1 and abs_heading_error <= HEADING_TOLERANCE_RAD:
            advanced = True
        elif self._stage == 2 and math.hypot(p2[0] - pos_x, p2[1] - pos_y) <= POSITION_TOLERANCE_M:
            advanced = True
        elif self._stage == 3 and self._stop_steps >= STOP_HOLD_STEPS:
            advanced = True

        if advanced:
            self._stage += 1
            if self._stage >= NUM_STAGES:
                self._completed = True
            if self._stage == 3:
                # anchor the stop stage at the position where it began
                self._stop_anchor_x = pos_x
                self._stop_anchor_y = pos_y

        command = self._command(pos_x, pos_y, heading)
        remaining = self._remaining(pos_x, pos_y, heading, speed)
        onehot_index = min(self._stage, NUM_STAGES - 1)
        onehot = tuple(1.0 if i == onehot_index else 0.0 for i in range(NUM_STAGES))
        return StageUpdate(
            stage=self._stage,
            command=command,
            onehot=onehot,
            remaining=remaining,
            heading_error=heading_error,
            advanced=advanced,
            completed=self._completed,
            extra={"drift": drift, "stop_steps": self._stop_steps, "speed": speed},
        )
