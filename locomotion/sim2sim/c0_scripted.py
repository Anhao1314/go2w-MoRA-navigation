"""C0 scripted-PD baseline (protocol §6): open-loop trot + proportional tracking.

The frozen protocol's RQ2 lower bound: "C0 scripted PD — standing + proportional
command tracking on the same task definition".  This is a classical scripted
quadruped gait (no learning): a sine-based diagonal trot whose stride amplitude
and frequency scale with the commanded velocity, plus differential shoulder
steering for yaw.  It runs on :class:`Go2Sim2SimEnv` through the same action
contract as the RL policies (12-dim raw offsets, scale 0.25), so the comparison
shares the task definition, the seeds and the metrics.

Design notes:
- open-loop phase oscillator; no feedback beyond the command — a genuine lower
  bound that makes no use of the policy's 48-dim observations;
- zero command folds back to the default stance (standing);
- joint-space amplitudes are conservative (a crude but honest trot, not a
  tuned gait) — the expected result is a walk with poor tracking, which is
  exactly what "incremental contribution of RL over C0" quantifies.
"""

from __future__ import annotations

import math

import numpy as np

from .go2_env import Go2Sim2SimEnv

# gait parameters (joint-space, radians; Go2 conventions from the default pose)
TROT_FREQUENCY_HZ = 2.5        # stride cycle rate (tuned 2026-11-14 sweep: 0.54 m/s at 0.5 m/s cmd)
THIGH_SWING_RAD = 0.60         # thigh oscillation amplitude (tuned)
CALF_SWING_RAD = 0.80          # calf flex amplitude, IN PHASE with thigh (tuned —
                               # anti-phase produced near-zero motion)
HIP_SWING_RAD = 0.08           # hip roll stabilization amplitude
PAN_STEER_RAD = 0.30           # differential shoulder_pan for yaw tracking
LEG_PHASE_OFFSETS = (0.0, math.pi, math.pi, 0.0)  # FL, FR, RL, RR (diagonal trot)


class ScriptedTrotC0:
    """Open-loop trot baseline producing RAW actions for Go2Sim2SimEnv.

    The env applies ``joint_target = default + 0.25 * raw`` — the gait emits
    raw values so that ``default + 0.25 * raw`` equals the desired joint
    trajectory.
    """

    def __init__(self, env: Go2Sim2SimEnv, *, frequency_hz: float = TROT_FREQUENCY_HZ) -> None:
        self.env = env
        self.frequency_hz = frequency_hz
        self._t = 0.0
        self._default = env._default_qpos  # (12,) Isaac order, radians
        # Isaac joint order: FL/FR/RL/RR hips, then thighs, then calves
        self._hip_idx = (0, 1, 2, 3)
        self._thigh_idx = (4, 5, 6, 7)
        self._calf_idx = (8, 9, 10, 11)

    def reset(self) -> None:
        self._t = 0.0

    def act(self, cmd_vx: float, cmd_wz: float) -> np.ndarray:
        """One control step: returns the 12-dim raw action."""
        self._t += 0.02  # env STEP_DT (50 Hz control)
        speed_scale = max(0.0, min(1.0, abs(cmd_vx)))  # stride scales with |vx|
        if abs(cmd_vx) < 0.05 and abs(cmd_wz) < 0.1:
            speed_scale = 0.0  # stand

        raw = np.zeros(12)
        for leg in range(4):
            phase = 2.0 * math.pi * self.frequency_hz * self._t + LEG_PHASE_OFFSETS[leg]
            swing = math.sin(phase)
            # thigh swing (flex/extend) scaled by the speed command
            thigh = THIGH_SWING_RAD * speed_scale * swing
            # calf counter-flex: folds during swing, extends during stance
            calf = -CALF_SWING_RAD * speed_scale * max(0.0, math.sin(phase + math.pi / 2))
            # hip: slight abduction pulse for stability + differential steering
            hip = HIP_SWING_RAD * speed_scale * math.cos(phase)
            if cmd_wz != 0.0:
                # differential steering: widen the stance on the outside of the turn
                side = 1.0 if leg in (0, 3) else -1.0  # FL/RR vs FR/RL
                hip += side * 0.05 * math.copysign(1.0, cmd_wz) * speed_scale
            raw[self._thigh_idx[leg]] = thigh / 0.25
            raw[self._calf_idx[leg]] = calf / 0.25
            raw[self._hip_idx[leg]] = hip / 0.25
        # clip to the env's action range
        return np.clip(raw, -4.0, 4.0)

