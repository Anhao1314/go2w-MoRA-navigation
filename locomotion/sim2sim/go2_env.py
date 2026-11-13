"""MuJoCo rollout environment matching the Isaac S4 contract (zero policy change).

This is the transfer side of RQ4: the Isaac-trained policies run unchanged on
top of the official legged Go2 MJCF (``models/go2/go2_flat_scene.xml``).  The
contract implemented here comes from the frozen Isaac configuration (see
``locomotion/sim2sim/README.md`` for the full mapping):

- timing: physics 200 Hz (``dt = 0.005``, set at load time) × decimation 4 →
  50 Hz control; velocity episodes 1000 steps, sequence episodes 3000 steps
- observation: 48-dim blind proprioceptive obs (velocity task) or the 53-dim
  sequence obs (sequence task), body-frame velocities via
  ``mj_objectVelocity(..., flg_local=True)``, projected gravity ``Rᵀ·(0,0,-1)``,
  joint terms relative to the Isaac default pose, per-joint order remapped
  from the MJCF's per-leg grouping to the Isaac per-joint-type grouping
- action: ``joint_target = default_joint_pos + 0.25 * raw_action`` (offset
  semantics), actuated by PD torque ``25·(q_target − q) − 0.5·q̇`` with the
  Isaac DC-motor saturation (stall 23.5 N·m, no-load 30 rad/s, cont 23.5 N·m)
- termination: net contact force on the base body > 1.0 N (Isaac
  ``base_contact``), or timeout; sequence success is reported by the stage
  machine (``locomotion.sequence.stage_logic``)

No Isaac Lab import; inference runs on ONNX (onnxruntime).  Everything is
seed-deterministic: one ``numpy.random.default_rng(seed)`` per episode covers
the reset pose (x, y ∈ ±0.5 m, yaw ∈ ±π) and the velocity-command stream
(same ranges/resampling as the Isaac command manager, different RNG engine —
recorded in the S4 report).
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

from ..sequence.stage_logic import NUM_STAGES, SequenceStageMachine

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "go2" / "go2_flat_scene.xml"

# --- Isaac-side frozen contract (locomotion/isaac/*, UnitreeGo2 base cfg) ---
PHYS_DT = 0.005
DECIMATION = 4
STEP_DT = PHYS_DT * DECIMATION  # 0.02 s -> 50 Hz control
PD_STIFFNESS = 25.0
PD_DAMPING = 0.5
TAU_STALL = 23.5
QD_MAX = 30.0
TAU_CONT = 23.5
ACTION_SCALE = 0.25
CONTACT_FORCE_THRESHOLD_N = 1.0
VELOCITY_EPISODE_STEPS = 1000
SEQUENCE_EPISODE_STEPS = 3000
COMMAND_RESAMPLE_STEPS = 500  # resampling_time_range = (10.0, 10.0) at 50 Hz
STANDING_PROBABILITY = 0.02  # rel_standing_envs
COMMAND_RANGE_VX = (-1.0, 1.0)
COMMAND_RANGE_WZ = (-1.0, 1.0)
RESET_RANGE_XY = 0.5
RESET_RANGE_YAW = math.pi
DEFAULT_BASE_Z = 0.4

# Isaac joint order: grouped by joint type (authoritative:
# locomotion/guardrail/go2_model.json, exported from the go2.usd runtime).
_LEG_PREFIXES = ("FL", "FR", "RL", "RR")
_JOINT_TYPES = ("hip", "thigh", "calf")
ISAAC_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{joint}_joint" for joint in _JOINT_TYPES for leg in _LEG_PREFIXES
)

# Isaac default joint pose (= the action offset anchor).  Sign convention from
# UnitreeGo2 init: .*L_hip_joint = +0.1, .*R_hip_joint = -0.1, front thigh 0.8,
# rear thigh 1.0, calf -1.5.
DEFAULT_JOINT_POS_ISAAC: np.ndarray = np.array(
    [
        0.1,   # FL_hip
        -0.1,  # FR_hip
        0.1,   # RL_hip
        -0.1,  # RR_hip
        0.8,   # FL_thigh
        0.8,   # FR_thigh
        1.0,   # RL_thigh
        1.0,   # RR_thigh
        -1.5,  # FL_calf
        -1.5,  # FR_calf
        -1.5,  # RL_calf
        -1.5,  # RR_calf
    ]
)

NUM_JOINTS = 12
NUM_OBS_S0 = 48
NUM_OBS_SEQUENCE = 53

# --- S4 baseline engine-matching config (frozen TODO #2 actuator decision) ---
# The Isaac training contract has PhysX joint damping/armature/frictionloss ≈ 0
# and stiff contacts; the menagerie MJCF ships interactive-sim values (damping
# 2.0 N·m·s/rad, frictionloss 0.2, armature 0.01, soft solref) that are NOT part
# of that contract.  Ablation with the deployed s0_flat policy (2026-11-14,
# recorded in the S4 report) measured, per variant, realized yaw rate at a
# +0.8 rad/s command and 4 s walk displacement at +0.5 m/s:
#
#   menagerie defaults        turn +0.111 rad/s   walk +1.274 m
#   damping=0                 turn +0.132         walk +1.504
#   all passive params = 0    turn +0.163         walk +1.661
#   + stiff foot contacts     turn +0.308         walk +1.672   <- adopted
#
# (torsional foot friction raised to 0.5 collapsed BOTH metrics to ~0 and is
# excluded).  Applied at load time so upstream files stay unmodified.
S4_BASELINE_LEG_DAMPING = 0.0
S4_BASELINE_LEG_FRICTIONLOSS = 0.0
S4_BASELINE_LEG_ARMATURE = 0.0
S4_BASELINE_FOOT_SOLREF = (0.005, 1.0)
FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")  # collision spheres on the calf bodies


def dc_motor_saturation(tau_desired: float, joint_velocity: float) -> float:
    """Isaac ``DCMotor`` torque-speed saturation for one joint.

    tau_max = clip(tau_stall * (1 - qd/qd_max), -inf, tau_cont)
    tau_min = clip(tau_stall * (-1 - qd/qd_max), -tau_cont, +inf)
    """
    tau_max = min(TAU_STALL * (1.0 - joint_velocity / QD_MAX), TAU_CONT)
    tau_min = max(TAU_STALL * (-1.0 - joint_velocity / QD_MAX), -TAU_CONT)
    return max(tau_min, min(tau_max, tau_desired))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class Go2Sim2SimEnv:
    """Deterministic MuJoCo rollout of one Go2 episode under the S4 contract."""

    def __init__(
        self,
        task: str,
        *,
        model_path: pathlib.Path | str | None = None,
        seed: int = 0,
        velocity_command: tuple[float, float] | None = None,
    ) -> None:
        if task not in ("velocity", "sequence"):
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.model_path = pathlib.Path(model_path or DEFAULT_MODEL_PATH)
        self.episode_steps = VELOCITY_EPISODE_STEPS if task == "velocity" else SEQUENCE_EPISODE_STEPS
        self.seed = int(seed)
        # fixed velocity command (S4 smoke/standing runs); None -> sampled per seed
        self._fixed_command = velocity_command

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = PHYS_DT
        self.data = mujoco.MjData(self.model)

        # per-joint-type (Isaac) -> per-leg (MJCF) index remaps, built from names
        # so any upstream model re-ordering is picked up automatically.
        mj_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ISAAC_JOINT_NAMES
        ]
        if any(jid < 0 for jid in mj_joint_ids):
            missing = [n for n, jid in zip(ISAAC_JOINT_NAMES, mj_joint_ids) if jid < 0]
            raise ValueError(f"model {self.model_path} is missing joints: {missing}")
        self.isaac_qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in mj_joint_ids]
        )
        self.isaac_dof_adr = np.array([self.model.jnt_dofadr[jid] for jid in mj_joint_ids])
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self._base_body_id < 0:
            raise ValueError(f"model {self.model_path} has no body named 'base'")

        # default pose in MJCF qpos layout
        self._default_qpos = np.array(self.data.qpos, dtype=float)
        self._default_qpos[self.isaac_qpos_adr] = DEFAULT_JOINT_POS_ISAAC
        # actuator order is the MJCF per-leg order; build name-based remap
        mj_act_ids = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0]): i
            for i in range(self.model.nu)
        }
        self._act_id_isaac = np.array(
            [mj_act_ids[name] for name in ISAAC_JOINT_NAMES], dtype=int
        )
        self._apply_s4_baseline_config()

        self._stage = SequenceStageMachine()
        self._rng = np.random.default_rng(self.seed)
        self._step_count = 0
        self._last_action = np.zeros(NUM_JOINTS)
        self._command = np.zeros(3)  # vx, vy, wz (velocity task)
        self._episode_over = False
        self._fell = False
        self._stage_update = None
        self.reset()

    def _apply_s4_baseline_config(self) -> None:
        """Engine-matching config (see the S4_BASELINE_* block above)."""
        dofs = self.isaac_dof_adr
        self.model.dof_damping[dofs] = S4_BASELINE_LEG_DAMPING
        self.model.dof_frictionloss[dofs] = S4_BASELINE_LEG_FRICTIONLOSS
        self.model.dof_armature[dofs] = S4_BASELINE_LEG_ARMATURE
        foot_ids = []
        for name in FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"model {self.model_path} is missing foot geom {name!r}")
            foot_ids.append(gid)
        self._foot_geom_ids = np.array(foot_ids, dtype=int)
        self.model.geom_solref[self._foot_geom_ids] = S4_BASELINE_FOOT_SOLREF

    # --- observation helpers -------------------------------------------------
    def _heading(self) -> float:
        rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        return math.atan2(rotation[1, 0], rotation[0, 0])

    def _base_velocities_body_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Base linear/angular velocity in the body frame (Isaac root_*_vel_b).

        Empirically pinned by tests: ``mj_objectVelocity(flg_local=1)`` returns
        mis-rotated readings for free-joint bodies, so we take the world-aligned
        object-centered velocity (flg_local=0, rot-first layout) and rotate the
        two 3-vectors into the body frame with the transposed body rotation.
        """
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._base_body_id,
            velocity,
            0,  # world-aligned axes, velocity of the body-frame origin
        )
        rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        linear_body = rotation.T @ velocity[3:6]
        angular_body = rotation.T @ velocity[0:3]
        return linear_body, angular_body

    def _joint_state_isaac_order(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self.data.qpos[self.isaac_qpos_adr]
        velocities = self.data.qvel[self.isaac_dof_adr]
        return positions, velocities

    def _observation(self) -> np.ndarray:
        linear, angular = self._base_velocities_body_frame()
        rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        joint_pos, joint_vel = self._joint_state_isaac_order()
        joint_pos_rel = joint_pos - DEFAULT_JOINT_POS_ISAAC
        if self.task == "velocity":
            return np.concatenate(
                [
                    linear,
                    angular,
                    projected_gravity,
                    self._command,
                    joint_pos_rel,
                    joint_vel,
                    self._last_action,
                ]
            )
        update = self._stage_update
        assert update is not None  # set by reset()/step()
        return np.concatenate(
            [
                linear,
                angular,
                projected_gravity,
                np.array(update.command),
                joint_pos_rel,
                joint_vel,
                self._last_action,
                np.array(update.onehot),
                np.array([update.remaining, update.heading_error]),
            ]
        )

    # --- actuation -----------------------------------------------------------
    def _apply_action(self, raw_action: np.ndarray) -> None:
        """PD + DC saturation for ``decimation`` physics steps at 200 Hz."""
        joint_target = DEFAULT_JOINT_POS_ISAAC + ACTION_SCALE * raw_action
        tau_isaac = np.zeros(NUM_JOINTS)
        for _ in range(DECIMATION):
            joint_pos, joint_vel = self._joint_state_isaac_order()
            for i in range(NUM_JOINTS):
                tau_desired = PD_STIFFNESS * (joint_target[i] - joint_pos[i]) - PD_DAMPING * joint_vel[i]
                tau_isaac[i] = dc_motor_saturation(tau_desired, joint_vel[i])
            self.data.ctrl[self._act_id_isaac] = tau_isaac
            mujoco.mj_step(self.model, self.data)

    def _base_contact_force(self) -> float:
        """Net contact-force magnitude on the base body (Isaac base_contact)."""
        total = 0.0
        force = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self._base_body_id in (body1, body2):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += float(np.linalg.norm(force[0:3]))
        return total

    # --- commands ------------------------------------------------------------
    def _sample_velocity_command(self) -> None:
        if self._fixed_command is not None:
            self._command[:] = (self._fixed_command[0], 0.0, self._fixed_command[1])
            return
        if self._rng.random() < STANDING_PROBABILITY:
            self._command[:] = (0.0, 0.0, 0.0)
            return
        self._command[:] = (
            self._rng.uniform(*COMMAND_RANGE_VX),
            0.0,
            self._rng.uniform(*COMMAND_RANGE_WZ),
        )

    # --- episode API ----------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._rng = np.random.default_rng(self.seed)
        self._step_count = 0
        self._episode_over = False
        self._fell = False
        self._last_action = np.zeros(NUM_JOINTS)

        x = self._rng.uniform(-RESET_RANGE_XY, RESET_RANGE_XY)
        y = self._rng.uniform(-RESET_RANGE_XY, RESET_RANGE_XY)
        yaw = self._rng.uniform(-RESET_RANGE_YAW, RESET_RANGE_YAW)
        self.data.qpos[:] = self._default_qpos
        self.data.qpos[0:2] = (x, y)
        self.data.qpos[2] = DEFAULT_BASE_Z
        self.data.qpos[3:7] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._sample_velocity_command()
        heading = self._heading()
        self._stage.reset(x, y, heading)
        # one update pass so the first observation is well-defined
        self._stage_update = self._stage.update(x, y, heading, 0.0)
        return self._observation()

    def step(self, raw_action: np.ndarray) -> None:
        """Apply one control step; the caller reads state via properties."""
        if self._episode_over:
            raise RuntimeError("episode is over; call reset() first")
        raw_action = np.asarray(raw_action, dtype=np.float64).ravel()
        if raw_action.shape != (NUM_JOINTS,):
            raise ValueError(f"expected action shape ({NUM_JOINTS},), got {raw_action.shape}")
        self._last_action = raw_action.copy()
        self._apply_action(raw_action)
        self._step_count += 1

        if self.task == "velocity" and self._step_count % COMMAND_RESAMPLE_STEPS == 0:
            self._sample_velocity_command()

        position = self.data.qpos[0:2]
        heading = self._heading()
        speed = float(np.linalg.norm(self.data.qvel[0:2]))
        if self.task == "sequence":
            self._stage_update = self._stage.update(
                float(position[0]), float(position[1]), heading, speed
            )

        if self._base_contact_force() > CONTACT_FORCE_THRESHOLD_N:
            self._fell = True
            self._episode_over = True
        elif self.task == "sequence" and self._stage.completed:
            # Isaac's `sequence_complete` termination ends the episode on success
            self._episode_over = True
        elif self._step_count >= self.episode_steps:
            self._episode_over = True

    # --- introspection --------------------------------------------------------
    @property
    def over(self) -> bool:
        return self._episode_over

    @property
    def fell(self) -> bool:
        return self._fell

    @property
    def completed(self) -> bool:
        return self.task == "sequence" and self._stage.completed

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def stage(self) -> int:
        return self._stage.stage

    @property
    def base_position(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def command(self) -> np.ndarray:
        if self.task == "velocity":
            return self._command.copy()
        assert self._stage_update is not None
        return np.array(self._stage_update.command)

    def observation(self) -> np.ndarray:
        return self._observation()


@dataclass(frozen=True)
class Sim2SimPolicy:
    """ONNX inference wrapper for the three S4 actor presets.

    - ``s0_flat``: obs 48 → 12 joints (velocity task / layered base policy)
    - ``single_layer``: obs 53 → 12 joints (end-to-end sequence policy)
    - ``layered``: obs 53 → S1(→ 2-dim command, clamped) → rebuilt 48-dim obs →
      S0 → 12 joints, mirroring ``layered_env.LayeredSequenceEnv``

    ``s0_path`` must be provided for ``layered``; the S0 observation rebuild
    follows ``layered_env._s0_observation``:
    ``[obs53[0:9], (cmd_vx, 0.0, cmd_wz), obs53[11:35], obs53[35:47]]``.
    """

    mode: str
    s0_path: pathlib.Path | str | None = None
    s1_path: pathlib.Path | str | None = None
    single_path: pathlib.Path | str | None = None

    def __post_init__(self) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        # frozen dataclass: internal session handles are set through object.__setattr__
        object.__setattr__(
            self,
            "_s0",
            ort.InferenceSession(str(self.s0_path), providers=providers) if self.s0_path else None,
        )
        object.__setattr__(
            self,
            "_s1",
            ort.InferenceSession(str(self.s1_path), providers=providers) if self.s1_path else None,
        )
        object.__setattr__(
            self,
            "_single",
            ort.InferenceSession(str(self.single_path), providers=providers) if self.single_path else None,
        )
        if self.mode == "layered" and (self._s0 is None or self._s1 is None):
            raise ValueError("layered mode requires both s0_path and s1_path")
        if self.mode == "single" and self._single is None:
            raise ValueError("single mode requires single_path")
        if self.mode == "s0" and self._s0 is None:
            raise ValueError("s0 mode requires s0_path")
        if self.mode not in ("s0", "single", "layered"):
            raise ValueError(f"unknown mode {self.mode!r}")

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)[None, :]
        if self.mode == "s0":
            return self._s0.run(None, {"observation": observation})[0].ravel()
        if self.mode == "single":
            return self._single.run(None, {"observation": observation})[0].ravel()
        # layered: S1 emits the (vx, wz) command; S0 executes it
        command = self._s1.run(None, {"observation": observation})[0].ravel()
        command = np.clip(command, -1.0, 1.0)
        command3 = np.array([command[0], 0.0, command[1]], dtype=np.float32)
        obs53 = observation[0]
        s0_obs = np.concatenate(
            [obs53[0:9], command3, obs53[11:35], obs53[35:47]]
        ).astype(np.float32)[None, :]
        return self._s0.run(None, {"observation": s0_obs})[0].ravel()
