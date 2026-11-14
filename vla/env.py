"""Thin wrapper over the Hub env ``lerobot/so101-tabletop-mujoco``.

Imports lerobot lazily (function-local) so the pure-logic parts of ``vla``
stay testable on hosts without the LeRobot stack.

Two access levels, deliberately:

- ``make_so101_env()`` — the OFFICIAL EnvHub path (LocalVectorEnv), used when
  a trained VLA rolls out (the policy consumes only public observations; the
  sealed wrapper is fine and keeps the serving semantics).
- ``make_so101_raw_env()`` — direct instantiation of the snapshot's
  ``SO101Env`` for **scripted-expert data generation**, which needs
  ground-truth geometry (cube/box positions from ``_World``).  The
  LocalVectorEnv deliberately does not expose ``_world``; rather than fight
  the seal, the raw path instantiates the same official class so all task
  logic (reward, success, seeded randomization) stays authoritative.

Verified against the downloaded EnvHub source (``so101_env.py``, 411 lines):
four cubes (``red/green/blue/yellow_cube``), box body ``box``, actions are
absolute joint targets at 30 fps — 5 arm joints in DEGREES + gripper 0-100
(``_World.to_ctrl`` converts to actuator radians), reward = fraction of cubes
in the box, success = all four, episode cap 3600 steps.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

TASK_INSTRUCTION = "pick up the cubes and put them in the green box"
ENV_HUB_PATH = "lerobot/so101-tabletop-mujoco"

CUBE_BODIES: tuple[str, ...] = ("red_cube", "green_cube", "blue_cube", "yellow_cube")
BOX_BODY = "box"
GRIPPER_JOINT = "gripper"  # 6th joint: opening 0-100 in action space
ARM_JOINTS = 5  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
STEP_DT_S = 1.0 / 30.0  # the SO-101 env steps at 30 fps (verified in its source)
CARTESIAN_SPEED_MPS = 0.08  # servo reference speed: smooth tracking, no waypoint teleports


class EnvNotAvailable(RuntimeError):
    """lerobot / EnvHub env unavailable on this host."""


def _env_module():
    """Import the EnvHub snapshot's ``env.py`` (official SO101Env + DEFAULTS)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise EnvNotAvailable(f"huggingface_hub unavailable: {error}")
    path = hf_hub_download(repo_id=ENV_HUB_PATH, filename="env.py")
    spec = importlib.util.spec_from_file_location("so101_tabletop_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_so101_raw_env(**overrides):
    """Direct ``SO101Env`` instance (ground truth accessible for the expert)."""
    module = _env_module()
    config = {**module.DEFAULTS, **overrides}
    return module.SO101Env(**config), module


def make_so101_env(**overrides):
    """Official EnvHub path (sealed LocalVectorEnv) — for VLA policy rollouts."""
    try:
        from lerobot.envs.configs import HubEnvConfig
        from lerobot.envs.factory import make_env
    except ImportError as error:
        raise EnvNotAvailable(f"lerobot stack unavailable: {error}")

    config = HubEnvConfig(
        hub_path=ENV_HUB_PATH,
        trust_remote_code=True,
        **overrides,
    )
    envs = make_env(config)
    while isinstance(envs, dict):  # v0.7 factory nests: {group: {index: env}}
        envs = next(iter(envs.values()))
    return envs


def camera_frames_from(obs) -> dict:
    """Normalized {camera1, camera2} RGB arrays from a plain obs dict."""
    pixels = obs.get("pixels", obs)
    mapping = {
        "front": "camera1",
        "wrist": "camera2",
    }
    frames = {}
    for key, name in mapping.items():
        frame = pixels.get(key)
        if frame is None:
            raise RuntimeError(f"env observation missing front/wrist camera {key!r}")
        frames[name] = frame
    return frames


def default_actor_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(repo_root) / "logs" / "vla" / "actors"


class SO101TabletopSession:
    """Single-env session for scripted-expert data generation.

    Ground truth comes from the official ``_World`` (reached through the
    ``SO101Env`` instance we own); the IK runs on a **shadow** ``MjData`` so
    planning never mutates the live physics state; actions are converted to
    the env's units (arm joints in degrees, gripper 0-100) and clamped to the
    official ctrlrange.  Success is always the official counter
    (``_World.cubes_in_box()``) — never a second implementation.
    """

    def __init__(self, env) -> None:
        import mujoco

        self.env = env
        self._mujoco = mujoco
        self.unwrapped = env.unwrapped
        self._world = None  # built lazily by the env on first reset/render
        self._body_ids: dict[str, int] = {}
        self._planning_data = None
        self._ik = None
        self._contact_calibrated = False
        self._ee_site = "gripperframe"  # official EE frame in the SO-101 MJCF
        self._tip_offset_z = 0.0  # calibrated: gripperframe → fingertip, world +z
        self._jaw_body_names = ("gripper", "moving_jaw_so101_v1")
        self._ref_ee = None  # rate-limited Cartesian servo reference (twitch fix)

    def _ensure_world(self):
        if self._world is None:
            self._world = self.unwrapped._ensure_world()
            self._planning_data = self._mujoco.MjData(self._world.model)
            self._body_ids = {
                name: self._body_id(name)
                for name in (*CUBE_BODIES, BOX_BODY)
            }
            self._calibrate_fingertip_offset()
        return self._world

    def _calibrate_fingertip_offset(self) -> None:
        """Exact fingertip offset: transform every gripper/jaw geom's AABB
        corners into world frame and take the lowest z, at the SAME pose as
        the site reading — the difference is the jaw reach along world z.

        (A sphere-approximation of the capsule jaws overestimated the reach by
        ~2 cm and made the expert press the jaws into the table beside the
        cube; the corner transform is exact for the compiled convex hulls.)
        """
        import mujoco

        import numpy as np

        model = self._world.model
        data = self._world.data
        mujoco.mj_forward(model, data)  # populate geom_xpos/site_xpos (zeros before this)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site)
        if site_id < 0:
            raise RuntimeError(f"site {self._ee_site!r} not found in the SO-101 model")
        jaw_ids = [
            self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._jaw_body_names
        ]
        jaw_ids = [b for b in jaw_ids if b >= 0]
        lowest = float("inf")
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] not in jaw_ids:
                continue
            center = np.asarray(model.geom_aabb[geom, :3], dtype=float)
            half = np.asarray(model.geom_aabb[geom, 3:], dtype=float)
            xpos = np.asarray(data.geom_xpos[geom], dtype=float)
            xmat = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner_local = center + half * np.array([sx, sy, sz])
                        corner_world = xpos + xmat @ corner_local
                        lowest = min(lowest, float(corner_world[2]))
        if lowest == float("inf"):
            raise RuntimeError("no gripper/jaw geoms found for fingertip calibration")
        self._tip_offset_z = float(data.site_xpos[site_id][2]) - lowest

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(self._world.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"body {name!r} not found in the SO-101 model")
        return body_id

    def _ee_pos_fingertip(self) -> tuple[float, float, float]:
        """EE position in FINGERTIP space (gripperframe shifted down by the
        calibrated jaw-reach offset)."""
        import mujoco

        site_id = self._mujoco.mj_name2id(
            self._world.model, self._mujoco.mjtObj.mjOBJ_SITE, self._ee_site
        )
        pos = self._world.data.site_xpos[site_id]
        return (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]) - self._tip_offset_z,
        )

    def _body_pos(self, name: str) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._world.data.xpos[self._body_ids[name]])

    def _gripper_open(self) -> bool:
        """Opening ratio from the gripper qpos vs its ctrlrange (radians)."""
        qadr = int(self._world.qadr[5])
        lo, hi = float(self._world.lo[5]), float(self._world.hi[5])
        if hi <= lo:
            return False
        ratio = (float(self._world.data.qpos[qadr]) - lo) / (hi - lo)
        return ratio > 0.5

    def _cube_grasped(self, cube_pos, ee_pos, gripper_open: bool) -> bool:
        """Proximity heuristic: cube rides with the closed gripper."""
        if gripper_open:
            return False
        return sum((cube_pos[i] - ee_pos[i]) ** 2 for i in range(3)) ** 0.5 < 0.04

    # --- gym-facing API (plain, unbatched) -------------------------------------
    def reset(self, seed: int = 0) -> dict:
        self._ensure_world()
        obs, info = self.env.reset(seed=seed)
        self._last_info = info
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def cubes_in_box(self) -> int:
        return int(self._world.cubes_in_box())

    def total_cubes(self) -> int:
        return len(CUBE_BODIES)

    # --- expert-facing API -------------------------------------------------------
    def expert_observation(self, obs, cube_name: str):
        """Geometry snapshot for one target cube (others are ignored)."""
        from vla.expert import ExpertObservation

        ee = self._ee_pos_fingertip()
        cube = self._body_pos(cube_name)
        box = self._body_pos(BOX_BODY)
        gripper_open = self._gripper_open()
        qadr = int(self._world.qadr[5])
        lo, hi = float(self._world.lo[5]), float(self._world.hi[5])
        ratio = (float(self._world.data.qpos[qadr]) - lo) / (hi - lo) if hi > lo else 0.5
        return ExpertObservation(
            ee_pos=ee,
            cube_pos=cube,
            box_pos=box,
            gripper_open=gripper_open,
            cube_grasped=self._cube_grasped(cube, ee, gripper_open),
            gripper_ratio=ratio,
        )

    def expert_action(self, command):
        """Waypoint + gripper bool → SO-101 action (degrees + 0-100).

        The command waypoint is in FINGERTIP space; a rate-limited Cartesian
        servo (``_ref_ee``) slides toward it at CARTESIAN_SPEED_MPS so the arm
        tracks a continuous reference instead of teleporting between the
        phase machine's discrete waypoints (the twitch fix).  The IK target
        for the ``gripperframe`` origin is the reference shifted UP by the
        calibrated jaw-reach offset.
        """
        import numpy as np

        if self._ik is None:
            from vla.expert import EndEffectorIK

            self._ik = EndEffectorIK(
                self._world.model, self._planning_data,
                site_name=self._ee_site,
                joint_ids=tuple(self._arm_joint_ids()),
            )
        if self._ref_ee is None:  # start servoing from where the arm actually is
            self._ref_ee = np.asarray(self._ee_pos_fingertip(), dtype=float)

        fingertip_target = np.asarray(command.target_ee_pos, dtype=float)
        delta = fingertip_target - self._ref_ee
        max_step = CARTESIAN_SPEED_MPS * STEP_DT_S
        dist = float(np.linalg.norm(delta))
        if dist > max_step:
            self._ref_ee = self._ref_ee + delta / dist * max_step
        else:
            self._ref_ee = fingertip_target

        # shadow data starts from the live state; IK never touches physics
        self._planning_data.qpos[:] = self._world.data.qpos
        target = (
            float(self._ref_ee[0]),
            float(self._ref_ee[1]),
            float(self._ref_ee[2]) + self._tip_offset_z,
        )
        targets_rad = self._ik.solve(target, target_z_axis=(0.0, 0.0, 1.0))

        action = []
        for k in range(ARM_JOINTS):
            qadr = int(self._world.qadr[k])
            deg = targets_rad[k] * 180.0 / math.pi
            lo_deg = math.degrees(float(self._world.lo[k]))
            hi_deg = math.degrees(float(self._world.hi[k]))
            action.append(max(lo_deg, min(hi_deg, deg)))
        action.append(100.0 if command.gripper_open else 0.0)
        return np.asarray(action, dtype=np.float32)

    def _arm_joint_ids(self):
        """The first five joints are the arm; the gripper is channel 5."""
        hinge_ids = [
            j for j in range(self._world.model.njnt)
            if self._world.model.jnt_type[j] == self._mujoco.mjtJoint.mjJNT_HINGE
        ]
        return hinge_ids[:ARM_JOINTS]

    def front_frame_rgb(self, obs):
        pixels = obs.get("pixels", obs)
        frame = pixels.get("front")
        if frame is None:
            raise RuntimeError("env observation missing front camera")
        return frame
