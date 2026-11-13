"""Thin wrapper over the Hub env ``lerobot/so101-tabletop-mujoco``.

Imports lerobot lazily (function-local) so the pure-logic parts of ``vla``
stay testable on hosts without the LeRobot stack.  The wrapper normalizes:

- camera frames → ``camera1`` (front) / ``camera2`` (wrist), RGB uint8;
- ground-truth geometry for the scripted expert (cube / box / end-effector
  positions read from the MuJoCo data — privileged, the VLA never sees these);
- success detection (reward-based, from the env itself).

The exact EnvHub observation/action keys are verified at runtime in V1c
(``scripts/vla_smoke.py``); anything that drifts fails loudly here rather
than silently corrupting a dataset.
"""

from __future__ import annotations

import pathlib

TASK_INSTRUCTION = "pick up the cube and put it in the green box"
ENV_HUB_PATH = "lerobot/so101-tabletop-mujoco"


class EnvNotAvailable(RuntimeError):
    """lerobot / EnvHub env unavailable on this host."""


def make_so101_env(**overrides):
    """Create the SO-101 tabletop MuJoCo env (lazy lerobot import)."""
    try:
        from lerobot.envs import utils as env_utils  # EnvHub make_env helpers
        from lerobot.envs.config import HubEnvConfig
    except ImportError as error:
        raise EnvNotAvailable(f"lerobot stack unavailable: {error}")

    config = HubEnvConfig(
        hub_path=ENV_HUB_PATH,
        trust_remote_code=True,
        **overrides,
    )
    return env_utils.make_env(config)


class SO101TabletopSession:
    """Stateful session: reset/step + expert observation extraction.

    Ground-truth positions are read from the underlying MuJoCo data via the
    body/site names pinned in the env's XML; the exact names are resolved
    lazily and fail loudly if the upstream env changes them.
    """

    CUBE_BODY = "cube_0"   # verified at runtime; see scripts/vla_smoke.py
    BOX_BODY = "box"
    EE_SITE = "gripper_site"

    def __init__(self, env) -> None:
        self.env = env
        self._mujoco_data = None
        self._resolve_bodies()

    def _resolve_bodies(self) -> None:
        import mujoco

        unwrapped = self.env.unwrapped
        model = getattr(unwrapped, "model", None) or unwrapped.unwrapped.model
        data = getattr(unwrapped, "data", None) or unwrapped.unwrapped.data
        for name in (self.CUBE_BODY, self.BOX_BODY):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                raise RuntimeError(
                    f"body {name!r} not found in the SO-101 env model — the "
                    "upstream env changed; update the wrapper"
                )
        self._mujoco_model = model
        self._mujoco_data = data

    def _body_pos(self, name: str) -> tuple[float, float, float]:
        import mujoco

        body_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, name)
        return tuple(float(v) for v in self._mujoco_data.xpos[body_id])

    def observation_for_expert(self, obs) -> "ExpertObservation":  # noqa: F821
        from vla.expert import ExpertObservation

        return ExpertObservation(
            ee_pos=self._body_pos("gripper"),
            cube_pos=self._body_pos(self.CUBE_BODY),
            box_pos=self._body_pos(self.BOX_BODY),
            gripper_open=bool(obs.get("gripper_open", True)),
            cube_grasped=False,  # resolved from sim state by the caller
        )

    def camera_frames(self, obs) -> dict[str, bytes]:
        """Return {camera1, camera2} raw RGB frames from the env observation."""
        frames: dict[str, bytes] = {}
        mapping = {
            "observation.images.front": "camera1",
            "observation.images.wrist": "camera2",
        }
        for key, name in mapping.items():
            frame = obs.get(key)
            if frame is None:
                raise RuntimeError(f"env observation missing {key!r}")
            frames[name] = frame.tobytes() if hasattr(frame, "tobytes") else frame
        return frames


def default_actor_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(repo_root) / "logs" / "vla" / "actors"
