"""Checkpoint → ONNX export for S4 Sim2Sim (zero-weight-change chain).

RSL-RL checkpoints (``logs/rsl_rl/**/model_*.pt``) store the PPO actor under
``model_state_dict`` with keys prefixed ``actor.``. This module rebuilds the
same ELU MLP layout used by ``locomotion.sequence.layered_env.S0Actor``
(``Linear(num_obs, 512) → ELU → Linear(512, 256) → ELU → Linear(256, 128) →
ELU → Linear(128, num_actions)``), loads the actor weights verbatim and
exports a static ONNX graph for the MuJoCo-side runtime (onnxruntime, so the
evaluation environment needs no torch).

Zero-change contract: the exported graph must reproduce the torch forward
pass of the loaded weights within float32 tolerance (asserted at export time
against a fixed seeded input), and both the checkpoint and the ONNX artifact
are hashed (SHA-256) into a sidecar manifest so an evaluation run can pin the
exact policy binary it executed. Weights are never re-initialised, rescaled
or re-ordered here; the only transformation is the key remap
``actor.N.* → net.N.*`` mirroring ``layered_env.load_s0_actor``.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import pathlib

import torch
from torch import nn

MANIFEST_SCHEMA_VERSION = "s4-policy-export-v1"
ACTOR_HIDDEN_DIMS = (512, 256, 128)
DEFAULT_OPSET = 17
PARITY_TOLERANCE = 1e-5
PARITY_INPUT_SEED = 0

#: Frozen (num_obs, num_actions) pairs for the three S4 actors.
#: - s0_flat:     M1 velocity policy, 48-dim blind proprioceptive obs → 12 joints
#:                (source of truth: locomotion/sequence/layered_env.py constants
#:                NUM_OBS_S0=48 / NUM_ACTIONS_S0=12).
#: - s1_layered:  upper-level sequence policy over the 53-dim sequence obs →
#:                2-dim closed-loop command (vx, wz).
#: - single_layer: end-to-end sequence policy, same 53-dim obs → 12 joints.
SUPPORTED_PRESETS: dict[str, tuple[int, int]] = {
    "s0_flat": (48, 12),
    "s1_layered": (53, 2),
    "single_layer": (53, 12),
}


def sha256_file(path: pathlib.Path) -> str:
    """SHA-256 hex digest of a file, streamed (matches repo manifest style)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class S4Actor(nn.Module):
    """Inference-only MLP with the RSL-RL actor layout (see module docstring)."""

    def __init__(self, num_obs: int, num_actions: int) -> None:
        super().__init__()
        hidden = ACTOR_HIDDEN_DIMS
        layers: list[nn.Module] = [nn.Linear(num_obs, hidden[0]), nn.ELU()]
        for in_dim, out_dim in zip(hidden, hidden[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
        layers.append(nn.Linear(hidden[-1], num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(obs)


def load_actor_weights(
    actor: S4Actor, checkpoint: pathlib.Path, device: str = "cpu"
) -> None:
    """Load actor weights from an RSL-RL checkpoint, mirroring
    ``layered_env.load_s0_actor`` (``actor.N.* → net.N.*``, critic/std dropped).

    Raises:
        ValueError: if the checkpoint contains no ``actor.*`` keys.
    """
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    actor_state = {
        f"net.{key.removeprefix('actor.')}": value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise ValueError(f"no actor weights in {checkpoint}")
    actor.load_state_dict(actor_state)
    actor.eval()


def export_actor_onnx(
    checkpoint: pathlib.Path,
    output_path: pathlib.Path,
    num_obs: int,
    num_actions: int,
    *,
    opset_version: int = DEFAULT_OPSET,
    device: str = "cpu",
    parity_tolerance: float = PARITY_TOLERANCE,
) -> dict[str, object]:
    """Export one actor checkpoint to ONNX and verify forward parity.

    Writes ``output_path`` (the ONNX graph) and ``output_path`` + ``.manifest.json``
    (hashes, parity stats and environment info per ``s4-policy-export-v1``).

    Raises:
        RuntimeError: if the ONNX runtime output deviates from torch beyond
            ``parity_tolerance`` (max abs diff over the seeded parity input).
    """
    import onnxruntime as ort

    checkpoint = pathlib.Path(checkpoint)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actor = S4Actor(num_obs, num_actions).to(device)
    load_actor_weights(actor, checkpoint, device=device)

    torch.manual_seed(PARITY_INPUT_SEED)
    sample = torch.randn(1, num_obs, device=device)
    with torch.inference_mode():
        reference = actor(sample)

    torch.onnx.export(
        actor,
        sample,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        opset_version=opset_version,
        dynamo=False,
    )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    (exported,) = session.run(None, {"observation": sample.detach().cpu().numpy()})
    max_abs_diff = float(
        max(
            abs(float(a) - float(b))
            for a, b in zip(reference.detach().cpu().numpy().ravel(), exported.ravel())
        )
    )
    if max_abs_diff > parity_tolerance:
        raise RuntimeError(
            f"ONNX forward parity failed: max_abs_diff={max_abs_diff:.3e} "
            f"> tolerance={parity_tolerance:.1e}"
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "onnx": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "num_obs": num_obs,
        "num_actions": num_actions,
        "opset_version": opset_version,
        "parity": {
            "input_seed": PARITY_INPUT_SEED,
            "max_abs_diff": max_abs_diff,
            "tolerance": parity_tolerance,
        },
        "torch_version": torch.__version__,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
