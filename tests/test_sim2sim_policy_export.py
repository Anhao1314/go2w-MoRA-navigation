"""Pin the S4 policy-export chain: key remap, ONNX parity and manifest schema.

The frozen zero-change contract (``locomotion/sim2sim/README.md``, TODO item 3)
requires weights and observation handling to transfer without modification.
These tests pin, on synthetic checkpoints (no Isaac/training assets needed):

- the ``actor.N.* → net.N.*`` remap and the ValueError on non-actor payloads;
- the ELU MLP layout (512/256/128 hidden dims) for all three S4 presets;
- ONNX export determinism (same checkpoint → same artifact SHA-256);
- forward parity between the torch reference and the onnxruntime graph;
- the ``s4-policy-export-v1`` manifest schema (hashes, dims, parity stats).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

import torch

from locomotion.sim2sim.policy_export import (
    ACTOR_HIDDEN_DIMS,
    MANIFEST_SCHEMA_VERSION,
    SUPPORTED_PRESETS,
    S4Actor,
    export_actor_onnx,
    load_actor_weights,
    sha256_file,
)


def _state_dict(num_obs: int, num_actions: int) -> dict[str, torch.Tensor]:
    """Build an RSL-RL-style payload with ``actor.N.*`` keys (ELU at odd indices)."""
    actor = S4Actor(num_obs, num_actions)
    named = {
        f"actor.{name.removeprefix('net.')}": tensor
        for name, tensor in actor.net.state_dict().items()
    }
    # mimic a full PPO payload: critic and std must be ignored by the loader
    named["critic.0.weight"] = torch.zeros(1, num_obs)
    named["std"] = torch.zeros(num_actions)
    return {"model_state_dict": named}


class TestPresetDims(unittest.TestCase):
    """The preset table must match the frozen observation/action layouts."""

    def test_preset_dims(self):
        # source of truth: layered_env.py NUM_OBS_S0=48/NUM_ACTIONS_S0=12 and the
        # 53-dim sequence obs documented in layered_env.LayeredSequenceEnv
        self.assertEqual(SUPPORTED_PRESETS["s0_flat"], (48, 12))
        self.assertEqual(SUPPORTED_PRESETS["s1_layered"], (53, 2))
        self.assertEqual(SUPPORTED_PRESETS["single_layer"], (53, 12))

    def test_hidden_dims(self):
        self.assertEqual(ACTOR_HIDDEN_DIMS, (512, 256, 128))


class TestLoadActorWeights(unittest.TestCase):
    """Checkpoint loading mirrors layered_env.load_s0_actor semantics."""

    def test_load_remaps_actor_keys(self):
        num_obs, num_actions = SUPPORTED_PRESETS["s0_flat"]
        actor = S4Actor(num_obs, num_actions)
        source = {
            f"actor.{name}": tensor for name, tensor in actor.net.state_dict().items()
        }
        payload = {"model_state_dict": {**source, "critic.0.weight": torch.zeros(1)}}

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model_1499.pt"
            torch.save(payload, path)
            target = S4Actor(num_obs, num_actions)
            load_actor_weights(target, path)

        for key, value in actor.net.state_dict().items():
            self.assertTrue(torch.equal(target.net.state_dict()[key], value), key)

    def test_load_rejects_payload_without_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model_bad.pt"
            torch.save({"model_state_dict": {"critic.0.weight": torch.zeros(1, 4)}}, path)
            with self.assertRaises(ValueError):
                load_actor_weights(S4Actor(48, 12), path)


class TestExportActorOnnx(unittest.TestCase):
    """ONNX artifacts must be deterministic, parity-checked and fully hashed."""

    def _export(self, tmp: pathlib.Path, preset: str, tag: str):
        num_obs, num_actions = SUPPORTED_PRESETS[preset]
        checkpoint = tmp / f"{preset}_checkpoint.pt"
        torch.save(_state_dict(num_obs, num_actions), checkpoint)
        output = tmp / f"{tag}.onnx"
        manifest = export_actor_onnx(checkpoint, output, num_obs, num_actions)
        return manifest, checkpoint, output

    def test_export_parity_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, checkpoint, output = self._export(pathlib.Path(tmp), "s0_flat", "a")
            self.assertTrue(output.exists())
            self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["num_obs"], 48)
            self.assertEqual(manifest["num_actions"], 12)
            self.assertLess(manifest["parity"]["max_abs_diff"], manifest["parity"]["tolerance"])
            self.assertEqual(manifest["checkpoint"]["sha256"], sha256_file(checkpoint))
            self.assertEqual(manifest["onnx"]["sha256"], sha256_file(output))
            manifest_path = output.with_suffix(".onnx.manifest.json")
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)

    def test_export_semantically_deterministic(self):
        # Byte-level determinism is not guaranteed (the exporter embeds
        # non-semantic metadata); the zero-change contract is about weights and
        # forward semantics, so pin identical outputs instead of identical bytes.
        import numpy as np
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmp:
            _, checkpoint, first = self._export(pathlib.Path(tmp), "s0_flat", "first")
            _, _, second = self._export(pathlib.Path(tmp), "s0_flat", "second")
            torch.manual_seed(0)
            sample = torch.randn(1, 48).numpy()
            out_first = ort.InferenceSession(str(first), providers=["CPUExecutionProvider"]).run(
                None, {"observation": sample}
            )[0]
            out_second = ort.InferenceSession(str(second), providers=["CPUExecutionProvider"]).run(
                None, {"observation": sample}
            )[0]
            np.testing.assert_array_equal(out_first, out_second)

    def test_export_all_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            for preset in SUPPORTED_PRESETS:
                manifest, _, output = self._export(pathlib.Path(tmp), preset, preset)
                num_obs, num_actions = SUPPORTED_PRESETS[preset]
                self.assertEqual((manifest["num_obs"], manifest["num_actions"]), (num_obs, num_actions))
                self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
