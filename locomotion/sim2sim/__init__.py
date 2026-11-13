"""S4 Sim2Sim: Isaac-trained Go2 policies transferred to MuJoCo (RQ4).

The frozen protocol lives in ``docs/EXPERIMENT_DESIGN.md`` (RQ4/S4) and
``locomotion/sim2sim/README.md``. This package adds, one module per concern:

- ``policy_export``: RSL-RL checkpoint → ONNX, zero weight change, hashed.
- ``go2_env``: MuJoCo rollout matching the Isaac observation/action/actuator
  contract (per-joint order remap, PD + DC-motor saturation, base-contact
  termination, 200 Hz physics / 50 Hz control).
- ``evaluate_sim2sim``: frozen-seed evaluation reusing ``locomotion.eval``
  metrics (single source of truth — no second metric implementation here).

Nothing in this package imports Isaac Lab; evaluation runs in a plain
MuJoCo + onnxruntime environment.
"""
