"""VLA chain: SO-101 MuJoCo env → scripted expert → LeRobotDataset v3 → SmolVLA.

Layout:
- ``env.py``    thin wrapper over the Hub env ``lerobot/so101-tabletop-mujoco``
                (camera naming, task string, success check) — imports lerobot
                lazily so Windows CI can test everything else.
- ``expert.py`` scripted demonstrator: a pure pick-and-place phase machine
                (Windows-testable) + the IK/motion primitive executor.
- ``record.py`` expert rollouts → LeRobotDataset v3 (create/add_frame/
                save_episode/finalize).
- ``train.py``  SmolVLA LoRA fine-tune launcher (lerobot-train).
- ``evaluate.py`` fine-tuned policy rollout + success rate + videos.

Heavy deps (lerobot, mujoco rendering) live in the WSL2 ``/root/vla-env``
venv; the pure-logic parts are tested on Windows CI.
"""
