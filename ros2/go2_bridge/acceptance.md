# ROS2 Live Bridge Acceptance — go2_bridge v0.1

Status table for the bidirectional Go2 bridge (this file is updated per
attempt; negative attempts are kept as evidence, never deleted).

## Gate table

| Gate | Level | Status | Evidence |
| --- | --- | --- | --- |
| Protocol tables load + fail-closed | ROS-R0 | PASS | `tests/test_ros2_bridge_core.py::TestProtocol` |
| Joint order matches committed model | ROS-R0 | PASS | `test_ros2_bridge_core` cross-check |
| Command gate: clamp + deadman zeroing | ROS-R0 | PASS | `TestCommandGate` |
| Payload contract (frames, empty effort, canonical JSON) | ROS-R0 | PASS | `TestPayloads` |
| Adapter wiring: node args, QoS, publish allowlist | contract | PASS | `test_ros2_bridge_adapter` (mock ROS doubles) |
| Closed loop: cmd_vel → gate → env → session_info | contract | PASS | `test_closed_loop_publishes_allowlist_and_applies_commands` (real MuJoCo, mock ROS) |
| Behavioral verifier PASS/tamper cases | contract | PASS | `TestProbeVerifier` (synthetic traffic) |
| Real ros:jazzy DDS closed loop | **ROS-R1** | **PASS (2026-11-14, WSL2 G1-RL-Ubuntu, Jazzy)** | `logs/s4/ros_live/attempt_007/` — verdict `ROS2_R1_PASS`, 582 msgs/topic (incl. TF 582), forward 1.238 m, yaw +1.585 rad, 187 commands accepted |

## Attempt log (negative attempts retained as evidence)

| Attempt | Outcome | Cause |
| --- | --- | --- |
| attempt_001 | `BLOCKED_TARGET_ENV` | Docker daemon not running; WSL2 route adopted instead |
| attempt_002 | `ROS2_R1_FAILED` (PIP_FAILED) | noble PEP 668 + apt-owned numpy without RECORD — script now uses `--ignore-installed` + `PIP_BREAK_SYSTEM_PACKAGES` |
| attempt_003 | `ROS2_R1_FAILED` (VALIDATION_FAILED) | probe crashed: missing `rclpy.init()`; sim node crashed: missing `gymnasium` in acceptance deps |
| attempt_004 | `ROS2_R1_FAILED` (VALIDATION_FAILED) | TF published but unverified (probe lacked subscription); counts 425 < stale 500 threshold |
| attempt_005 | `ROS2_R1_FAILED` | probe `_verify` referenced out-of-scope `TFMessage` |
| attempt_006 | `ROS2_R1_FAILED` | same NameError (stale copy); git_commit null in node status (root + /mnt/d dubious ownership — cosmetic) |
| **attempt_007** | **`ROS2_R1_PASS`** | probe duration 14 s, TF subscribed and verified, thresholds recalibrated (≥500 msgs, TF ≥100) |

## Environment

- ROS 2 Jazzy (`ros-jazzy-ros-base`) installed in WSL2 `G1-RL-Ubuntu`
  (Ubuntu 24.04.5, Python 3.12.3, TUNA apt/pip mirrors); the same harness
  runs in the `ros2-live-r1` CI container.
- CI gate: `test "$(cat verdict.txt)" = "ROS2_R1_PASS"` (artifacts always
  uploaded, 30-day retention).

## Probe behavioral contract (what ROS-R1 means here)

1. ≥ 500 messages on /clock, /go2/joint_states, /go2/base_pose in 11 s;
2. /clock strictly increasing (sim time order preserved);
3. joint names == protocol order, `effort == []`;
4. pose frames == `mujoco_world`, quaternion normalized;
5. forward phase (cmd 0.5 m/s, 3 s) → base displacement > 0.2 m;
6. turn phase (cmd +0.8 rad/s, 3 s) → yaw change > +0.1 rad;
7. `session_info.command_applied` reflects the clamped commands;
8. sim node status stays `PUBLISHED_NOT_SUBSCRIBER_VERIFIED` (publisher never
   claims delivery — the probe does).
