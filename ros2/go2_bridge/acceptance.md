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
| Real ros:jazzy DDS closed loop | **ROS-R1** | **PENDING_TARGET_ENV** | `run_r1_acceptance.sh` (CI `ros2-live-r1` workflow; local Docker when available) |

## Environment limitation (recorded, per sibling-repo doctrine)

The development host is Windows; Jazzy has no Windows rclpy tier and Docker
Desktop availability on this host is unverified at packaging time. The
acceptance harness and CI gate exist; a real-DDS PASS must come from the
`ros2-live-r1` workflow (or a local Docker run of the same script). A blocked
local run is recorded here as `BLOCKED_TARGET_ENV` and never as ROS-R1.

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
