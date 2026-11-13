# ros2/go2_bridge — Go2 ROS2 online bridge (sim-only closed loop)

**One policy. Two engines. Real DDS in between.** This package wires the
S4-validated Go2 legged sim2sim policy into ROS 2 Jazzy as a *live* 50 Hz
closed loop: standard-topic state out, `/go2/cmd_vel` in, policy unchanged.

[返回首页](../../README.md) · [S4 Sim2Sim](../../locomotion/sim2sim/README.md) · [ROS2 架构总规划](../../docs/CPP_ROS2_ARCHITECTURE.md)

## Scope and acceptance levels

`ROS-R0` verifies the rclpy-free core (payloads, command gate, protocol
tables) and the adapter wiring with in-memory ROS doubles — no ROS 2 needed
(`tests/test_ros2_bridge_*.py`, CI on plain Python).

`ROS-R1` requires a **real** ros:jazzy DDS round trip: the sim node (real
MuJoCo + the s0_flat actor) runs against an independent behavioral probe that
publishes a scripted `/go2/cmd_vel` sequence and verifies the loop through the
received topics — forward commands move the base, turn commands yaw it, and
`session_info` reports the clamped applied commands. In-memory doubles never
count as ROS-R1. Run it locally (Docker) or in CI with:

```bash
bash ros2/go2_bridge/run_r1_acceptance.sh "$PWD/logs/s4/ros_live/attempt_001"
# verdict: logs/s4/ros_live/attempt_001/verdict.txt == ROS2_R1_PASS
```

## Topics (v0.1, all standard messages, QoS KEEP_LAST/32/RELIABLE/VOLATILE)

| Topic | Type | Dir | Freq | Notes |
| --- | --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/Clock` | pub | 50 Hz | sim time; consumers opt into `use_sim_time` |
| `/go2/joint_states` | `sensor_msgs/JointState` | pub | 50 Hz | 12 joints, MJCF per-leg order; `effort=[]` (not measured) |
| `/go2/base_pose` | `geometry_msgs/PoseStamped` | pub | 50 Hz | `frame_id=mujoco_world`, quaternion xyzw |
| `/tf` | `tf2_msgs/TFMessage` | pub | 50 Hz | sim ground truth `mujoco_world→base`; skipped if `tf2_ros` absent |
| `/go2/session_info` | `std_msgs/String` | pub | ~1 Hz | canonical JSON: git/actor provenance, reset/fall counts, **applied command** |
| `/go2/cmd_vel` | `geometry_msgs/Twist` | **sub** | external | clamped to ±1; **deadman: stale 0.5 s → zero** |

`protocol.json` is the machine-checked contract (topic allowlist, frames,
forbidden claims); `core.load_protocol` fail-closes on drift.

## Relationship to the P1 ROS2 plan

`docs/CPP_ROS2_ARCHITECTURE.md` plans a Go2W C++ 500 Hz stack
(`/go2w/...`, custom msgs) that remains unimplemented. This bridge is a
separate, smaller vertical slice for the legged Go2 S4 policy (`/go2/...`,
standard msgs, Python rclpy) and claims nothing against that plan.

## Honest boundaries

- **Simulation-only closed loop**: no hardware, no EKF/odometry, no map/odom
  frames; `/tf` carries simulation ground truth, not localization.
- Policy weights are the hash-pinned s0_flat actor
  (`fixtures/s0_flat.onnx` — an exported inference artifact, not a training
  checkpoint; provenance in `fixtures/s0_flat.onnx.manifest.json`).
- ROS 2 runs in Docker/CI only (no Windows rclpy tier; see
  `docs/WINDOWS_NATIVE.md`). Local Windows runs are contract tests (ROS-R0)
  by construction; a real-DDS run without Docker is recorded as
  `BLOCKED_TARGET_ENV` negative evidence, not as ROS-R1.

## Layout

| Path | Role |
| --- | --- |
| `core.py` | rclpy-free payloads, command gate, protocol loader |
| `sim_node.py` | lazy-rclpy 50 Hz sim node (env + policy + publishers + cmd_vel) |
| `probe.py` | independent behavioral verifier (publisher + subscriptions) |
| `run_r1_acceptance.sh` | real-DDS acceptance harness (container/local Docker) |
| `fixtures/s0_flat.onnx` | hash-pinned actor for acceptance (regenerable) |
| `protocol.json` | machine-checked topic/frame/claim contract |
