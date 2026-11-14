#!/usr/bin/env bash
# Real-DDS acceptance for the Go2 ROS2 online bridge (ros:jazzy container).
#
# Runs the sim node (real MuJoCo + the s0_flat actor fixture) against the
# independent behavioral probe: the probe publishes a scripted /go2/cmd_vel
# sequence and verifies the closed loop through the received topics.  A PASS
# receipt plus cross-validation flips verdict.txt to ROS2_R1_PASS; every other
# outcome keeps the pessimistic verdict and still leaves full artifacts.
#
# Usage: bash ros2/go2_bridge/run_r1_acceptance.sh <output_dir>
set -uo pipefail

output_dir="${1:?usage: run_r1_acceptance.sh <output_dir>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$output_dir"; chmod 700 "$output_dir"

# pessimistic verdict first; artifacts must survive every failure path
printf '%s\n' 'ROS2_R1_FAILED' > "$output_dir/verdict.txt"
blocker() { printf '%s\n' "$1" > "$output_dir/blocker.txt"; exit 0; }
on_exit() {
  ( cd "$output_dir" && find . -type f ! -name file_hashes.sha256 -print0 \
      | xargs -0 sha256sum > file_hashes.sha256 ) || true
}
trap on_exit EXIT

{ git -C "$repo_root" rev-parse HEAD; \
  printf 'ROS_DISTRO=%s\nRMW_IMPLEMENTATION=%s\nROS_DOMAIN_ID=%s\n' \
    "${ROS_DISTRO:-unset}" "${RMW_IMPLEMENTATION:-default}" "${ROS_DOMAIN_ID:-unset}"; \
  date -u +%Y-%m-%dT%H:%M:%SZ; sha256sum "$repo_root/ros2/go2_bridge/fixtures/s0_flat.onnx"; \
} > "$output_dir/environment.log" 2>&1

# ROS setup (set -u guard around the sourcing, as in the g1-repo harness)
if [ ! -f /opt/ros/jazzy/setup.bash ]; then blocker "ROS_JAZZY_SETUP_MISSING"; fi
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash || blocker "ROS_SETUP_FAILED"
set -u

# sim dependencies (Linux wheels; the fixture actor ships with the repo).
# Ubuntu 24.04 (noble) notes: PEP 668 externally-managed env + an apt-owned
# numpy without a RECORD file — hence the two flags below (harmless on CI,
# where neither applies).
export PIP_BREAK_SYSTEM_PACKAGES=1
# Jazzy docker images ship without python3-pip (removed from the base image)
apt-get update -qq > /dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip > /dev/null 2>&1 \
  || blocker "PIP3_INSTALL_FAILED"
pip3 install --quiet --ignore-installed "mujoco==3.11.0" "onnxruntime" "numpy>=2.2.6,<3" "gymnasium" \
  > "$output_dir/pip.log" 2>&1 || blocker "PIP_FAILED"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY' > "$output_dir/ros_imports.log" 2>&1 || blocker "ROS_IMPORTS_FAILED"
import numpy, rclpy, mujoco, onnxruntime
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import importlib.util
print("imports-ok", "tf2_ros" if importlib.util.find_spec("tf2_ros") else "tf2_ros-absent")
PY

# behavioral probe first, sim node second (publisher waits for subscribers)
timeout 75s python3 "$repo_root/ros2/go2_bridge/probe.py" \
  --duration 14 --out "$output_dir" > "$output_dir/probe_stdout.log" 2>&1 &
probe_pid=$!
sleep 2
timeout 55s python3 "$repo_root/ros2/go2_bridge/sim_node.py" \
  --actor "$repo_root/ros2/go2_bridge/fixtures/s0_flat.onnx" \
  --steps 750 --out "$output_dir" > "$output_dir/node_stdout.log" 2>&1 &
node_pid=$!
wait "$node_pid"; node_exit=$?
wait "$probe_pid"; probe_exit=$?
printf 'node_exit=%s\nprobe_exit=%s\n' "$node_exit" "$probe_exit" > "$output_dir/process_status.txt"

python3 - "$output_dir" <<'PY' || blocker "VALIDATION_FAILED"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
probe = json.loads((out / "probe_receipt.json").read_text(encoding="utf-8"))
node = json.loads((out / "node_status.json").read_text(encoding="utf-8"))
assert probe["status"] == "ROS2_ROUNDTRIP_PASS", probe
assert node["ros2"]["status"] == "PUBLISHED_NOT_SUBSCRIBER_VERIFIED", node
assert probe["topic_counts"]["clock"] >= 500, probe
assert probe["topic_counts"]["joint_states"] >= 500, probe
assert probe["topic_counts"]["base_pose"] >= 500, probe
assert probe["tf_verified"] is True and probe["topic_counts"]["tf"] >= 100, probe
assert probe["forward_displacement_m"] > 0.2, probe
assert probe["turn_yaw_change_rad"] > 0.1, probe
assert probe["commands_accepted"] > 0, probe
print("REAL_ROS2_CLOSED_LOOP_PASS")
PY

printf '%s\n' 'ROS2_R1_PASS' > "$output_dir/verdict.txt"
exit 0
