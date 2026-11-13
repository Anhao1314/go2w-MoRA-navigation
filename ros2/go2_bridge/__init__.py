"""Go2 ROS2 online bridge (sim-only closed loop, Jazzy-targeted).

Modules:
- ``core``: rclpy-free payloads, protocol, command gate (testable anywhere)
- ``sim_node``: lazy-rclpy node stepping the MuJoCo env + policy at 50 Hz
- ``probe``: independent publisher/subscriber for the real-DDS acceptance

Nothing here imports rclpy at module import time; ROS imports are function-local
so the package stays usable (and fully tested) on hosts without ROS 2.
"""
