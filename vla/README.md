# vla/ — SO-101 → LeRobotDataset v3 → SmolVLA LoRA（自建全链路）

**从仿真到 VLA 的完整数据管道，无遥操硬件依赖。** 官方 [`so101-tabletop-mujoco`](https://huggingface.co/lerobot/so101-tabletop-mujoco) 环境只提供任务，不提供示范——脚本专家（状态机 + IK）、数据落盘、微调启动、评测全部自建。

[返回首页](../README.md) · [架构总规划](../docs/CPP_ROS2_ARCHITECTURE.md) · [S4 Sim2Sim](../locomotion/sim2sim/README.md)

## 管线

```text
lerobot/so101-tabletop-mujoco (SO-101, front+wrist 相机, 抓方块入盒)
        │  ground-truth geometry (cube/box/EE, 仅供专家使用)
        ▼
vla/expert.py  PickPlacePhaseMachine (纯逻辑状态机, 9 阶段, 失败重试)
        │      EndEffectorIK (阻尼最小二乘, MJCF 原生)
        ▼
vla/record.py  LeRobotDataset v3 落盘 (add_frame + save_episode + finalize)
        ▼      50-100 episodes, 4-5 个方块位置随机化, camera1/camera2 480x640
vla/train.py   SmolVLA LoRA r=64 (lerobot-train, 16GB 卡实证可行)
        ▼
vla/evaluate.py  仿真评测: 成功率 (20+ episodes) + 专家 vs VLA 对比视频
```

## 平台与依赖

- **WSL2 `G1-RL-Ubuntu`**（Ubuntu 24.04.5, Python 3.12.3），独立 venv `/root/vla-env`
- `lerobot[training,smolvla]`（v0.7.x, torch cu13 轮子, 驱动 621.01 满足）
- 相机渲染：`MUJOCO_GL=egl`（mesa/llvmpipe 软渲染，与 CI 同方案）
- Windows 主机只跑纯逻辑测试（状态机/IK，无需 lerobot）

## 模块

| 文件 | 职责 | 测试 |
| --- | --- | --- |
| `expert.py` | 9 阶段抓放状态机 + DLS IK（纯逻辑） | Windows CI（`tests/test_vla_expert.py`） |
| `env.py` | EnvHub 环境封装（相机命名/几何提取，懒加载） | WSL 冒烟（`scripts/vla_smoke.py`） |
| `record.py` | 专家 rollout → LeRobotDataset v3 | WSL 集成 |
| `train.py` | SmolVLA LoRA 启动器 | WSL 训练 |
| `evaluate.py` | 成功率评测 + 对比视频 | WSL 集成 |

## 诚实边界

- 仿真内数据 + 仿真内评测；无真机声明
- 专家使用 ground-truth 几何（特权信息）——VLA 只看相机和关节，特权不进数据集
- 训练配置与结果以 `logs/vla/` 证据束为准（复用仓库 run_config + SHA-256 约定）
