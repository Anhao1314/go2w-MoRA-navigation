# models/go2 — Unitree Go2 (legged, 12-DoF) MuJoCo model

S4 Sim2Sim 所需的四足版 Go2 官方 MuJoCo 模型（`models/go2w` 是轮足版 Go2-W，腿关节同名但含 4 个 wheel joint，不能直接用于 legged 策略迁移）。

## 来源

- 上游：[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `unitree_go2/`
- 上游固定提交：`370c43965323c0944b433b202aac5058b5f907a9`
- 许可：BSD 3-Clause（Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD "Unitree Robotics"），全文见 [LICENSE](LICENSE)

## 文件清单

| 文件 | 状态 |
| --- | --- |
| `go2.xml` | 上游原样，未修改（joint 名 `{FL,FR,RL,RR}_{hip,thigh,calf}_joint` 与 Isaac USD 侧 `locomotion/guardrail/go2_model.json` 完全一致） |
| `assets/*.obj` | 上游原样，未修改 |
| `LICENSE` | 上游原样 |
| `UPSTREAM_README.md` | 上游 README（改名避免与项目文档混淆） |
| `go2_flat_scene.xml` | **项目自建**：平面场景 + `track`/`overview` 相机，仿照 `models/go2w/go2w_factory_scene.xml` |

## 与 Isaac 侧的对应关系（S4 冻结约束）

- 根 body：`base`（自由关节）；12 个腿关节，无 wheel
- 上游默认 `<option cone="elliptic" impratio="100"/>`，timestep 保持 MuJoCo 默认；S4 在加载后由代码设为 0.005 s（对应 Isaac `sim.dt`，decimation 4 → 50 Hz 控制，见 `locomotion/sim2sim/go2_env.py`）
- 上游执行器为 torque `<motor>`（ctrlrange hip/thigh ±23.7、knee ±45.43）；S4 在代码内按 Isaac DCMotor 契约计算 PD 力矩并统一裁剪到 ±23.5 N·m（`effort_limit`，见 sim2sim README 的执行器决策文档）
- 上游被动参数（joint damping 2.0、armature 0.01、frictionloss 0.2、geom friction 0.6/足 0.8）保留为 S4 基线，与 Isaac PhysX 侧差异归入归因矩阵（actuator/contact 因素）
