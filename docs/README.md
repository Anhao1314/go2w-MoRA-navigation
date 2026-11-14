# 文档索引

[返回首页](../README.md)

本页是 `docs/` 的唯一导航入口，按「系统层 / Phase-1（已冻结）/ Phase-2（进行中）/ 跨阶段」分组。
每篇文档标注状态：🟢 活跃 · 🧊 冻结（随 Phase-1 归档，仅为复现维护）。
各代码区域的归属与冻结状态见[仓库布局契约](ARCHITECTURE.md#仓库布局契约)。

## 跨阶段（🟢 活跃）

| 文档 | 内容 |
| --- | --- |
| [PROJECT_FLOW.md](PROJECT_FLOW.md) | **项目全流程**：研究问题 → 7 个阶段（基础设施 / S0 基线 / 任务定义 / 单层 A / 分层 B / 对比归因 / 交付），每阶段的输入、动作、产出与判据，以及当前进度位置 |
| [RESEARCH.md](RESEARCH.md) | **研究主线**：MoRA 式多层架构 vs 端到端单层的受控对比——样本效率、成功率、泛化、失败归因、接口损失；仲裁层降为层间约束子研究 |
| [notes/training-efficiency.md](notes/training-efficiency.md) | **训练效率调研**：实测算力基线、Rudin 2021 与 Isaac Lab 官方基准、包装器开销、2 路并行与评估并行化方案、效率实验的纪律边界 |
| [notes/literature-survey.md](notes/literature-survey.md) | **开题文献清单**：按六条主题线组织（并行 RL 运动控制 / 分层 RL 理论 / 奖励设计 / 安全约束 / sim2real / 工具平台），含链接、与课题的关系及综述论证主线；线 7 为 2023–2026 增补 |
| [notes/recent-papers-2023-2026-data-practices.md](notes/recent-papers-2023-2026-data-practices.md) | **近三年文献研究**：分层 RL 足式最新工作、评估与记录惯例、数据集格式惯例（LeRobot/RLDS/robomimic）、运动模仿管线字段——并据此给出 go2-sequence-v1 的 v1.1 字段修订依据 |
| [notes/deepseek-architecture-notes.md](notes/deepseek-architecture-notes.md) | **DeepSeek 2026 架构调研**：Engram 条件记忆（查 vs 算、U 型缩放律）、mHC 流形约束；与 MoRA 分层/"查表层 vs 学习层"的对应关系及可借鉴点 |
| [PLAN.md](PLAN.md) | **执行计划**：WBS（逐任务产出与验收）、过程记录规范（日志 / ADR / 证据卡）、质量门、止损规则 |
| [thesis/proposal.md](thesis/proposal.md) | **毕业设计开题报告底稿**：选题背景、四线文献综述、研究内容与技术路线、预期成果、进度安排、16 篇参考文献（交前按学校模板套版） |
| [PROGRESS.md](PROGRESS.md) | **进度看板（单一入口）**：里程碑状态、本周任务、阻塞项、最近产物 |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | **架构 v0.2 提案 + 研究计划**：在 MoRA 的 S1 与 S0 之间引入**C++ 可验证仲裁层**（浮基 FK、CoM/支撑多边形、摩擦锥、RNID；ALLOW/CLAMP/FALLBACK/HALT），研究干预率与性能损失的权衡、失败可预判性、批量实时性 |
| [PHASE1_REPORT.md](PHASE1_REPORT.md) | 🧊 Phase-1 完整研究报告（Go2W 轮足导航，2026-10-22 归档）：分层 adapter 的首个完整证据链 |
| [ROADMAP.md](ROADMAP.md) | **Q4 项目目标与四周路线图**：岗位能力模型、P0–P4 优先级、四周时间表、验收标准、风险预案、讲述要点 |
| [MVP.md](MVP.md) | 🧊 历史：跨引擎迁移 MVP（已被 ARCHITECTURE_V2 取代，三种子基线仍继承） |
| [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | **M1/M2 预注册实验协议**（v1.0，2026-10-22 冻结）：研究问题、冻结种子与指标、验收阈值、场景阶梯 S1–S4、基线。Phase-2 的执行契约，changelog 记录一切冻结后变更 |
| [EXPERIMENT_DATA.md](EXPERIMENT_DATA.md) | 实验数据记录与论文证据协议：一次 run 保存什么、每类数据支持什么结论；Phase-2 沿用同一证据束约定 |
| [DATASET.md](DATASET.md) | **轨迹数据集规格 v1（2026-10-27 定稿）**：采集范围（L3 主实验 + L1 + 泛化，240 条 episode）、逐 step 字段、存储与校验、采集命令与成本——产出为论文数据集交付物 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方许可登记（含 C1 基线仓库的待审查条目） |
| [LIVE_SYNC.md](LIVE_SYNC.md) | 本地与 GitHub 的实时同步：提交后自动推送 + 每 2 分钟自动快进 `main`，安装/卸载与边界说明 |
| [MORA_TRAIN.md](MORA_TRAIN.md) | MoRA-Train 系统层（v0.1）：`mora/` 包接口、当前能力与边界，跨 adapter 复用 |
| [MORA_RESEARCH.md](MORA_RESEARCH.md) | MoRA 架构深度研究：公开资料还原、仓库差距分析与 v2 路线（研究记录，非复现声明） |

## Phase-2：Go2 关节级 locomotion（🟢 活跃，代码在 [`locomotion/`](../locomotion/README.md)）

| 文档/区域 | 内容 |
| --- | --- |
| [`locomotion/sim2sim/README.md`](../locomotion/sim2sim/README.md) | **S4 Sim2Sim（已实现并双门 PASS）**：零策略改动 Isaac→MuJoCo，ONNX 导出链、冻结种子评测、单因素归因 |
| [`ros2/go2_bridge/README.md`](../ros2/go2_bridge/README.md) | **ROS2 真 DDS 闭环（ROS-R1 PASS）**：50 Hz 双向桥、独立探针验收、CI 硬门禁 |
| [`vla/README.md`](../vla/README.md) | **VLA 链（进行中）**：SO-101 环境封装、脚本专家（WIP）、SmolVLA 微调与评测 |
| [`locomotion/baselines/convex_mpc/README.md`](../locomotion/baselines/convex_mpc/README.md) | C1 凸 MPC 接入位（待许可审查） |

## Phase-1：Go2W 轮足导航 adapter（🧊 已冻结归档）

以下文档正文保持原样，不随 Phase-2 回写重释；其描述的架构、命令与脚本仍可复现 Phase-1 结果。

### 研究与结果

| 文档 | 内容 |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 分层控制架构：S2 规则 / S1 PPO / S0 脚本-PD 的机制、观测演进与实现入口（含仓库布局契约段） |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 实验总览与结论：成功与失败实验结果表（证据口径：只认 `reports/` 与 `data/` 归档） |
| [INTEGRITY_UPGRADE.md](INTEGRITY_UPGRADE.md) | 2026-10-06 `navigation-integrity-v1` 完整性升级：种子、归一化、计数与报告覆盖修复（未追认历史成功率） |
| [PORTFOLIO_VALIDATION.md](PORTFOLIO_VALIDATION.md) | 2026-09-12 封存验证记录（196 passed / Pyright 0 errors，特定环境历史结果） |
| [VALIDATION.md](VALIDATION.md) | P0 验证报告：C++ 控制核心、Python 适配器、多段集成、MPC 核心的数值验证 |

### C++ S0 控制栈

| 文档 | 内容 |
| --- | --- |
| [CONTROL_DESIGN.md](CONTROL_DESIGN.md) | C++ 控制核心设计：模型、公式与实现约定 |
| [CPP_STACK_PROTOCOL.md](CPP_STACK_PROTOCOL.md) | C++ Stack P0 评估协议（版本 `cpp-stack-p0-v1`，含模型 SHA-256） |
| [CPP_ROS2_ARCHITECTURE.md](CPP_ROS2_ARCHITECTURE.md) | C++/ROS2 System-0 架构：P0 已实现，P1（ROS2 垂直切片）尚未开始 |
| [CALIBRATION.md](CALIBRATION.md) | C++ 控制栈标定记录（符号/顺序约定须有探针或夹具支撑） |
| [WINDOWS_NATIVE.md](WINDOWS_NATIVE.md) | Native Windows (MSVC) 环境验证记录 |

### 运行与复现

| 文档 | 内容 |
| --- | --- |
| [USAGE.md](USAGE.md) | 使用与复现指南（Phase-1 训练/评估/检查命令） |
| [WINDOWS_TRAINING.md](WINDOWS_TRAINING.md) | Windows 训练流程（Phase-1 Go2W） |

## 阅读路径建议

- **了解项目当前在做什么**：[PROGRESS.md](PROGRESS.md) → [PLAN.md](PLAN.md) → [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)
- **回放工作过程**：[journal/](journal/)（每日日志）→ [decisions/](decisions/)（架构决策）
- 复现 Phase-1 结果：首页研究报告 → [USAGE.md](USAGE.md) → 对应 `reports/` 目录
- 参与 Phase-2：[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) → [仓库布局契约](ARCHITECTURE.md#仓库布局契约) → [locomotion/README](../locomotion/README.md) → [MORA_TRAIN.md](MORA_TRAIN.md)（若接入 MoRA runtime）
- 审查证据链：[EXPERIMENTS.md](EXPERIMENTS.md)（Phase-1 结果表）→ [EXPERIMENT_DATA.md](EXPERIMENT_DATA.md)（证据束约定）→ run 目录与 `evidence_manifest.json`
