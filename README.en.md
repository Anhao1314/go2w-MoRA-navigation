# Hierarchical RL turns an impossible long-horizon quadruped task into a reliable one: a controlled comparison

**Design and Simulation Validation of a Quadruped Robot Locomotion Control System Based on Hierarchical Reinforcement Learning** (undergraduate thesis project repository)

[中文版](README.md) · [Docs index](docs/README.md) · [Full Phase-1 research report](docs/PHASE1_REPORT.md) · [Experiment protocol](docs/EXPERIMENT_DESIGN.md) · [Daily journal](docs/journal/)

---

## One-line result

> Under identical task, observation and training budget, the **hierarchical architecture reached 100% success on all 3 training seeds within 100 iterations on a four-stage long-horizon task; the end-to-end single-layer policy succeeded on only 1 of 3 seeds (450 iterations), failing completely on the other two after 1500 iterations**. The hierarchy lifts training success from 1/3 to 3/3 and compresses convergence from "450 iterations at best / never at worst" to a stable 100. On an easy task both succeed, with the hierarchical agent converging 3x faster. Failures are attributable: non-converging single-layer seeds die at the walk-to-turn gait switch.

## Key numbers

### L3 (hard): forward 2 m -> turn 90 deg in place -> forward 2 m -> stop 1 s (3 training seeds per group)

| Training seed | Single-layer success | Single-layer convergence | Layered success | Layered convergence |
| --- | --- | --- | --- | --- |
| 0 | 0% (no convergence in 1500, fell every episode) | — | 100% | 100 iters |
| 1 | 0% (no convergence in 1500) | — | 100% | 100 iters |
| 2 | 100% | 450 iters | 100% | 100 iters |

> Methodological note: running only seed 2 would conclude "the single layer works, just 4.5x slower"; running only seeds 0/1 would conclude "the single layer cannot work at all". Both would be wrong. Multiple seeds are not a nicety — they are a precondition for a correct conclusion.

### L1 (easy): forward 1.2 m -> stop 0.5 s

| Metric | Single layer | Layered |
| --- | --- | --- |
| Success rate | 100% (converged at 150 iters) | 100% (**50 iters**) |

**What the ladder means**: on the easy task the hierarchy is merely *faster* (an efficiency win); once the task requires switching locomotion modes, it becomes *possible vs impossible* — and the single-layer failure is precisely localized to the walk-to-turn switch by stage reach rates, a diagnostic only the layered structure provides.

**Sample videos**: [layered completing all four stages](media/training/robot_motion_l3_layered.mp4) · [single layer falling at the turn](media/training/robot_motion_l3_single_fail.mp4) · [sample-efficiency curves](media/training/sweep_ab_comparison.png)

Full reports: [L3 comparison](reports/training/l3_ab_comparison.md) · [L1 comparison](reports/training/ab_comparison.md) · [M1 low-level policy](reports/locomotion/m1_evaluation.md)

## System

```text
Task/stage manager (rule layer: closed-loop velocity commands + stage gating)
   |  [vx, wz] command
   v
S1 command policy (PPO, 2-dim actions) -- layered group --+
                                                          +-> 12 joints -> Unitree Go2 (Isaac Sim 5.1)
S0 locomotion policy (PPO, frozen, reused) --------------+
   ^
   |  48-dim proprioception (M1-pretrained: vx_rmse 0.057 m/s, three seeds)
```

- **Single-layer group**: the 53-dim observation maps directly to 12-dim joint actions — identical task information to the layered group
- **Layered group**: S1 learns in 2-dim command space; execution is delegated to a pretrained, frozen S0

## Engineering for reproducibility

| Facility | Description |
| --- | --- |
| Pre-registered protocol | evaluation seeds E={0..19}, metrics and thresholds frozen before training ([EXPERIMENT_DESIGN](docs/EXPERIMENT_DESIGN.md)) |
| Per-run evidence bundle | `run_config.json` (git SHA/dependency versions/command line) + SHA-256 manifest of every file, auto-generated |
| Training reports | every run auto-produces a Markdown report with first/last/extrema/trends and automated anomaly detection ([example](reports/training/2026-10-24_01-18-00.md)) |
| Convergence early-stop | deterministic evaluation on 2048 envs every 50 iterations; stops after consecutive passes and records the convergence point |
| Live dashboard | local web panel: progress, reward curves, GPU, snapshot replay, report and video links |
| Motion videos | recorded body trajectories rendered as 3-D skeleton animations (success and failure both replayable) |
| Daily journal | [docs/journal/](docs/journal/): goals, decisions and **failures-with-fixes** per day (a dozen real debugging cases) |

## Honest boundaries

- **Simulation only**: no real-robot experiments; conclusions hold for Isaac Sim 5.1
- **S0 pretraining cost**: the layered S0 took ~1200 s to pretrain; part of the layered advantage comes from action abstraction and S0 reuse (an untrained S1 already scores 75%) rather than S1 learning faster — attributed in the report
- **Closed-loop commands encode task knowledge**: the rule layer's heading correction embeds part of the task prior — this is a deliberate hierarchical trade-off, discussed in the report
- **3 training seeds per group** (S0 fixed to the M1 seed-0 policy; the seed varies the upper-level training); more seeds and real-robot validation are listed as future work

## Repository layout

| Path | Content |
| --- | --- |
| [`locomotion/`](locomotion/README.md) | Phase-2 code: task environments, A/B trainer, evaluation, action mapping |
| [`mora/`](docs/MORA_TRAIN.md) | Layered runtime system layer (S2->S1->S0 contracts, memory, progress loop) |
| [`cpp/`](cpp/README.md) | C++ control stack (FK/IK/MPC numerically validated) + guardrail kinematics core |
| [`reports/`](reports/training/ab_comparison.md) | Training reports, comparison reports, sweep data (all auto-generated) |
| [`docs/`](docs/README.md) | Protocol, architecture, roadmap, daily journal, ADR decision records |
| `scripts/` | Training, evaluation, sweep, report, dashboard and video entry points |
| `media/training/` | Key videos and curves (process videos archived under `media/archive/`) |

## Phase-1: Go2W wheeled navigation (completed, archived)

This repository previously hosted long-horizon navigation research on the Unitree Go2W
wheeled quadruped (the first MoRA-Train adapter: curve 20/20, multi-segment 20/20,
junction 40/40, C++ S0 numerically validated). The full research report is preserved at
[docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md); the Phase-2 comparison methodology and
engineering facilities evolved from that stage.

## Reproducing

```powershell
$env:PYTHONPATH = "$PWD"; $env:OMNI_KIT_ACCEPT_EULA = "YES"

# Layered group (L3)
.venv311\Scripts\python.exe scripts\train_ab.py --mode layered `
    --s0-checkpoint <S0 checkpoint> --num_envs 2048 --max_iterations 1500 --seed 0

# Single-layer group (L3)
.venv311\Scripts\python.exe scripts\train_ab.py --mode single `
    --num_envs 2048 --max_iterations 1500 --seed 0

# Frozen-seed evaluation
.venv311\Scripts\python.exe scripts\eval_policy.py --task Go2Loco-Sequence-v0 `
    --run-dir <run> [--layered --s0-checkpoint <S0>] --episodes 20
```

Environment: Python 3.11 + Isaac Lab 2.7.0 + Isaac Sim 5.1.0.0 + RTX 5060 Ti (version pins in [locomotion/README](locomotion/README.md)).

## License and contributions

Original code is [MIT licensed](LICENSE); Go2W assets and third-party materials retain their own licenses ([notices](docs/THIRD_PARTY_NOTICES.md)). Phase-1 contribution credits are preserved in the archived report ([docs/PHASE1_REPORT.en.md](docs/PHASE1_REPORT.en.md)).
