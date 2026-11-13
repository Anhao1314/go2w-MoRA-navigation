# Third-Party Notices

[返回首页](../README.md) · [架构](ARCHITECTURE.md) · [实验](EXPERIMENTS.md) · [使用指南](USAGE.md)

## westonrobot/unitree_mujoco (BSD 3-Clause License)

本项目 `rl/terrain.py` 的运行时高度场地形生成器，其“Perlin 噪声生成高度场”
思路参考自：

- 项目：https://github.com/westonrobot/unitree_mujoco
- 文件：`terrain_tool/terrain_generator.py`（`AddPerlinHeighField`）

### 适配说明

原实现使用 `noise` + `cv2` 生成 PNG 高度图，再通过 XML 的 `<hfield file=...>`
加载。本项目改为：在 MuJoCo 3.11 运行时用 numpy 生成归一化高度矩阵，直接写入
`model.hfield_data`（逐 episode 随机化，无需新增依赖、无需重新编译模型）。
仅借鉴“Perlin 高度场”算法思路，未复制原代码。

### BSD 3-Clause License

Copyright (c) 2025, Weston Robot
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## google-deepmind/mujoco_menagerie (BSD 3-Clause License)

S4 Sim2Sim（`locomotion/sim2sim/`）所需的四足版 Unitree Go2 官方 MuJoCo 模型，
直接采用上游打包好的 MJCF 与 mesh，未做修改：

- 项目：https://github.com/google-deepmind/mujoco_menagerie
- 子目录：`unitree_go2/`
- 上游固定提交：`370c43965323c0944b433b202aac5058b5f907a9`
- 引入文件：`models/go2/go2.xml`、`models/go2/assets/*.obj`、`models/go2/LICENSE`、`models/go2/UPSTREAM_README.md`

### 适配说明

`go2.xml` 与 `assets/` 上游原样（关节名 `{FL,FR,RL,RR}_{hip,thigh,calf}_joint`
与 Isaac USD 侧导出 `locomotion/guardrail/go2_model.json` 完全一致）。项目自建
`models/go2/go2_flat_scene.xml`（平面场景 + 演示相机，仿照 `models/go2w/` 场景
模式，`<include>` 引入上游模型）；timestep 与执行器契约在运行时代码中设置，不
修改上游文件。详细对应关系见 [models/go2/README.md](../models/go2/README.md)。

### BSD 3-Clause License

Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Pending license review (no code adopted yet)

### elijah-waichong-chan/go2-convex-mpc

- 项目：https://github.com/elijah-waichong-chan/go2-convex-mpc
- 计划用途：冻结协议 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) §6 的 **C1 凸 MPC 基线**（MuJoCo + Pinocchio），接入与 RL 策略相同的评估循环（RQ2 按种子配对对比）。
- 状态：**待审查**。按协议要求，许可审查完成并在此登记结论之前，`locomotion/baselines/convex_mpc/` 不引入任何上游代码或衍生实现。
- 接入方式意向：wrapper 封装调用（不修改上游代码）；接入位占位见 [locomotion/baselines/convex_mpc/README.md](../locomotion/baselines/convex_mpc/README.md)。

## C++ control stack dependencies

The P0 `cpp/` build fetches these dependencies at configure/build time; they are
not vendored in this repository. See each upstream project for the full license
text.

| Dependency | Version | License | Use |
| --- | --- | --- | --- |
| Eigen | 3.4.0 | MPL-2.0 | linear algebra |
| OSQP | 0.6.3 | Apache-2.0 | SRBD MPC QP |
| qdldl | bundled with OSQP | Apache-2.0 | sparse LDL factorization |
| Catch2 | 3.7.1 | BSL-1.0 | C++ tests |
| pybind11 | 3.x | BSD-3-Clause | Python bindings |
