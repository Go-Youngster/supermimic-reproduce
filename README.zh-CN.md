# SuperMimic 复现（非官方）

[English](README.md) | 简体中文

本仓库是下面这篇论文的非官方复现，基于 PyTorch 从零实现：

> **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**<br>
> Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath<br>
> CoRL 2026（Oral）· [论文（arXiv:2610.28480）](https://arxiv.org/abs/2610.28480) · [项目主页](https://supermimic.github.io)

> [!IMPORTANT]
> 这**不是**官方实现，与 SuperMimic 作者没有关联，也未经作者认可。
> 全部代码依据论文正文与附录重新实现；论文没有写明的细节由本复现自行决定（见[与论文的差异](#与论文的差异)）。
> 下文数字均来自 A100 上的单种子实验，不代表论文结果。方法的全部功劳归原作者，复现中的错误由本仓库负责。

## 概述

SuperMimic 为 Unitree G1 人形机器人训练通用动作跟踪策略，策略这一侧采用基于模型的优化：先用刚体仿真器的转移数据拟合一个可微世界模型，再在短程想象轨迹上沿时间反向传播（BPTT）得到解析梯度来优化策略，想象步长末端由评论家自举回报。每次训练迭代（算法 A.1）交替三步：

1. **仿真采集**：4096 个 MuJoCo Warp 环境运行确定性策略并叠加动作噪声 N(0, 0.1²)，转移写入 GPU 回放缓冲（每环境 1000 步，更新/数据比 0.4）。每回合重新采样摩擦、编码器偏差和躯干质心偏移，作为仿真上下文 ξ。
2. **拟合世界模型**：在回放缓冲的 8 步窗口上做自回归展开训练（每次迭代 10 批 × 2048 个窗口）。
3. **在想象中更新策略**：冻结世界模型，从回放状态出发想象 4096 条 H = 32 步的轨迹；对想象回报做 BPTT 更新策略，评论家用想象轨迹上的 GAE 目标拟合，并在第 H 步自举。

想象状态的奖励、观测与终止都用解析公式计算，所用的 G1 可微正向运动学由仿真器与世界模型共用。

## 仓库内容

- **仿真环境**（`supermimic/sim_env.py`）：基于 MuJoCo Warp 的 G1（29 自由度）动作跟踪环境。模型、执行器与可随机化参数由 mjlab 构建；步进和重置绕过 mjlab 的管理器循环，直接读写 `qpos`/`qvel`（4096 个环境单步从 257 ms 降到 34 ms，A100）。
- **世界模型、策略、评论家**（`supermimic/networks.py`）：世界模型为 3 × 2048 的 MLP（LayerNorm + SiLU），预测根坐标系下的状态增量后解析积分；策略由目标编码器和共享解码器组成。
- **任务头与运动学**（`supermimic/task.py`、`supermimic/kinematics_fast.py`）：BeyondMimic 形式的跟踪奖励、观测、软（可微）终止；并行正向运动学与 MuJoCo 逐项核对，误差小于 1e-6。
- **SuperMimic 训练器**（`supermimic/train_supermimic.py`）；加 `--policy_algo ppo` 即为 **PPO-WM** 基线（在世界模型轨迹上做 PPO）。
- **PPO-Sim 基线**（`supermimic/train_ppo.py`）：在仿真器中做 PPO，采用 BeyondMimic 超参，动作、观测、奖励、终止、随机化、网络尺寸与评测均与 SuperMimic 相同。
- **动作库**（`supermimic/motion_data.py`、`supermimic/motion_lib.py`）：50 Hz 参考状态与按回报采样的重置（附录 D.1）。
- **评测**（`supermimic/evaluate.py`）：每条动作从首帧播放一次，无随机化、无噪声（摩擦 1.0）；触发训练终止条件即记为失败。
- **实验工具**（`scripts/`）：支持崩溃续训的 GPU 任务调度器、吞吐基准、推搡恢复测试、世界模型推演对比、参考动作质量检查、MuJoCo 离屏视频与浏览器交互式可视化器。

## 结果

LAFAN1（40 条，2.45 h），每个运行一张 A100 80GB，单种子。评测回报归一化到 [0, 1]。

| 论文结论（图） | 论文（H100） | 本复现（A100） |
| --- | --- | --- |
| SuperMimic 在 LAFAN1 上达到回报 0.82（图 4a） | 1.2 h | 3.0 h（前约 45 分钟 GPU 与其他进程分时）；15.7 h 时最佳 0.875，0/40 失败 |
| 单卡 PPO-Sim 18 h 内的表现（图 4a） | 0.60 | 17.0 h 时最佳 0.590（5/40 失败），始终未到 0.82 |
| 世界模型推演 vs MuJoCo Warp（图 5a） | 并行 2k 时 11×、16k 时 26×、64k 时 62× | 并行 1k–64k 时 13.5–16.5×（512 时 8×）；A100 上两者都在 16k 之后趋平 |
| 评论家让短想象步长可行（图 4b、c） | 无评论家 H ≤ 16 失败；H = 32 最快到 0.82 | 无评论家 H = 8 / 16 崩溃（0.016 / 0.034，40/40 失败）；有评论家 H = 16 在 5 h 内达到 0.817，H = 8 最高 0.605 后退化；到 0.82 的时间 H = 64 约 2.5 h、H = 32 约 2.7 h（未跑 H = 128） |
| 一阶梯度 vs PPO-WM（图 5b） | 想象步数少约 64× | 达到 PPO-WM 最佳训练回报所需想象步数少 35×（对 PPO-WM 16k × 32）和 14×（对 4k × 32）；walk/run/sprint 子集（18 条）评测 0.876 vs 0.555 / 0.685 |
| 世界模型误差随数据量变化（图 3b、c） | 1–264 h 时 0.007–0.011 | LAFAN1 子集代理实验：0.35 / 1.22 / 2.45 h 时 0.169 / 0.163 / 0.152（归一化方式不同，只比较趋势） |
| 推搡恢复（4.3 节） | 有评论家的策略在 0.5–2.4 s 内稳定 | 有评论家：≤ 2 m/s 全部恢复，稳定时间中位数 0.8–1.4 s，4 m/s 时 38% 摔倒；H = 32 无评论家：3 / 4 m/s 时 50% / 88% 摔倒 |

消融中“到达目标回报的时间”按无争用时长换算（迭代数 × 无争用时每次迭代耗时）。未复现部分：BONES-SEED 相关实验（与 SONIC 对比、129,514 条动作基准、1–264 h 规模曲线）、真机迁移、速度指令接口。

## 与论文的差异

| 项目 | 论文 | 本复现 | 原因 |
| --- | --- | --- | --- |
| 硬件 / 精度 | H100，TF32 | A100，MLP 矩阵乘用 bf16 autocast，其余 fp32 | 可用硬件；A100 上 bf16 吞吐翻倍 |
| 训练长度 | 32 万次迭代 | 按墙钟预算（主实验 16 h，消融 4–6 h） | 每次迭代约 0.56 s，论文约 0.11 s |
| 动作数据 | BONES-SEED（264 h）+ LAFAN1 | 仅 LAFAN1 | 数据可得性 |
| PPO-Sim 规模 | 1 / 4 / 8 卡 | 1 卡 | GPU 预算 |
| 评测集 | 129,514 条 BONES-SEED 动作 | 40 条 LAFAN1 | 数据 |
| 随机种子 | – | 每个配置 1 个种子 | 算力 |

论文未写明的细节按如下方式实现：

- **世界模型状态**：MuJoCo 模型的完整马尔可夫状态（71 维：根位置、四元数、线速度、角速度，29 个关节角与角速度）。PD 执行器与软接触都没有内部状态。
- **世界模型输入/输出**：输入对平移与偏航不变（根高度、投影重力、根系下速度、关节状态、由可微正向运动学得到的 30 个刚体在根系下的位置/高度/速度、动作、PD 目标误差、ξ）；输出为根系下的位移、旋转向量、速度增量与关节增量。
- **预测头**：解析奖励（表 A.2）、解析观测（表 A.1），终止用根高度/倾斜偏差的软 sigmoid（温度 2 cm / 0.04），得到可微的存活权重。
- **目标编码**：10 个未来参考帧（710 维），不含全局位置，部署时无需状态估计。
- **奖励时序**：执行动作后的状态 s<sub>t+1</sub> 与第 t+1 个参考帧比较；根为骨盆。
- **重置采样**：50% 均匀 + 40% 按 1 秒片段 softmax(−25 · G<sub>b</sub> / G<sub>max</sub>) + 10% 历史权重；重置时沿用 BeyondMimic 的位姿/速度/关节扰动。

## 安装

测试环境：Linux、NVIDIA A100 80GB、Python 3.12、PyTorch 2.13（CUDA）、mjlab 1.6.0（MuJoCo 3.11、MuJoCo Warp 3.11、Warp 1.18）、viser 1.1.1、mjviser 0.0.14。默认 SuperMimic 运行的显存峰值约 16 GB（H = 64 时约 27 GB）。

```bash
git clone https://github.com/Go-Youngster/supermimic-reproduce.git && cd supermimic-reproduce
python3.12 -m venv .venv                     # 调度器默认使用 .venv/ 下的环境
.venv/bin/pip install torch                  # 按驱动选择对应的 CUDA 版本
.venv/bin/pip install mjlab==1.6.0           # 会一并安装 MuJoCo、MuJoCo Warp、Warp、viser、mjviser、imageio-ffmpeg
.venv/bin/pip install huggingface_hub        # 提供下载数据用的 hf 命令
sudo apt-get install libosmesa6 fonts-wqy-microhei   # 渲染视频用：OSMesa（未设置 MUJOCO_GL 时）与字幕字体
```

## 数据

论文的 LAFAN1 实验使用 Unitree 官方把 LAFAN1 运动学重定向到 G1 的版本（40 条，30 FPS 的 CSV）。

```bash
# 1. 下载 G1 的 CSV 文件
.venv/bin/hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
    --include "g1/*" --local-dir data/raw/lafan1_g1

# 2. 生成 50 Hz 动作库（用 MuJoCo 运动学计算被跟踪刚体的状态）
PYTHONPATH=. .venv/bin/python -m supermimic.motion_data \
    --csv-dir data/raw/lafan1_g1/g1 --out data/motions/lafan1.npz
# -> {'clips': 40, 'frames': 441120, 'hours': 2.45...}
```

## 训练

以下命令均在仓库根目录执行。

```bash
# SuperMimic（默认值与论文一致：4096 个环境、H = 32、带评论家）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m supermimic.train_supermimic \
    --run_dir runs/sm_lafan1_main --max_hours 16 --eval_every_min 20

# PPO-Sim 基线（8192 个环境）
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m supermimic.train_ppo \
    --run_dir runs/ppo_lafan1_1gpu --max_hours 18 --eval_every_min 30
```

`Config`（`train_supermimic.py`）与 `PPOConfig`（`train_ppo.py`）的每个字段都是命令行参数，布尔值写 `true` / `false`。常用参数：

| 参数 | 含义 |
| --- | --- |
| `--horizon 16` | 想象步长 H |
| `--use_critic false` | 不用评论家自举（纯 H 步 BPTT） |
| `--clips walk,run,sprint` | 只保留名称包含任一子串的动作 |
| `--policy_algo ppo --ppo_rollouts 16384 --ppo_steps 32` | PPO-WM 基线 |
| `--max_hours` / `--eval_every_min` | 墙钟训练预算（不含评测时间）/ 评测间隔 |

每个运行在 `--run_dir` 下写出 `config.json`、`metrics.jsonl`、`eval.jsonl`、`status.json` 与 `ckpt/{latest,best}.pt`。重复执行同一命令会从 `ckpt/latest.pt` 续训（回放缓冲会重新填充）。

### 全部实验

`experiments/queue.json` 包含上述结果对应的 17 个任务：14 个训练运行（主对照、想象步长/评论家消融、PPO-WM、数据规模代理实验）和 3 个后处理任务（推搡测试、世界模型推演、速度基准）。调度器每张卡跑一个任务，后处理任务在其依赖的运行完成后启动，崩溃的任务从最新检查点续训：

```bash
.venv/bin/python scripts/scheduler.py --gpus 0,1,2,3
```

其中 3 个任务带 `"gpu"` 字段，固定在原机器的某张卡上；换机器时删掉该字段，或改成 `--gpus` 中的编号。进度写在 `runs/scheduler_state.json`、`runs/scheduler.log` 与 `runs/<任务名>/train.log`。

## 分析与可视化

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_speed.py        # 世界模型 vs MuJoCo Warp -> results/speed_benchmark.json
PYTHONPATH=. .venv/bin/python scripts/push_test.py --runs sm_lafan1_main,sm_h32_nocritic   # -> results/push_test.json
PYTHONPATH=. .venv/bin/python scripts/wm_rollout.py --run sm_lafan1_main                    # -> results/wm_rollout.json
PYTHONPATH=. .venv/bin/python scripts/ref_quality.py        # 参考动作穿地/悬空检查 -> results/ref_quality.json
PYTHONPATH=. .venv/bin/python scripts/render_videos.py      # 训练好的策略视频 -> media/videos（OSMesa）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python scripts/sim_viewer.py --port 8800     # 交互式可视化器（viser）
```

`results/` 中是本复现运行上述脚本得到的结果，可与自己的结果对比。

> [!NOTE]
> 可视化器默认监听 `0.0.0.0`，在共享机器上请加 `--host 127.0.0.1`。

## 测试

```bash
PYTHONPATH=. .venv/bin/python tests/test_kinematics.py                 # 可微正向运动学 vs MuJoCo
```

## 目录结构

```text
supermimic/              核心代码
  sim_env.py             基于 MuJoCo Warp（mjlab）的 G1 动作跟踪仿真环境
  task.py                解析的奖励 / 观测 / 终止
  g1.py, kinematics*.py  G1 描述与可微正向运动学
  networks.py            世界模型、策略与评论家
  motion_data.py         重定向 CSV -> 50 Hz 动作库
  motion_lib.py          GPU 动作库与按回报采样的重置
  replay.py              GPU 回放缓冲
  train_supermimic.py    SuperMimic（加 --policy_algo ppo 为 PPO-WM）
  train_ppo.py           PPO-Sim 基线
  evaluate.py            评测协议
  runlog.py              指标、检查点与状态文件
  postprocess.py         调度器启动的分析任务
  vis.py                 视频与可视化器使用的 MuJoCo 场景
scripts/                 调度器、基准、分析、渲染、可视化器
experiments/queue.json   上述结果使用的实验队列
results/                 本复现的分析结果（JSON）
tests/                   正向运动学与 MuJoCo 的一致性检查
```

## 致谢

- 衷心感谢 **SuperMimic** 的作者 Qiayuan Liao、Yiyang Shao、Junfeng Long、Kevin Zakka、Roberto Horowitz、Pieter Abbeel 与 Koushil Sreenath 提出这一方法，并在论文与附录中给出了足以支撑复现的细节。如使用本仓库代码，请引用原论文（见下文）。
- [BeyondMimic](https://beyondmimic.github.io/)：跟踪奖励、重置随机化与 PPO 超参。
- [mjlab](https://github.com/mujocolab/mjlab)：G1 模型、执行器与域随机化；以及 [MuJoCo](https://github.com/google-deepmind/mujoco)、[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)、[NVIDIA Warp](https://github.com/NVIDIA/warp) 和 [PyTorch](https://pytorch.org)。
- Unitree 的 [LAFAN1 重定向数据集](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 与 Ubisoft La Forge 的 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)。
- [viser](https://github.com/viser-project/viser) 与 [mjviser](https://github.com/mujocolab/mjviser)（交互式可视化器）。

## 引用

请引用原论文：

```bibtex
@article{liao2026supermimic,
  title   = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author  = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}
```

## 数据与第三方许可

- LAFAN1 采用 CC BY-NC-ND 4.0 许可，Unitree 的重定向版本基于该数据集。本仓库不包含任何动作数据、动作库、检查点或视频；使用或分享由其派生的内容时请遵守数据集许可。
- G1 模型在运行时从 mjlab（Apache-2.0）加载。
