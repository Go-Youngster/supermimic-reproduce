# SuperMimic Reproduction (Unofficial)

English | [简体中文](README.zh-CN.md)

An unofficial, from-scratch PyTorch reproduction of

> **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**<br>
> Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath<br>
> CoRL 2026 (Oral) · [Paper (arXiv:2610.28480)](https://arxiv.org/abs/2610.28480) · [Project page](https://supermimic.github.io)

> [!IMPORTANT]
> This is **not** the official implementation, and it is not affiliated with or endorsed by the SuperMimic authors.
> Everything here was re-implemented from the paper and its appendix. Where the paper leaves details open, the
> choices are ours (see [Differences from the paper](#differences-from-the-paper)). All numbers below come from
> single-seed runs on A100 GPUs and are not the paper's results. Credit for the method goes to the original authors;
> any mistakes are ours.

## Overview

SuperMimic trains a general motion-tracking policy for the Unitree G1 humanoid without model-free RL on the policy side.
A differentiable world model is fit to transitions logged from a rigid-body simulator, and the policy is optimized
with analytic gradients obtained by backpropagation through time over short imagined rollouts, with a critic
bootstrapping the return at the end of the horizon. One training iteration (Algorithm A.1) alternates:

1. **Collect** – 4096 MuJoCo Warp environments run the deterministic policy plus action noise N(0, 0.1²) and write
   transitions to a GPU replay buffer (1000 steps per environment, update-to-data ratio 0.4). Friction, encoder bias
   and torso COM offset are resampled per episode as the simulation context ξ.
2. **Fit the world model** – on 8-step windows from the replay buffer, rolled out autoregressively
   (10 batches × 2048 windows per iteration).
3. **Improve the policy in imagination** – the frozen world model rolls out 4096 trajectories of H = 32 steps from
   replay states; the policy is updated by BPTT through the imagined return, and the critic (fit to GAE targets on the
   imagined trajectories) bootstraps step H.

Rewards, observations and terminations of imagined states are computed analytically, with a differentiable forward
kinematics of the G1 shared by the simulator and the world model.

## What is included

- **Simulator** (`supermimic/sim_env.py`): G1 (29 DoF) motion-tracking environment on MuJoCo Warp. mjlab builds the
  model, actuators and randomizable fields; stepping and resets bypass the manager loop and read `qpos`/`qvel`
  directly (4096 environments: 257 ms → 34 ms per control step on an A100).
- **World model, policy, critic** (`supermimic/networks.py`): world-model MLP 3 × 2048 (LayerNorm + SiLU) that predicts
  root-frame state increments which are integrated analytically; policy with a goal encoder and a shared decoder.
- **Task heads and kinematics** (`supermimic/task.py`, `supermimic/kinematics_fast.py`): BeyondMimic-style tracking reward,
  observations, soft (differentiable) termination; parallel forward kinematics checked against MuJoCo (error < 1e-6).
- **SuperMimic trainer** (`supermimic/train_supermimic.py`); `--policy_algo ppo` turns it into the **PPO-WM** baseline
  (PPO on world-model rollouts).
- **PPO-Sim baseline** (`supermimic/train_ppo.py`): PPO in the simulator with BeyondMimic hyperparameters and the same
  motions, observations, reward, termination, randomization, network sizes and evaluation.
- **Motion library** (`supermimic/motion_data.py`, `supermimic/motion_lib.py`): 50 Hz reference states and
  return-based reset sampling (Appendix D.1).
- **Evaluation** (`supermimic/evaluate.py`): every clip is played once from its first frame without randomization or
  noise (friction 1.0); a clip fails if a training termination triggers.
- **Experiment tooling** (`scripts/`): GPU job scheduler with resume-on-crash, throughput benchmark, push-recovery test,
  world-model rollout check, reference-motion quality check, offscreen MuJoCo videos and an interactive browser viewer.

## Results

LAFAN1 (40 clips, 2.45 h), one A100 80GB per run, single seed. Evaluation return is normalized to [0, 1].

| Paper claim (figure) | Paper (H100) | This reproduction (A100) |
| --- | --- | --- |
| SuperMimic reaches return 0.82 on LAFAN1 (Fig. 4a) | 1.2 h | 3.0 h (the GPU was time-shared for the first ~45 min); best 0.875 at 15.7 h, 0/40 failures |
| PPO-Sim on 1 GPU within 18 h (Fig. 4a) | 0.60 | best 0.590 at 17.0 h (5/40 failures); never reaches 0.82 |
| World-model rollouts vs MuJoCo Warp (Fig. 5a) | 11× at 2k, 26× at 16k, 62× at 64k parallel | 13.5–16.5× at 1k–64k (8× at 512); both saturate beyond 16k on the A100 |
| A critic makes short horizons viable (Fig. 4b, c) | no critic: H ≤ 16 fails; H = 32 is fastest to 0.82 | no critic: H = 8 / 16 collapse (0.016 / 0.034, 40/40 failures); with critic: H = 16 reaches 0.817 in 5 h, H = 8 peaks at 0.605 and then degrades; time to 0.82: H = 64 ≈ 2.5 h, H = 32 ≈ 2.7 h (H = 128 not run) |
| First-order gradients vs PPO-WM (Fig. 5b) | ≈ 64× fewer imagined steps | 35× (vs PPO-WM 16k × 32) and 14× (vs 4k × 32) fewer imagined steps to PPO-WM's best training return; walk/run/sprint subset (18 clips): 0.876 vs 0.555 / 0.685 |
| World-model error vs data size (Fig. 3b, c) | 0.007–0.011 from 1 to 264 h | proxy on LAFAN1 subsets: 0.169 / 0.163 / 0.152 at 0.35 / 1.22 / 2.45 h (different normalization, compare trends only) |
| Push recovery (Sec. 4.3) | critic policy settles within 0.5–2.4 s | with critic: all pushes ≤ 2 m/s recovered, median settling 0.8–1.4 s, 38% falls at 4 m/s; H = 32 without critic: 50% / 88% falls at 3 / 4 m/s |

Times to a target return in the ablations are converted to uncontended hours (iterations × uncontended seconds per
iteration). Not reproduced: the BONES-SEED experiments (comparison with SONIC, the 129,514-motion benchmark, the 1–264 h
scaling study), real-robot transfer and the velocity-command interface.

## Differences from the paper

| Item | Paper | This reproduction | Reason |
| --- | --- | --- | --- |
| Hardware / precision | H100, TF32 | A100, bf16 autocast for MLP matmuls, fp32 elsewhere | available hardware; bf16 doubles A100 throughput |
| Training length | 320k iterations | wall-clock budgets (16 h main run, 4–6 h ablations) | ≈ 0.56 s per iteration vs ≈ 0.11 s in the paper |
| Motion data | BONES-SEED (264 h) + LAFAN1 | LAFAN1 only | data availability |
| PPO-Sim scale | 1 / 4 / 8 GPUs | 1 GPU | GPU budget |
| Evaluation set | 129,514 BONES-SEED motions | 40 LAFAN1 clips | data |
| Seeds | – | one seed per configuration | compute |

Details that the paper leaves open were decided as follows:

- **World-model state**: the full Markov state of the MuJoCo model (71-D: root position, quaternion, linear and angular
  velocity, 29 joint positions and velocities). PD actuators and soft contacts carry no internal state.
- **World-model inputs/outputs**: translation- and yaw-invariant inputs (root height, projected gravity, root-frame
  velocities, joint states, root-frame positions/heights/velocities of 30 bodies from differentiable FK, action,
  PD-target error, ξ); outputs are root-frame displacement, rotation vector, velocity and joint increments.
- **Prediction heads**: analytic reward (Table A.2), observations (Table A.1) and a soft sigmoid termination on root
  height / tilt deviation (temperatures 2 cm / 0.04) that yields a differentiable survival weight.
- **Goal encoding**: 10 future reference frames (710-D) without global positions, so no state estimation is needed.
- **Reward timing**: the state after the action, s<sub>t+1</sub>, is compared with reference frame t+1; the root is the pelvis.
- **Reset sampling**: 50% uniform, 40% softmax(−25 · G<sub>b</sub> / G<sub>max</sub>) over 1-second bins, 10% historical
  weights; BeyondMimic pose/velocity/joint perturbations at reset.

## Installation

Tested on Linux with an NVIDIA A100 80GB, Python 3.12, PyTorch 2.13 (CUDA), mjlab 1.6.0 (MuJoCo 3.11, MuJoCo Warp 3.11,
Warp 1.18), viser 1.1.1 and mjviser 0.0.14. The default SuperMimic run peaks at about 16 GB of GPU memory
(about 27 GB with H = 64).

```bash
git clone https://github.com/Go-Youngster/supermimic-reproduce.git && cd supermimic-reproduce
python3.12 -m venv .venv                     # the scheduler expects the environment at .venv/
.venv/bin/pip install torch                  # choose the CUDA build that matches your driver
.venv/bin/pip install mjlab==1.6.0           # also installs MuJoCo, MuJoCo Warp, Warp, viser, mjviser, imageio-ffmpeg
.venv/bin/pip install huggingface_hub        # provides the `hf` CLI for the data download
sudo apt-get install libosmesa6 fonts-wqy-microhei   # video rendering: OSMesa (unless MUJOCO_GL is set) and the caption font
```

## Data

The paper's LAFAN1 experiments use Unitree's kinematic retargeting of LAFAN1 to the G1 (40 clips, 30 FPS CSV).

```bash
# 1. Download the G1 CSV files
.venv/bin/hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
    --include "g1/*" --local-dir data/raw/lafan1_g1

# 2. Build the 50 Hz motion library (tracked-body states via MuJoCo kinematics)
PYTHONPATH=. .venv/bin/python -m supermimic.motion_data \
    --csv-dir data/raw/lafan1_g1/g1 --out data/motions/lafan1.npz
# -> {'clips': 40, 'frames': 441120, 'hours': 2.45...}
```

## Training

Run all commands from the repository root.

```bash
# SuperMimic (defaults follow the paper: 4096 environments, H = 32, critic)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m supermimic.train_supermimic \
    --run_dir runs/sm_lafan1_main --max_hours 16 --eval_every_min 20

# PPO-Sim baseline (8192 environments)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m supermimic.train_ppo \
    --run_dir runs/ppo_lafan1_1gpu --max_hours 18 --eval_every_min 30
```

Every field of `Config` (`train_supermimic.py`) and `PPOConfig` (`train_ppo.py`) is a command-line flag; booleans
take `true` / `false`. Frequently used ones:

| Flag | Meaning |
| --- | --- |
| `--horizon 16` | imagination horizon H |
| `--use_critic false` | no critic bootstrap (pure H-step BPTT) |
| `--clips walk,run,sprint` | keep clips whose names contain any of these substrings |
| `--policy_algo ppo --ppo_rollouts 16384 --ppo_steps 32` | PPO-WM baseline |
| `--max_hours` / `--eval_every_min` | wall-clock training budget (evaluation excluded) / evaluation interval |

A run writes `config.json`, `metrics.jsonl`, `eval.jsonl`, `status.json` and `ckpt/{latest,best}.pt` to its
`--run_dir`. Re-running the same command resumes from `ckpt/latest.pt` (the replay buffer is refilled).

### All experiments

`experiments/queue.json` contains the 17 jobs behind the results above: 14 training runs (main comparison,
horizon/critic ablation, PPO-WM, data-size proxy) and 3 post-processing jobs (push test, world-model rollout, speed
benchmark). The scheduler runs one job per GPU, starts post-processing jobs once the runs they depend on are done and
resumes crashed jobs from their latest checkpoint:

```bash
.venv/bin/python scripts/scheduler.py --gpus 0,1,2,3
```

Three jobs carry a `"gpu"` field that pins them to a GPU index of the original machine; remove it or change it to an
index from `--gpus`. Progress is written to `runs/scheduler_state.json`, `runs/scheduler.log` and
`runs/<job>/train.log`.

## Analysis and visualization

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_speed.py        # world model vs MuJoCo Warp -> results/speed_benchmark.json
PYTHONPATH=. .venv/bin/python scripts/push_test.py --runs sm_lafan1_main,sm_h32_nocritic   # -> results/push_test.json
PYTHONPATH=. .venv/bin/python scripts/wm_rollout.py --run sm_lafan1_main                    # -> results/wm_rollout.json
PYTHONPATH=. .venv/bin/python scripts/ref_quality.py        # floor penetration / floating of the references -> results/ref_quality.json
PYTHONPATH=. .venv/bin/python scripts/render_videos.py      # MP4s of the trained policies -> media/videos (OSMesa)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python scripts/sim_viewer.py --port 8800     # interactive viewer (viser)
```

`results/` holds the outputs of these scripts from this reproduction, for comparison with your own runs.

> [!NOTE]
> The viewer listens on `0.0.0.0` by default; pass `--host 127.0.0.1` on shared machines.

## Tests

```bash
PYTHONPATH=. .venv/bin/python tests/test_kinematics.py                 # differentiable FK vs MuJoCo
```

## Repository layout

```text
supermimic/              core package
  sim_env.py             G1 motion-tracking simulator on MuJoCo Warp (via mjlab)
  task.py                analytic reward / observation / termination heads
  g1.py, kinematics*.py  G1 description and differentiable forward kinematics
  networks.py            world model, policy and critic
  motion_data.py         retargeted CSV -> 50 Hz motion library
  motion_lib.py          GPU motion library with return-based reset sampling
  replay.py              GPU replay buffer
  train_supermimic.py    SuperMimic (and PPO-WM with --policy_algo ppo)
  train_ppo.py           PPO-Sim baseline
  evaluate.py            evaluation protocol
  runlog.py              metrics, checkpoints and status files
  postprocess.py         analysis jobs launched by the scheduler
  vis.py                 MuJoCo scene for videos and the viewer
scripts/                 scheduler, benchmarks, analysis, rendering, viewer
experiments/queue.json   experiment queue used for the results above
results/                 analysis outputs of this reproduction (JSON)
tests/                   forward-kinematics check against MuJoCo
```

## Acknowledgements

- Many thanks to the **SuperMimic** authors — Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz,
  Pieter Abbeel and Koushil Sreenath — for the method and for a paper and appendix detailed enough to make this
  reproduction possible. If you use this code, please cite their paper (below).
- [BeyondMimic](https://beyondmimic.github.io/) for the tracking reward, reset randomization and PPO hyperparameters.
- [mjlab](https://github.com/mujocolab/mjlab) for the G1 model, actuators and domain randomization;
  [MuJoCo](https://github.com/google-deepmind/mujoco), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
  [NVIDIA Warp](https://github.com/NVIDIA/warp) and [PyTorch](https://pytorch.org).
- Unitree's [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) and
  Ubisoft La Forge's [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
- [viser](https://github.com/viser-project/viser) and [mjviser](https://github.com/mujocolab/mjviser) for the
  interactive viewer.

## Citation

Please cite the original paper:

```bibtex
@article{liao2026supermimic,
  title   = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author  = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}
```

## Data and third-party licenses

- LAFAN1 is licensed under CC BY-NC-ND 4.0, and Unitree's retargeted version is based on it. This repository contains
  no motion data, motion libraries, checkpoints or videos; respect the dataset license when using or sharing anything
  derived from it.
- The G1 model is loaded at runtime from mjlab (Apache-2.0).
