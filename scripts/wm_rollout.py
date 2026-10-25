"""World-model fidelity check: roll the learned model next to the simulator from the same start state.

open loop  : the world model replays the simulator's actions;
closed loop: the policy acts on the world model's own predictions (as in imagination).
Writes results/wm_rollout.json with per-step root height, root xy, selected joints and normalized error.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from supermimic.g1 import JOINT_NAMES
from supermimic.motion_lib import MotionLibrary
from supermimic.networks import AMP, Policy, WorldModel, state_error
from supermimic.sim_env import TrackingSim
from supermimic.task import GOAL_FRAME_DIM, OBS_DIM, goal_obs, proprio_obs

JOINTS = ("left_hip_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint")


@torch.no_grad()
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="sm_lafan1_main")
  ap.add_argument("--ckpt", default="latest")
  ap.add_argument("--clips", default="walk1_subject1:600,dance2_subject1:1500,jumps1_subject1:2400,fallAndGetUp1_subject1:900")
  ap.add_argument("--steps", type=int, default=150)
  a = ap.parse_args()
  dev = "cuda:0"
  torch.backends.cuda.matmul.allow_tf32 = True
  cfg = json.loads(Path(f"runs/{a.run}/config.json").read_text())["config"]
  AMP["bf16"] = cfg.get("bf16", False)
  lib = MotionLibrary(cfg["motion_file"], dev)
  specs = [(c.split(":")[0], int(c.split(":")[1])) for c in a.clips.split(",")]
  frames = torch.tensor([int(lib.clip_start[lib.names.index(n)]) + off for n, off in specs], device=dev)
  sim = TrackingSim(lib, len(specs), dev, train=False, max_episode_steps=10**9, seed=3)
  sd = torch.load(f"runs/{a.run}/ckpt/{a.ckpt}.pt", map_location=dev, weights_only=False)
  pol = Policy(OBS_DIM, cfg["hist"], GOAL_FRAME_DIM * cfg["goal_k"]).to(dev).eval()
  pol.load_state_dict(sd["policy"])
  wm = WorldModel(sim.kin, hidden=(cfg["wm_hidden"],) * 3).to(dev).eval()
  wm.load_state_dict(sd["wm"])
  n = len(specs)
  sim.reset(torch.arange(n, device=dev), frames=frames, perturb=False)
  xi = sim.context().clone()
  s0 = sim.state()
  hist0 = sim.obs_hist.clone()
  sim_states, acts = [s0], []
  for _ in range(a.steps):
    act = pol(sim.obs_hist, sim.goal())
    sim.step(act)
    acts.append(act)
    sim_states.append(sim.state())
    sim.push_obs()
  # open loop: same actions as the simulator
  s = s0
  open_states = [s]
  for act in acts:
    s = wm(s, act, xi)
    open_states.append(s)
  # closed loop: policy acts on predicted states (noise-free observations, as in evaluation)
  s, f, hist = s0, frames.clone(), hist0
  prev_a = hist0[:, -1, 2 * 29 + 6 :]
  closed_states = [s]
  for _ in range(a.steps):
    act = pol(hist, goal_obs(lib, s, f, cfg["goal_k"]))
    s = wm(s, act, xi)
    f = torch.minimum(f + 1, lib.last_frame[f])
    o = proprio_obs(sim.kin, s, xi, act)
    hist = torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
    closed_states.append(s)
    prev_a = act
  S, O, C = (torch.stack(x, 1) for x in (sim_states, open_states, closed_states))  # (n, T+1, 71)
  ji = [13 + JOINT_NAMES.index(j) for j in JOINTS]
  nominal = wm.nominal
  out = dict(run=a.run, ckpt=a.ckpt, it=sd.get("it"), steps=a.steps, dt=0.02, joints=list(JOINTS), clips=[])
  for i, (name, off) in enumerate(specs):
    e_open = ((state_error(O[i], S[i]) / nominal) ** 2).mean(-1)
    e_closed = ((state_error(C[i], S[i]) / nominal) ** 2).mean(-1)
    rd = lambda x: [round(float(v), 4) for v in x]  # noqa: E731
    out["clips"].append(dict(
      name=name, start_frame=off,
      root_z=dict(sim=rd(S[i, :, 2]), open=rd(O[i, :, 2]), closed=rd(C[i, :, 2])),
      root_x=dict(sim=rd(S[i, :, 0] - S[i, 0, 0]), open=rd(O[i, :, 0] - S[i, 0, 0]), closed=rd(C[i, :, 0] - S[i, 0, 0])),
      joints={j: dict(sim=rd(S[i, :, k]), open=rd(O[i, :, k]), closed=rd(C[i, :, k])) for j, k in zip(JOINTS, ji)},
      err_open=rd(e_open), err_closed=rd(e_closed),
    ))
  out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
  Path("results/wm_rollout.json").write_text(json.dumps(out))
  print({c["name"]: (c["err_open"][8], c["err_open"][-1], c["err_closed"][-1]) for c in out["clips"]})


if __name__ == "__main__":
  main()
