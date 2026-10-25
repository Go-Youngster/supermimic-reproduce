"""Push-recovery test (paper §4.3): hold a static pose, apply a root velocity push, measure settling.

For each policy checkpoint and push speed, pushes are applied in several horizontal directions with
several trials. A trial "settles" once the root horizontal speed stays below 0.2 m/s; it "falls" if the
training termination triggers. Writes results/push_test.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from supermimic.g1 import compile_g1_model
from supermimic.motion_data import body_states, resample_clip
from supermimic.motion_lib import MotionLibrary
from supermimic.networks import Policy
from supermimic.sim_env import TrackingSim
from supermimic.task import GOAL_FRAME_DIM, OBS_DIM

STATIC = Path("data/motions/static_stand.npz")


def make_static_clip(src_csv: str, seconds: float = 12.0):
  raw = np.loadtxt(src_csv, delimiter=",", dtype=np.float64)[:1]
  raw = np.repeat(raw, int(seconds * 30) + 1, axis=0)
  pos, quat, lin, ang, dof, dof_vel = resample_clip(raw, 30.0, 50.0)
  lin[:] = 0
  ang[:] = 0
  dof_vel[:] = 0
  bp, bq, bl, ba = body_states(compile_g1_model(), pos, quat, lin, ang, dof, dof_vel)
  from supermimic.g1 import TRACKED_BODIES

  n = dof.shape[0]
  np.savez(STATIC, names=np.array(["static_stand"]), clip_start=np.array([0]), clip_len=np.array([n]), fps=50.0,
           body_names=np.array(TRACKED_BODIES), joint_pos=dof.astype(np.float32), joint_vel=dof_vel.astype(np.float32),
           body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=bl, body_ang_vel_w=ba)


def load_policy(run: str, dev: str, which: str = "best") -> Policy:
  sd = torch.load(f"runs/{run}/ckpt/{which}.pt", map_location=dev, weights_only=False)
  cfg = json.loads(Path(f"runs/{run}/config.json").read_text())["config"]
  pol = Policy(OBS_DIM, cfg["hist"], GOAL_FRAME_DIM * cfg["goal_k"]).to(dev)
  pol.load_state_dict(sd["policy"] if "policy" in sd else sd["actor"])
  return pol.eval()


@torch.no_grad()
def run_push(sim: TrackingSim, policy: Policy, speeds, dirs: int, trials: int, push_t: float = 2.0, after: float = 8.0):
  dev = sim.device
  n = sim.num_envs
  per = dirs * trials
  assert n == len(speeds) * per
  speed = torch.tensor(np.repeat(speeds, per), device=dev, dtype=torch.float32)
  ang = torch.tensor(np.tile(np.repeat(np.arange(dirs) * 2 * np.pi / dirs, trials), len(speeds)), device=dev, dtype=torch.float32)
  ids = torch.arange(n, device=dev)
  sim.reset(ids, frames=torch.zeros(n, dtype=torch.long, device=dev), perturb=False)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  settled_at = torch.full((n,), float("nan"), device=dev)
  calm_since = torch.full((n,), float("nan"), device=dev)
  steps_pre, steps_post = int(push_t * 50), int(after * 50)
  for t in range(steps_pre + steps_post):
    if t == steps_pre:
      fq, fv = sim.fv, sim.fv
      qvel = sim.sim.data.qvel
      push = torch.stack((torch.cos(ang), torch.sin(ang)), -1) * speed[:, None]
      qvel[:, sim.fv[0:2]] += push
    a = policy(sim.obs_hist, sim.goal())
    out = sim.step(a)
    sim.frames.clamp_(max=int(sim.lib.last_frame[0]) - 2)  # hold the static reference indefinitely
    fell |= out["terminated"]
    if t >= steps_pre:
      v = out["next_state"][:, 7:9].norm(dim=-1)
      tt = (t - steps_pre + 1) / 50.0
      calm = v < 0.2
      calm_since = torch.where(calm & torch.isnan(calm_since), torch.full_like(calm_since, tt), calm_since)
      calm_since = torch.where(~calm, torch.full_like(calm_since, float("nan")), calm_since)
    sim.push_obs()
  settled_at = torch.where(fell, torch.full_like(calm_since, float("nan")), calm_since)
  res = {}
  for i, s in enumerate(speeds):
    sl = slice(i * per, (i + 1) * per)
    st = settled_at[sl]
    ok = ~torch.isnan(st)
    res[str(s)] = dict(fall_rate=float(fell[sl].float().mean()), settle_rate=float(ok.float().mean()),
                       settle_median_s=float(st[ok].median()) if bool(ok.any()) else None)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", default="sm_lafan1_main,sm_h32_nocritic")
  ap.add_argument("--speeds", default="0.5,1,2,3,4")
  ap.add_argument("--dirs", type=int, default=8)
  ap.add_argument("--trials", type=int, default=4)
  ap.add_argument("--which", default="best")
  a = ap.parse_args()
  dev = "cuda:0"
  if not STATIC.exists():
    make_static_clip("data/raw/lafan1_g1/g1/walk1_subject1.csv")
  speeds = [float(x) for x in a.speeds.split(",")]
  lib = MotionLibrary(str(STATIC), dev)
  sim = TrackingSim(lib, len(speeds) * a.dirs * a.trials, dev, train=False, max_episode_steps=10**9, seed=7)
  out = {"speeds": speeds, "dirs": a.dirs, "trials": a.trials, "pose": "walk1_subject1 首帧静止站立", "results": {}}
  for run in [r for r in a.runs.split(",") if r]:
    if not Path(f"runs/{run}/ckpt/{a.which}.pt").exists():
      continue
    pol = load_policy(run, dev, a.which)
    out["results"][run] = run_push(sim, pol, speeds, a.dirs, a.trials)
    print(run, out["results"][run], flush=True)
  out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
  Path("results/push_test.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
  main()
