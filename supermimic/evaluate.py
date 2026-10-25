"""Checkpoint evaluation: every motion plays once from its first frame (no DR, no noise, friction 1.0)."""

import torch

from .motion_lib import MotionLibrary
from .sim_env import TrackingSim
from .task import R_MAX


class Evaluator:
  def __init__(self, lib: MotionLibrary, device: str, max_envs: int = 4096):
    self.lib = lib
    self.n_envs = min(lib.num_clips, max_envs)
    self.sim = TrackingSim(lib, self.n_envs, device, train=False, max_episode_steps=10**9, seed=123)

  @torch.no_grad()
  def run(self, act_fn) -> dict:
    lib, sim, dev = self.lib, self.sim, self.sim.device
    rets, fails, lens = [], [], []
    for c0 in range(0, lib.num_clips, self.n_envs):
      clips = torch.arange(c0, min(c0 + self.n_envs, lib.num_clips), device=dev)
      n = clips.numel()
      ids = torch.arange(self.n_envs, device=dev)
      starts = lib.clip_start[clips]
      frames = torch.cat((starts, starts[:1].expand(self.n_envs - n))) if n < self.n_envs else starts
      sim.reset(ids, frames=frames, perturb=False)
      alive = torch.ones(self.n_envs, dtype=torch.bool, device=dev)
      alive[n:] = False
      ret = torch.zeros(self.n_envs, device=dev)
      failed = torch.zeros(self.n_envs, dtype=torch.bool, device=dev)
      horizon = int(lib.clip_len[clips].max())
      for t in range(horizon - 1):
        a = act_fn(sim.obs_hist, sim.goal())
        out = sim.step(a)
        ret += out["reward"] * alive
        failed |= out["terminated"] & alive
        alive &= ~(out["terminated"] | out["clip_end"])
        if t % 50 == 0 and not bool(alive.any()):
          break
        sim.push_obs()
      rets.append(ret[:n])
      fails.append(failed[:n])
      lens.append(lib.clip_len[clips])
    ret, failed, ln = torch.cat(rets), torch.cat(fails), torch.cat(lens).float()
    norm = ret / (R_MAX * (ln - 1))
    return dict(
      norm_return=float(norm.mean()),
      norm_return_frames=float(ret.sum() / (R_MAX * (ln - 1).sum())),
      failed=int(failed.sum()),
      num_motions=int(ln.numel()),
      per_clip=[round(float(x), 4) for x in norm],
      clip_names=list(lib.names),
      failed_clips=[lib.names[i] for i in failed.nonzero().squeeze(-1).tolist()],
    )
