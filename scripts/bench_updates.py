"""Times the SuperMimic update phases and prints a kernel-level profile."""

import sys
import time

import torch

from supermimic.task import body_states
from supermimic.train_supermimic import SuperMimic, parse_config


def timed(fn, n=5):
  fn()
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(n):
    fn()
  torch.cuda.synchronize()
  return (time.time() - t0) / n


def main():
  cfg = parse_config(sys.argv[1:] + ["--run_dir", "runs/_bench", "--total_iters", "0", "--final_eval", "false"])
  sm = SuperMimic(cfg)
  for _ in range(30):
    sm.collect_step()
  print(f"collect step      {timed(sm.collect_step, 10) * 1000:.1f} ms")
  print(f"wm_update(1)      {timed(lambda: sm.wm_update(1)) * 1000:.1f} ms")
  print(f"policy_update     {timed(sm.policy_update, 3) * 1000:.1f} ms")
  s0, acts, s1, xi = sm.replay.sample_windows(cfg.wm_batch, cfg.wm_k)

  def fk_fb():
    s = s0.clone().requires_grad_(True)
    b = body_states(sm.kin, s)
    sum(x.sum() for x in b).backward()

  print(f"FK fwd+bwd eager  {timed(fk_fb) * 1000:.1f} ms (batch {cfg.wm_batch})")

  def mlp_fb():
    x = torch.randn(cfg.wm_batch * cfg.wm_k, sm.wm.in_norm.mean.numel(), device="cuda")
    sm.wm.net(x).sum().backward()

  print(f"WM MLP fwd+bwd x8 {timed(mlp_fb) * 1000:.1f} ms")
  from torch.profiler import ProfilerActivity, profile

  with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    sm.policy_update()
    torch.cuda.synchronize()
  print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=90))
  ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
  print("cuda kernels launched:", len(ev))


if __name__ == "__main__":
  main()
