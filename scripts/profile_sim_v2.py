"""Profiles simulator components (physics step, post-step reward, reset, observation)."""

import sys
import time

import torch

from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import TrackingSim


def timeit(fn, n=20):
  fn()
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(n):
    fn()
  torch.cuda.synchronize()
  return (time.time() - t0) / n * 1000


def main(num_envs: int = 4096, compile_: int = 1):
  dev = "cuda:0"
  lib = MotionLibrary("data/motions/lafan1.npz", dev)
  t0 = time.time()
  sim = TrackingSim(lib, num_envs, dev, compile=bool(compile_))
  print(f"build {time.time() - t0:.1f}s")
  a = torch.zeros(num_envs, 29, device=dev)
  t0 = time.time()
  sim.step(a)
  torch.cuda.synchronize()
  print(f"first step (compile) {time.time() - t0:.1f}s")
  print(f"physics_step   {timeit(lambda: sim.physics_step(a)):.1f} ms")
  print(f"sim.step x1    {timeit(lambda: sim.sim.step()):.1f} ms")
  print(f"state()        {timeit(lambda: sim.state()):.2f} ms")
  s1 = sim.state()
  print(f"post (rew/term){timeit(lambda: sim._post(sim.kin, lib, s1, sim.frames, a, a)):.2f} ms")
  print(f"full step      {timeit(lambda: sim.step(a)):.1f} ms")
  for k in (64, 512, 4096):
    ids = torch.arange(k, device=dev)
    print(f"reset {k:5d}    {timeit(lambda: sim.reset(ids), 5):.1f} ms")
  print(f"push_obs       {timeit(lambda: sim.push_obs()):.2f} ms")
  print(f"goal           {timeit(lambda: sim.goal()):.2f} ms")
  n_done, steps = 0, 100
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(steps):
    out = sim.step(a)
    done = out["terminated"] | out["clip_end"] | out["timeout"]
    n_done += int(done.sum())
    sim.finish_episodes(done)
    sim.push_obs()
  torch.cuda.synchronize()
  dt = time.time() - t0
  print(f"loop: {steps * num_envs / dt:,.0f} env-steps/s ({dt / steps * 1000:.1f} ms/step), episodes ended {n_done}")
  print("reward mean", out["reward"].mean().item(), "max mem GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
  main(*(int(x) for x in sys.argv[1:]))
