"""Throughput of MuJoCo Warp vs the learned world model (Fig. 5a analog).

One step = one 50 Hz control step (four 5 ms MuJoCo Warp substeps; one world-model call).
Writes results/speed_benchmark.json.
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch

from supermimic.g1 import build_kinematics
from supermimic.motion_lib import MotionLibrary
from supermimic.networks import AMP, WorldModel
from supermimic.sim_env import TrackingSim

OUT = Path("results/speed_benchmark.json")


def timed(fn, n, trials=5):
  """Best-of-trials mean time per call (robust to other processes sharing the GPU)."""
  fn()
  best = float("inf")
  for _ in range(trials):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
      fn()
    torch.cuda.synchronize()
    best = min(best, (time.time() - t0) / n)
  return best


def main(sizes=(512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)):
  torch.backends.cuda.matmul.allow_tf32 = True
  torch._dynamo.config.recompile_limit = 64
  dev = "cuda:0"
  lib = MotionLibrary("data/motions/lafan1.npz", dev)
  kin = build_kinematics().to(dev)
  wm = WorldModel(kin).to(dev).eval()
  step = torch.compile(lambda s, a, xi: wm(s, a, xi), dynamic=False)
  res = dict(sizes=[], mujoco_warp=[], wm_tf32=[], wm_bf16=[], gpu=torch.cuda.get_device_name(0),
             note="steps/s, best of 5 trials; one step = one 50 Hz control step")
  for n in sizes:
    row = {}
    try:
      sim = TrackingSim(lib, n, dev, compile=False)
      a = torch.zeros(n, 29, device=dev)
      sim.reset(torch.arange(n, device=dev))
      dt = timed(lambda: sim.physics_step(a), 10)
      row["mujoco_warp"] = n / dt
      s = sim.state()
      xi = sim.context().clone()
      del sim
    except Exception as e:  # e.g. out of memory at the largest batch
      print("mujoco_warp failed at", n, repr(e)[:200], flush=True)
      row["mujoco_warp"] = None
      s = torch.zeros(n, 71, device=dev)
      s[:, 3] = 1.0
      s[:, 2] = 0.75
      xi = torch.zeros(n, 33, device=dev)
    gc.collect()
    torch.cuda.empty_cache()
    a = torch.zeros(n, 29, device=dev)
    with torch.no_grad():
      for key, bf in (("wm_tf32", False), ("wm_bf16", True)):
        AMP["bf16"] = bf
        row[key] = n / timed(lambda: step(s, a, xi), 20)
    res["sizes"].append(n)
    for k in ("mujoco_warp", "wm_tf32", "wm_bf16"):
      res[k].append(row[k])
    print(n, {k: (round(v) if v else v) for k, v in row.items()}, flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
  res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
  OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main(*([tuple(int(x) for x in sys.argv[1].split(","))] if len(sys.argv) > 1 else []))
