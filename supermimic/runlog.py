"""Run bookkeeping: JSONL metrics, checkpoints, status file read by the scheduler."""

import json
import os
import time
from pathlib import Path

import torch


class RunLogger:
  def __init__(self, run_dir: str, config: dict, kind: str, extra_meta: dict | None = None):
    self.dir = Path(run_dir)
    self.dir.mkdir(parents=True, exist_ok=True)
    (self.dir / "ckpt").mkdir(exist_ok=True)
    cfg_path = self.dir / "config.json"
    if not cfg_path.exists():
      meta = dict(kind=kind, created=time.strftime("%Y-%m-%d %H:%M:%S"), gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                  gpu_name=torch.cuda.get_device_name(0), **(extra_meta or {}))
      cfg_path.write_text(json.dumps(dict(meta=meta, config=config), indent=1, ensure_ascii=False))
    self.metrics = open(self.dir / "metrics.jsonl", "a")
    self.evals = open(self.dir / "eval.jsonl", "a")

  def log(self, rec: dict):
    rec["ts"] = time.time()
    self.metrics.write(json.dumps(rec) + "\n")
    self.metrics.flush()

  def log_eval(self, rec: dict):
    rec["ts"] = time.time()
    self.evals.write(json.dumps(rec) + "\n")
    self.evals.flush()

  def status(self, **kw):
    kw["ts"] = time.time()
    tmp = self.dir / "status.json.tmp"
    tmp.write_text(json.dumps(kw))
    tmp.replace(self.dir / "status.json")

  def save(self, state: dict, name: str = "latest"):
    tmp = self.dir / "ckpt" / f"{name}.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(self.dir / "ckpt" / f"{name}.pt")

  def load(self, name: str = "latest"):
    p = self.dir / "ckpt" / f"{name}.pt"
    return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None
