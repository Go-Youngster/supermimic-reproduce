"""Experiment scheduler: runs queued jobs on a fixed set of GPUs, one job per GPU, with resume-on-crash.

The queue lives in experiments/queue.json (re-read every poll, so jobs can be appended while it runs):
  [{"name": "sm_lafan1", "module": "supermimic.train_supermimic", "args": ["--max_hours", "12"], "gpu": 4}, ...]
State is written to runs/scheduler_state.json and a log to runs/scheduler.log.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "experiments" / "queue.json"
STATE = ROOT / "runs" / "scheduler_state.json"
LOG = ROOT / "runs" / "scheduler.log"
PY = ROOT / ".venv" / "bin" / "python"


def log(msg: str):
  line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
  print(line, flush=True)
  with open(LOG, "a") as f:
    f.write(line + "\n")


def load_state() -> dict:
  try:
    return json.loads(STATE.read_text())
  except Exception:
    return {"jobs": {}}


def save_state(st: dict):
  tmp = STATE.with_suffix(".tmp")
  tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False))
  tmp.replace(STATE)


def run_finished(name: str) -> bool:
  try:
    return json.loads((ROOT / "runs" / name / "status.json").read_text()).get("state") == "finished"
  except Exception:
    return False


def pid_alive(pid: int, name: str) -> bool:
  try:
    return name.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
  except Exception:
    return False


class Adopted:
  """A job process started by an earlier scheduler instance; exit code is unknown."""

  def __init__(self, pid: int, name: str):
    self.pid, self.name = pid, name

  def poll(self):
    return None if pid_alive(self.pid, self.name) else (0 if run_finished(self.name) else -1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gpus", default="4,5,6,7")
  ap.add_argument("--poll", type=float, default=30.0)
  ap.add_argument("--max_retries", type=int, default=5)
  a = ap.parse_args()
  gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
  (ROOT / "runs").mkdir(exist_ok=True)
  st = load_state()
  procs: dict = {}
  log(f"scheduler start, gpus={gpus}")
  for name, j in st["jobs"].items():
    if j.get("status") == "running":
      if j.get("pid") and pid_alive(j["pid"], name):
        procs[name] = Adopted(j["pid"], name)
        log(f"adopted running job {name} (pid {j['pid']}, GPU {j.get('gpu')})")
      else:
        j["status"] = "queued"
  while True:
    try:
      queue = json.loads(QUEUE.read_text()) if QUEUE.exists() else []
    except Exception as e:
      log(f"bad queue file: {e}")
      queue = []
    for job in queue:
      j = st["jobs"].setdefault(job["name"], {"status": "queued", "retries": 0})
      j["spec"] = job
      if job.get("cancel") and j["status"] in ("queued",):
        j["status"] = "cancelled"
    # reap finished processes
    for name, p in list(procs.items()):
      rc = p.poll()
      if rc is None:
        continue
      j = st["jobs"][name]
      del procs[name]
      done_ok = run_finished(name) if "script" not in j["spec"] else True
      if rc == 0 and done_ok:
        j.update(status="done", end=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"job {name} finished on GPU {j.get('gpu')}")
      else:
        j["retries"] = j.get("retries", 0) + 1
        if j["retries"] > j["spec"].get("max_retries", a.max_retries):
          j["status"] = "failed"
          log(f"job {name} failed (rc={rc}), giving up after {j['retries'] - 1} retries")
        else:
          j["status"] = "queued"
          j["not_before"] = time.time() + 60
          log(f"job {name} exited rc={rc}; will resume from its latest checkpoint (retry {j['retries']})")
    busy = {st["jobs"][n].get("gpu") for n in procs}
    for job in queue:
      name = job["name"]
      j = st["jobs"][name]
      if j["status"] != "queued" or name in procs or time.time() < j.get("not_before", 0):
        continue
      deps = job.get("after", [])
      if any(st["jobs"].get(d, {}).get("status") != "done" for d in deps):
        continue
      want = job.get("gpu")
      free = [g for g in gpus if g not in busy]
      if want is not None:
        free = [g for g in free if g == want]
      if not free:
        continue
      g = free[0]
      run_dir = ROOT / "runs" / name
      run_dir.mkdir(parents=True, exist_ok=True)
      if "script" in job:
        cmd = [str(PY), job["script"], *job.get("args", [])]
      else:
        cmd = [str(PY), "-m", job.get("module", "supermimic.train_supermimic"), "--run_dir", f"runs/{name}", *job.get("args", [])]
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
      out = open(run_dir / "train.log", "a")
      out.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {g}: {' '.join(cmd)}\n")
      out.flush()
      # New session: jobs survive a scheduler restart or the tmux pane closing.
      procs[name] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
      j.update(status="running", gpu=g, pid=procs[name].pid, start=time.strftime("%Y-%m-%d %H:%M:%S"))
      busy.add(g)
      log(f"start {name} on GPU {g} (pid {procs[name].pid})")
    save_state(st)
    time.sleep(a.poll)


if __name__ == "__main__":
  sys.exit(main())
