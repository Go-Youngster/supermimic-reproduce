"""Post-processing jobs runnable by the scheduler (which always passes --run_dir runs/<job>).

Runs one of the analysis scripts and then marks the job directory as finished.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TASKS = {
  "push": "scripts/push_test.py",
  "rollout": "scripts/wm_rollout.py",
  "speed": "scripts/bench_speed.py",
}


def main():
  # allow_abbrev=False: otherwise a task argument such as `--run X` is taken as `--run_dir X`.
  ap = argparse.ArgumentParser(allow_abbrev=False)
  ap.add_argument("--run_dir", required=True)
  ap.add_argument("--task", choices=sorted(TASKS), required=True)
  a, rest = ap.parse_known_args()
  rc = subprocess.call([sys.executable, TASKS[a.task], *rest])
  d = Path(a.run_dir)
  d.mkdir(parents=True, exist_ok=True)
  (d / "status.json").write_text(json.dumps(dict(state="finished" if rc == 0 else "failed", task=a.task, rc=rc, ts=time.time())))
  sys.exit(rc)


if __name__ == "__main__":
  main()
