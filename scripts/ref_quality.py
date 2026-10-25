"""Kinematic sanity check of the reference motions: how far the retargeted robot sinks into or floats above the floor.

For every frame of the motion library the G1 collision geometry is placed by forward kinematics (reference root
pose + joint angles) and its lowest point is measured. Physically, some part must touch the floor (~0) unless the
robot is airborne (short flight phases in runs/jumps); long floating spans or deep penetration are artifacts of the
kinematics-only retargeting (or of props in the mocap scene). Writes results/ref_quality.json.

  .venv/bin/python scripts/ref_quality.py
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermimic.g1 import JOINT_NAMES, compile_g1_model  # noqa: E402

FLOAT, PEN, LYING = 0.03, -0.03, 0.4  # m


def main():
  m = compile_g1_model()
  d = mujoco.MjData(m)
  fq = int(m.jnt_qposadr[m.joint("floating_base_joint").id])
  jq = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES])
  coll = np.array([g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]])
  cap = m.geom_type[coll] == mujoco.mjtGeom.mjGEOM_CAPSULE
  size = m.geom_size[coll]
  z = np.load(ROOT / "data/motions/lafan1.npz")
  names = [str(n) for n in z["names"]]
  bp, bq, jp = z["body_pos_w"], z["body_quat_w"], z["joint_pos"]
  fps = float(z["fps"])
  clips = []
  for c, name in enumerate(names):
    s, n = int(z["clip_start"][c]), int(z["clip_len"][c])
    low = np.empty(n)
    for i, f in enumerate(range(s, s + n)):
      d.qpos[fq:fq + 3], d.qpos[fq + 3:fq + 7], d.qpos[jq] = bp[f, 0], bq[f, 0], jp[f]
      mujoco.mj_kinematics(m, d)
      zz = d.geom_xpos[coll, 2] - size[:, 0] - cap * np.abs(d.geom_xmat[coll, 8]) * size[:, 1]
      low[i] = zz.min()
    w, x, y, _ = bq[s:s + n, 0].T
    vertical = np.abs(2 * (x * bq[s:s + n, 0, 3] - w * y)) > 0.9  # pelvis x-axis near vertical (heading undefined)
    fl = low > FLOAT
    longest = run = 0
    for v in fl:
      run = run + 1 if v else 0
      longest = max(longest, run)
    clips.append(dict(
      name=name, seconds=round(n / fps, 1),
      float_pct=round(100 * float(fl.mean()), 1), longest_float_s=round(longest / fps, 2),
      max_float_cm=round(100 * float(low.max()), 1),
      pen_pct=round(100 * float((low < PEN).mean()), 1), pen_p5_cm=round(100 * float(np.percentile(low, 5)), 1),
      pen_min_cm=round(100 * float(low.min()), 1),
      lying_pct=round(100 * float((bp[s:s + n, 0, 2] < LYING).mean()), 1),
      vertical_pct=round(100 * float(vertical.mean()), 1),
    ))
  out = dict(thresholds=dict(float_m=FLOAT, penetration_m=PEN, lying_pelvis_m=LYING), geometry="G1 collision geoms",
             clips=clips, generated=time.strftime("%Y-%m-%d %H:%M:%S"))
  (ROOT / "results/ref_quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
  bad = sorted(clips, key=lambda r: r["pen_p5_cm"])[:6]
  print("deepest penetration:", [(r["name"], r["pen_p5_cm"], r["pen_min_cm"]) for r in bad])
  print("longest float:", sorted(((r["longest_float_s"], r["name"]) for r in clips), reverse=True)[:5])


if __name__ == "__main__":
  main()
