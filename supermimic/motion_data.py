"""Builds a motion library (50 Hz reference states of the tracked bodies) from retargeted G1 clips."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from .g1 import JOINT_NAMES, TRACKED_BODIES, compile_g1_model

_MODEL = None


def _quat_mul(a, b):
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  )


def _quat_conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def _slerp(a, b, t):
  dot = (a * b).sum(-1, keepdims=True)
  b = np.where(dot < 0, -b, b)
  dot = np.abs(dot).clip(max=1.0)
  theta = np.arccos(dot)
  s = np.sin(theta)
  small = s < 1e-6
  w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
  w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
  q = w0 * a + w1 * b
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _axis_angle(q):
  q = np.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  s = np.linalg.norm(v, axis=-1, keepdims=True)
  ang = 2 * np.arctan2(s, q[..., :1])
  return v * np.where(s > 1e-9, ang / np.maximum(s, 1e-12), 2.0)


def resample_clip(raw: np.ndarray, input_fps: float, output_fps: float):
  """raw: (N, 36) = root pos, root quat (xyzw), 29 joints. Returns 50 Hz arrays + velocities."""
  pos_in = raw[:, :3]
  quat_in = raw[:, [6, 3, 4, 5]]
  quat_in = quat_in / np.linalg.norm(quat_in, axis=-1, keepdims=True)
  dof_in = raw[:, 7:]
  n = raw.shape[0]
  duration = (n - 1) / input_fps
  times = np.arange(0, duration, 1.0 / output_fps)
  phase = times / duration
  i0 = np.floor(phase * (n - 1)).astype(np.int64)
  i1 = np.minimum(i0 + 1, n - 1)
  blend = (phase * (n - 1) - i0)[:, None]
  pos = pos_in[i0] * (1 - blend) + pos_in[i1] * blend
  quat = _slerp(quat_in[i0], quat_in[i1], blend)
  dof = dof_in[i0] * (1 - blend) + dof_in[i1] * blend
  dt = 1.0 / output_fps
  lin = np.gradient(pos, dt, axis=0)
  dof_vel = np.gradient(dof, dt, axis=0)
  rel = _quat_mul(quat[2:], _quat_conj(quat[:-2]))
  ang = _axis_angle(rel) / (2 * dt)
  ang = np.concatenate([ang[:1], ang, ang[-1:]], axis=0)
  return pos, quat, lin, ang, dof, dof_vel


def _init_worker():
  global _MODEL
  _MODEL = compile_g1_model()


def body_states(m, pos, quat, lin, ang, dof, dof_vel):
  """Runs MuJoCo kinematics per frame; returns tracked-body world states."""
  d = mujoco.MjData(m)
  jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
  vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
  bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in TRACKED_BODIES]
  root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  T = pos.shape[0]
  out_pos = np.zeros((T, len(bids), 3), np.float32)
  out_quat = np.zeros((T, len(bids), 4), np.float32)
  out_lin = np.zeros((T, len(bids), 3), np.float32)
  out_ang = np.zeros((T, len(bids), 3), np.float32)
  for t in range(T):
    d.qpos[:3] = pos[t]
    d.qpos[3:7] = quat[t]
    d.qpos[jadr] = dof[t]
    d.qvel[:3] = lin[t]
    # Free-joint angular velocity is expressed in the body frame.
    qc = _quat_conj(quat[t])
    v = np.concatenate([[0.0], ang[t]])
    d.qvel[3:6] = _quat_mul(_quat_mul(qc, v), quat[t])[1:]
    d.qvel[vadr] = dof_vel[t]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    xp = d.xpos[bids]
    cv = d.cvel[bids]
    out_pos[t] = xp
    out_quat[t] = d.xquat[bids]
    out_lin[t] = cv[:, 3:] - np.cross(cv[:, :3], d.subtree_com[root] - xp)
    out_ang[t] = cv[:, :3]
  return out_pos, out_quat, out_lin, out_ang


def convert_one(args):
  path, input_fps, output_fps = args
  raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
  pos, quat, lin, ang, dof, dof_vel = resample_clip(raw, input_fps, output_fps)
  bp, bq, bl, ba = body_states(_MODEL, pos, quat, lin, ang, dof, dof_vel)
  return dict(
    name=Path(path).stem,
    joint_pos=dof.astype(np.float32),
    joint_vel=dof_vel.astype(np.float32),
    body_pos_w=bp,
    body_quat_w=bq,
    body_lin_vel_w=bl,
    body_ang_vel_w=ba,
  )


def build_library(files, out_path, input_fps=30.0, output_fps=50.0, workers=32):
  files = sorted(files)
  with Pool(min(workers, len(files)), initializer=_init_worker) as pool:
    clips = pool.map(convert_one, [(str(f), input_fps, output_fps) for f in files], chunksize=1)
  lens = np.array([c["joint_pos"].shape[0] for c in clips], np.int64)
  starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
  keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
  arrays = {k: np.concatenate([c[k] for c in clips], axis=0) for k in keys}
  names = np.array([c["name"] for c in clips])
  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, names=names, clip_start=starts, clip_len=lens, fps=output_fps, body_names=np.array(TRACKED_BODIES), **arrays)
  info = dict(clips=len(clips), frames=int(lens.sum()), hours=float(lens.sum() / output_fps / 3600))
  Path(str(out_path) + ".json").write_text(json.dumps(info, indent=1))
  return info


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv-dir", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--glob", default="*.csv")
  ap.add_argument("--input-fps", type=float, default=30.0)
  ap.add_argument("--workers", type=int, default=32)
  a = ap.parse_args()
  files = sorted(Path(a.csv_dir).glob(a.glob))
  print(build_library(files, a.out, a.input_fps, 50.0, a.workers))
