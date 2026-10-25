"""Checks torch FK/velocities against MuJoCo on random G1 states."""

import mujoco
import numpy as np
import torch

from supermimic.g1 import JOINT_NAMES, build_kinematics, compile_g1_model
from supermimic.kinematics import forward_kinematics
from supermimic.math3d import quat_angle_sq


def mujoco_reference(m, qpos, qvel):
  d = mujoco.MjData(m)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  mujoco.mj_kinematics(m, d)
  mujoco.mj_comPos(m, d)
  mujoco.mj_comVel(m, d)
  root = 1
  pos = d.xpos[1:].copy()
  quat = d.xquat[1:].copy()
  cvel = d.cvel[1:].copy()
  off = d.subtree_com[root] - pos
  lin = cvel[:, 3:] - np.cross(cvel[:, :3], off)
  return pos, quat, lin, cvel[:, :3].copy()


def test_fk_matches_mujoco():
  m = compile_g1_model()
  kin = build_kinematics(m)
  jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
  vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
  rng = np.random.default_rng(0)
  for _ in range(5):
    qpos = np.zeros(m.nq)
    qpos[:3] = rng.normal(size=3)
    q = rng.normal(size=4)
    qpos[3:7] = q / np.linalg.norm(q)
    qpos[7:] = rng.uniform(-1, 1, size=m.nq - 7)
    qvel = rng.normal(size=m.nv)
    ref = mujoco_reference(m, qpos, qvel)
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)[None]  # noqa: E731
    jp = t(qpos[jadr])
    jv = t(qvel[vadr])
    root_quat = t(qpos[3:7])
    w_body = t(qvel[3:6])
    from supermimic.math3d import quat_apply

    w_world = quat_apply(root_quat, w_body)
    k64 = build_kinematics(m)
    for k, v in k64.__dict__.items():
      if isinstance(v, torch.Tensor) and v.is_floating_point():
        setattr(k64, k, v.double())
    pos, quat, lin, ang = forward_kinematics(k64, t(qpos[:3]), root_quat, jp, t(qvel[:3]), w_world, jv)
    assert np.allclose(pos[0].numpy(), ref[0], atol=1e-6)
    assert quat_angle_sq(quat[0], torch.tensor(ref[1])).max() < 1e-10
    assert np.allclose(lin[0].numpy(), ref[2], atol=1e-6), np.abs(lin[0].numpy() - ref[2]).max()
    assert np.allclose(ang[0].numpy(), ref[3], atol=1e-6)
    from supermimic.kinematics_fast import forward_kinematics as fk_fast

    pos2, quat2, lin2, ang2 = fk_fast(k64, t(qpos[:3]), root_quat, jp, t(qvel[:3]), w_world, jv)
    assert np.allclose(pos2[0].numpy(), ref[0], atol=1e-6)
    assert quat_angle_sq(quat2[0], torch.tensor(ref[1])).max() < 1e-10
    assert np.allclose(lin2[0].numpy(), ref[2], atol=1e-6), np.abs(lin2[0].numpy() - ref[2]).max()
    assert np.allclose(ang2[0].numpy(), ref[3], atol=1e-6)
  assert kin.nb == 30


if __name__ == "__main__":
  test_fk_matches_mujoco()
  print("ok")
