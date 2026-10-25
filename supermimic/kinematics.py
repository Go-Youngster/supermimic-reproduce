"""Differentiable forward kinematics/velocity for the G1 tree (world frame, link origins)."""

import torch

from .g1 import G1Kinematics
from .math3d import quat_apply, quat_from_axis_angle, quat_mul


def forward_kinematics(
  kin: G1Kinematics,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  joint_pos: torch.Tensor,
  root_lin_vel: torch.Tensor | None = None,
  root_ang_vel: torch.Tensor | None = None,
  joint_vel: torch.Tensor | None = None,
):
  """Returns body pos (B,NB,3), quat (B,NB,4) and optionally lin/ang vel (B,NB,3), all world frame."""
  bshape = root_pos.shape[:-1]
  nb = kin.nb
  with_vel = root_lin_vel is not None
  pos = [None] * nb
  quat = [None] * nb
  lin = [None] * nb
  ang = [None] * nb
  pos[0], quat[0] = root_pos, root_quat
  if with_vel:
    lin[0], ang[0] = root_lin_vel, root_ang_vel

  for lv, idx in zip(kin.levels, kin.levels_py):
    par = [kin.parent[i] for i in idx]
    p_pos = torch.stack([pos[i] for i in par], dim=-2)
    p_quat = torch.stack([quat[i] for i in par], dim=-2)
    q = joint_pos[..., kin.jnt_qidx[lv]]
    axis = kin.jnt_axis[lv].expand(bshape + (len(idx), 3))
    b_pos = p_pos + quat_apply(p_quat, kin.body_pos[lv].expand_as(p_pos))
    b_quat0 = quat_mul(p_quat, kin.body_quat[lv].expand_as(p_quat))
    b_quat = quat_mul(b_quat0, quat_from_axis_angle(axis, q))
    jp = kin.jnt_pos[lv].expand_as(p_pos)
    anchor = b_pos + quat_apply(b_quat0, jp)
    b_pos = anchor - quat_apply(b_quat, jp)
    for k, i in enumerate(idx):
      pos[i] = b_pos[..., k, :]
      quat[i] = b_quat[..., k, :]
    if with_vel:
      p_lin = torch.stack([lin[i] for i in par], dim=-2)
      p_ang = torch.stack([ang[i] for i in par], dim=-2)
      axis_w = quat_apply(b_quat, axis)
      w_j = axis_w * joint_vel[..., kin.jnt_qidx[lv]].unsqueeze(-1)
      b_ang = p_ang + w_j
      b_lin = p_lin + torch.cross(p_ang, b_pos - p_pos, dim=-1) + torch.cross(w_j, b_pos - anchor, dim=-1)
      for k, i in enumerate(idx):
        lin[i] = b_lin[..., k, :]
        ang[i] = b_ang[..., k, :]

  out = (torch.stack(pos, dim=-2), torch.stack(quat, dim=-2))
  if with_vel:
    out = out + (torch.stack(lin, dim=-2), torch.stack(ang, dim=-2))
  return out
