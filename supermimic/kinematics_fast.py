"""Differentiable forward kinematics/velocity for the G1 tree (world frame, link origins).

Body orientations are composed with pointer jumping (log2(depth) rounds over all bodies), and positions
and velocities are sums over ancestors (one matmul with the ancestor matrix), so the whole computation
is a handful of fused GPU kernels instead of a per-body loop.
"""

import torch

from .g1 import G1Kinematics
from .math3d import quat_apply, quat_from_axis_angle, quat_mul


class TreeTables:
  def __init__(self, kin: G1Kinematics):
    nb = kin.nb
    par = list(kin.parent)
    assert float(kin.jnt_pos.abs().max()) == 0.0, "joint anchors must coincide with body origins"
    dev = kin.body_pos.device
    virt = nb  # virtual identity node above the root
    anc = [virt if p < 0 else p for p in par] + [virt]
    jumps = []
    cur = anc[:]
    depth = 0
    for b in range(nb):
      d, x = 0, b
      while par[x] >= 0:
        x, d = par[x], d + 1
      depth = max(depth, d)
    for _ in range((depth + 1).bit_length()):
      jumps.append(torch.tensor(cur, device=dev))
      cur = [cur[c] for c in cur]
    A = torch.zeros(nb, nb, device=dev, dtype=kin.body_pos.dtype)  # A[b, a] = 1 if a is b or an ancestor of b (root excluded)
    for b in range(nb):
      x = b
      while x > 0:
        A[b, x] = 1.0
        x = par[x]
    self.jumps = jumps
    self.A = A
    self.parent = torch.tensor([max(p, 0) for p in par], device=dev)
    self.qidx = kin.jnt_qidx.clamp_min(0)
    self.has_joint = (kin.jnt_qidx >= 0).to(kin.body_pos.dtype)


_TABLES: dict = {}


def tables(kin: G1Kinematics) -> TreeTables:
  key = (id(kin), kin.body_pos.device, kin.body_pos.dtype)
  if key not in _TABLES:
    _TABLES[key] = TreeTables(kin)
  return _TABLES[key]


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
  T = tables(kin)
  nb = kin.nb
  bshape = root_pos.shape[:-1]
  q = joint_pos[..., T.qidx] * T.has_joint  # (..., NB)
  axis = kin.jnt_axis.expand(bshape + (nb, 3))
  local = quat_mul(kin.body_quat.expand(bshape + (nb, 4)), quat_from_axis_angle(axis, q))
  ident = torch.zeros(bshape + (1, 4), device=q.device, dtype=q.dtype)
  ident[..., 0] = 1.0
  G = torch.cat((root_quat.unsqueeze(-2), local[..., 1:, :], ident), -2)  # (..., NB+1, 4)
  for j in T.jumps:
    G = quat_mul(G[..., j, :], G)
  quat = G[..., :nb, :]
  # Each body's offset in its parent's frame, rotated to world; summed over ancestors.
  u = quat_apply(quat[..., T.parent, :], kin.body_pos.expand(bshape + (nb, 3)))
  u = u * (torch.arange(nb, device=q.device) > 0).unsqueeze(-1).to(u.dtype)
  rel = torch.einsum("ba,...ac->...bc", T.A, u)
  pos = root_pos.unsqueeze(-2) + rel
  if root_lin_vel is None:
    return pos, quat
  qd = joint_vel[..., T.qidx] * T.has_joint
  w_j = quat_apply(quat, axis) * qd.unsqueeze(-1)  # joint angular velocity contributions (world)
  w_sum = torch.einsum("ba,...ac->...bc", T.A, w_j)
  c = torch.einsum("ba,...ac->...bc", T.A, torch.cross(w_j, rel, dim=-1))
  wr = root_ang_vel.unsqueeze(-2).expand_as(rel)
  lin = root_lin_vel.unsqueeze(-2) + torch.cross(wr, rel, dim=-1) + torch.cross(w_sum, rel, dim=-1) - c
  ang = root_ang_vel.unsqueeze(-2) + w_sum
  return pos, quat, lin, ang
