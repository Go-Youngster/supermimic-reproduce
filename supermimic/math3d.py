"""Differentiable quaternion helpers (wxyz convention, batched over leading dims)."""

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    dim=-1,
  )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  w = q[..., :1]
  xyz = q[..., 1:]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_conj(q), v)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  half = 0.5 * angle.unsqueeze(-1)
  return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def quat_exp(rotvec: torch.Tensor) -> torch.Tensor:
  """Rotation vector -> unit quaternion, smooth at zero."""
  theta_sq = (rotvec * rotvec).sum(-1, keepdim=True)
  theta = torch.sqrt(theta_sq + 1e-12)
  half = 0.5 * theta
  # sin(theta/2)/theta with a Taylor branch near zero.
  small = theta_sq < 1e-6
  k = torch.where(small, 0.5 - theta_sq / 48.0, torch.sin(half) / theta)
  return torch.cat((torch.cos(half), rotvec * k), dim=-1)


def quat_log(q: torch.Tensor) -> torch.Tensor:
  """Unit quaternion -> rotation vector (shortest arc)."""
  q = torch.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  s = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
  angle = 2.0 * torch.atan2(s, q[..., :1])
  return v * (angle / s)


def quat_angle_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Squared rotation angle between a and b (smooth version of quat_error_magnitude**2)."""
  d = quat_mul(quat_conj(a), b)
  s = torch.sqrt((d[..., 1:] ** 2).sum(-1) + 1e-12)
  ang = 2.0 * torch.atan2(s, d[..., 0].abs())
  return ang * ang


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  sy, cy = 2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
  # Heading is undefined when the x-axis is vertical; nudge to keep atan2 gradients finite.
  cy = torch.where((cy.abs() < 1e-6) & (sy.abs() < 1e-6), torch.full_like(cy, 1e-6), cy)
  yaw = torch.atan2(sy, cy)
  zero = torch.zeros_like(yaw)
  half = 0.5 * yaw
  return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def quat_to_mat2(q: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix, flattened (6D rotation)."""
  w, x, y, z = q.unbind(-1)
  c0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)
  c1 = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), -1)
  return torch.stack((c0, c1), dim=-1).flatten(-2)


GRAVITY_W = (0.0, 0.0, -1.0)


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
  g = torch.tensor(GRAVITY_W, device=q.device, dtype=q.dtype).expand(q.shape[:-1] + (3,))
  return quat_apply_inv(q, g)
