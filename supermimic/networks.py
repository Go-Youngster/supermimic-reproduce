"""World model, policy (goal encoder + shared decoder) and critic networks (Table A.4)."""

import torch
import torch.nn as nn

from .g1 import NJ, G1Kinematics
from .math3d import quat_apply, quat_apply_inv, quat_conj, quat_exp, quat_log, quat_mul, quat_normalize
from .task import base_features, body_frame_features, body_states, split_state

DELTA_DIM = 12 + 2 * NJ
# Nominal per-component errors used to normalize state prediction errors (pos, rot, lin vel, ang vel, q, qd).
NOMINAL_ERR = [0.02] * 3 + [0.02] * 3 + [0.2] * 3 + [0.5] * 3 + [0.02] * NJ + [0.5] * NJ
AMP = {"bf16": False}


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
  if AMP["bf16"]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return net(x).float()
  return net(x)


class Normalizer(nn.Module):
  """Running mean/std (batch Welford merge)."""

  def __init__(self, dim: int, eps: float = 1e-2, clip: float = 10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))
    self.eps, self.clip = eps, clip

  @torch.no_grad()
  def update(self, x: torch.Tensor):
    x = x.reshape(-1, x.shape[-1]).float()
    n = x.shape[0]
    m, v = x.mean(0), x.var(0, unbiased=False)
    tot = self.count + n
    delta = m - self.mean
    # In-place updates keep buffer addresses stable for captured CUDA graphs.
    self.var.copy_((self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot)
    self.mean.add_(delta * n / tot)
    self.count.copy_(tot)

  @property
  def std(self):
    return torch.sqrt(self.var) + self.eps

  def forward(self, x):
    return ((x - self.mean) / self.std).clamp(-self.clip, self.clip)

  def denorm(self, y):
    return y * self.std + self.mean


def mlp(inp: int, hidden, out: int, act=nn.ELU, layernorm: bool = False, out_gain: float = 1.0) -> nn.Sequential:
  layers, d = [], inp
  for h in hidden:
    layers.append(nn.Linear(d, h))
    if layernorm:
      layers.append(nn.LayerNorm(h))
    layers.append(act())
    d = h
  last = nn.Linear(d, out)
  nn.init.orthogonal_(last.weight, gain=out_gain)
  nn.init.zeros_(last.bias)
  layers.append(last)
  return nn.Sequential(*layers)


def state_delta(s: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
  """Local-frame increment from s to s1 (inverse of WorldModel.integrate)."""
  p, q, v, w, jp, jv = split_state(s)
  p1, q1, v1, w1, jp1, jv1 = split_state(s1)
  return torch.cat(
    (
      quat_apply_inv(q, p1 - p),
      quat_log(quat_mul(quat_conj(q), q1)),
      quat_apply_inv(q, v1 - v),
      quat_apply_inv(q, w1 - w),
      jp1 - jp,
      jv1 - jv,
    ),
    -1,
  )


def integrate(s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
  p, q, v, w, jp, jv = split_state(s)
  return torch.cat(
    (
      p + quat_apply(q, d[..., 0:3]),
      quat_normalize(quat_mul(q, quat_exp(d[..., 3:6]))),
      v + quat_apply(q, d[..., 6:9]),
      w + quat_apply(q, d[..., 9:12]),
      jp + d[..., 12 : 12 + NJ],
      jv + d[..., 12 + NJ :],
    ),
    -1,
  )


def state_error(s_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
  """World-frame state error vector (70): pos, rotation vector, velocities, joints."""
  p, q, v, w, jp, jv = split_state(s)
  ph, qh, vh, wh, jph, jvh = split_state(s_hat)
  return torch.cat((ph - p, quat_log(quat_mul(quat_conj(q), qh)), vh - v, wh - w, jph - jp, jvh - jv), -1)


class WorldModel(nn.Module):
  """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi); reward/obs/termination heads are analytic."""

  def __init__(self, kin: G1Kinematics, hidden=(2048, 2048, 2048), ctx_dim: int = 1 + NJ + 3):
    super().__init__()
    self.kin = kin
    in_dim = 68 + 7 * kin.nb + 2 * NJ + ctx_dim
    self.net = mlp(in_dim, hidden, DELTA_DIM, act=nn.SiLU, layernorm=True, out_gain=0.01)
    self.in_norm = Normalizer(in_dim)
    self.out_norm = Normalizer(DELTA_DIM, eps=1e-6, clip=1e6)
    self.register_buffer("nominal", torch.tensor(NOMINAL_ERR))

  def features(self, s, a, xi, bodies):
    kin = self.kin
    jp = s[..., 13 : 13 + NJ]
    pd_err = kin.default_joint_pos + kin.action_scale * a - xi[..., 1 : 1 + NJ] - jp
    return torch.cat((base_features(kin, s), body_frame_features(kin, s, bodies), a, pd_err, xi), -1)

  def forward(self, s, a, xi, bodies=None):
    if bodies is None:
      bodies = body_states(self.kin, s)
    x = self.in_norm(self.features(s, a, xi, bodies))
    d = self.out_norm.denorm(run_mlp(self.net, x))
    return integrate(s, d)


class Policy(nn.Module):
  """a = dec(enc(g), o_{t-9..t}); enc output is a 32-d tanh latent."""

  def __init__(self, obs_dim: int, hist: int, goal_dim: int, act_dim: int = NJ, latent: int = 32,
               enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256)):
    super().__init__()
    self.obs_norm = Normalizer(obs_dim)
    self.goal_norm = Normalizer(goal_dim)
    self.enc = mlp(goal_dim, enc_hidden, latent, out_gain=1.0)
    self.dec = mlp(latent + obs_dim * hist, dec_hidden, act_dim, out_gain=0.01)

  def forward(self, obs_hist, goal):
    z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
    o = self.obs_norm(obs_hist).flatten(-2)
    return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
  def __init__(self, in_dim: int, hidden=(256, 128)):
    super().__init__()
    self.norm = Normalizer(in_dim)
    self.net = mlp(in_dim, hidden, 1, out_gain=1.0)

  def forward(self, x):
    return run_mlp(self.net, self.norm(x)).squeeze(-1)
