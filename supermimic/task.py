"""Analytic task heads shared by the simulator and the world model.

State layout (71): root pos (3, env frame) | root quat wxyz (4) | root lin vel world (3) |
root ang vel world (3) | joint pos (29) | joint vel (29).
Sim context xi (33): foot friction (1) | encoder bias (29) | torso COM offset (3).
"""

import torch

from .g1 import NJ, G1Kinematics
from .kinematics_fast import forward_kinematics
from .math3d import (
  projected_gravity,
  quat_angle_sq,
  quat_apply,
  quat_apply_inv,
  quat_conj,
  quat_mul,
  quat_to_mat2,
  yaw_quat,
)
from .motion_lib import MotionLibrary

STATE_DIM = 13 + 2 * NJ
CTX_DIM = 1 + NJ + 3
OBS_DIM = 3 * NJ + 6
GOAL_FRAME_DIM = 2 * NJ + 6 + 3 + 3 + 1
DT = 0.02
R_MAX = (4 * 1.0 + 2 * 0.5) * DT  # per-step upper bound of the tracking reward

OBS_NOISE = torch.tensor([0.01] * NJ + [0.5] * NJ + [0.2] * 3 + [0.05] * 3 + [0.0] * NJ)
GOAL_ORI_NOISE = 0.05


def split_state(s: torch.Tensor):
  return s[..., 0:3], s[..., 3:7], s[..., 7:10], s[..., 10:13], s[..., 13 : 13 + NJ], s[..., 13 + NJ :]


def body_states(kin: G1Kinematics, s: torch.Tensor):
  p, q, v, w, jp, jv = split_state(s)
  return forward_kinematics(kin, p, q, jp, v, w, jv)


def reference(lib: MotionLibrary, frames: torch.Tensor) -> dict:
  return dict(
    pos=lib.body_pos[frames],
    quat=lib.body_quat[frames],
    lin=lib.body_lin_vel[frames],
    ang=lib.body_ang_vel[frames],
    jpos=lib.joint_pos[frames],
    jvel=lib.joint_vel[frames],
  )


def tracking_reward(kin, bodies, ref, action, prev_action, joint_pos, with_terms=False):
  """BeyondMimic reward (Table A.2) with pelvis as root; bodies = FK outputs (all robot bodies)."""
  pos, quat, lin, ang = (x[..., kin.track_idx, :] for x in bodies)
  r_pos, r_quat = pos[..., 0, :], quat[..., 0, :]
  g_pos, g_quat = ref["pos"][..., 0, :], ref["quat"][..., 0, :]
  e_root_pos = ((g_pos - r_pos) ** 2).sum(-1)
  e_root_ori = quat_angle_sq(g_quat, r_quat)
  delta_pos = torch.cat((r_pos[..., :2], g_pos[..., 2:3]), -1).unsqueeze(-2)
  delta_ori = yaw_quat(quat_mul(r_quat, quat_conj(g_quat))).unsqueeze(-2)
  delta_ori_b = delta_ori.expand(ref["quat"].shape)
  rel_pos = delta_pos + quat_apply(delta_ori_b, ref["pos"] - g_pos.unsqueeze(-2))
  rel_quat = quat_mul(delta_ori_b, ref["quat"])
  e_body_pos = ((rel_pos - pos) ** 2).sum(-1).mean(-1)
  e_body_ori = quat_angle_sq(rel_quat, quat).mean(-1)
  e_body_lin = ((ref["lin"] - lin) ** 2).sum(-1).mean(-1)
  e_body_ang = ((ref["ang"] - ang) ** 2).sum(-1).mean(-1)
  terms = dict(
    root_pos=0.5 * torch.exp(-e_root_pos / 0.3**2),
    root_ori=0.5 * torch.exp(-e_root_ori / 0.4**2),
    body_pos=torch.exp(-e_body_pos / 0.3**2),
    body_ori=torch.exp(-e_body_ori / 0.4**2),
    body_lin_vel=torch.exp(-e_body_lin / 1.0**2),
    body_ang_vel=torch.exp(-e_body_ang / 3.14**2),
    action_rate=-0.1 * ((action - prev_action) ** 2).sum(-1),
    joint_limit=-2.0
    * (
      (joint_pos - kin.soft_limits[:, 1]).clamp_min(0) + (kin.soft_limits[:, 0] - joint_pos).clamp_min(0)
    ).sum(-1),
  )
  r = sum(terms.values()) * DT
  if with_terms:
    return r, {k: v * DT for k, v in terms.items()}
  return r


def termination(s: torch.Tensor, ref: dict, soft_z: float = 0.02, soft_tilt: float = 0.04):
  """Root height / tilt deviation (Appendix A). Returns (hard bool, soft probability)."""
  r_pos, r_quat = s[..., 0:3], s[..., 3:7]
  g_pos, g_quat = ref["pos"][..., 0, :], ref["quat"][..., 0, :]
  ez = (g_pos[..., 2] - r_pos[..., 2]).abs()
  et = (projected_gravity(g_quat)[..., 2] - projected_gravity(r_quat)[..., 2]).abs()
  hard = (ez > 0.25) | (et > 0.8)
  p_z = torch.sigmoid((ez - 0.25) / soft_z)
  p_t = torch.sigmoid((et - 0.8) / soft_tilt)
  soft = 1 - (1 - p_z) * (1 - p_t)
  return hard, soft


def proprio_obs(kin, s, xi, prev_action):
  """Noise-free proprioceptive observation (Table A.1)."""
  p, q, v, w, jp, jv = split_state(s)
  return torch.cat(
    (jp - kin.default_joint_pos + xi[..., 1 : 1 + NJ], jv, quat_apply_inv(q, w), projected_gravity(q), prev_action),
    -1,
  )


def goal_obs(lib: MotionLibrary, s, frames, k: int = 10):
  """K future reference frames; root orientation is expressed relative to the robot root."""
  ff = lib.future_frames(frames, k)  # (B, k)
  r_quat = s[..., 3:7].unsqueeze(-2).expand(ff.shape + (4,))
  g_quat = lib.body_quat[ff, 0]
  rel = quat_to_mat2(quat_mul(quat_conj(r_quat), g_quat))
  g_lin = quat_apply_inv(g_quat, lib.body_lin_vel[ff, 0])
  g_ang = quat_apply_inv(g_quat, lib.body_ang_vel[ff, 0])
  g_h = lib.body_pos[ff, 0, 2:3]
  per = torch.cat((lib.joint_pos[ff], lib.joint_vel[ff], rel, g_lin, g_ang, g_h), -1)
  return per.flatten(-2)


def goal_noise_mask(k: int = 10, device="cpu") -> torch.Tensor:
  m = torch.zeros(k, GOAL_FRAME_DIM, device=device)
  m[:, 2 * NJ : 2 * NJ + 6] = GOAL_ORI_NOISE
  return m.flatten()


def body_frame_features(kin, s, bodies):
  """Translation/yaw-invariant body features in the root frame, plus absolute body heights."""
  p, q = s[..., 0:3], s[..., 3:7]
  qb = q.unsqueeze(-2).expand(bodies[1].shape)
  rel = quat_apply_inv(qb, bodies[0] - p.unsqueeze(-2))
  vel = quat_apply_inv(qb, bodies[2])
  return torch.cat((rel.flatten(-2), bodies[0][..., 2], vel.flatten(-2)), -1)


def base_features(kin, s):
  p, q, v, w, jp, jv = split_state(s)
  return torch.cat(
    (p[..., 2:3], projected_gravity(q), quat_apply_inv(q, v), quat_apply_inv(q, w), jp - kin.default_joint_pos, jv), -1
  )


def critic_features(kin, lib, s, xi, frames, bodies, goal_clean, k: int = 10):
  """Privileged critic input: state features, goal, future root positions relative to the robot."""
  ff = lib.future_frames(frames, k)
  p, q = s[..., 0:3], s[..., 3:7]
  qb = q.unsqueeze(-2).expand(ff.shape + (4,))
  rel = quat_apply_inv(qb, lib.body_pos[ff, 0] - p.unsqueeze(-2)).flatten(-2)
  return torch.cat((base_features(kin, s), body_frame_features(kin, s, bodies), goal_clean, rel, xi), -1)
