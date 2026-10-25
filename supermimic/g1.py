"""Unitree G1 (29 DoF) description shared by the simulator wrapper and the world model."""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# BeyondMimic tracked bodies; index 0 (pelvis) is the root used for root rewards/terminations.
TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)

NJ = len(JOINT_NAMES)
NB_TRACK = len(TRACKED_BODIES)


def compile_g1_model() -> mujoco.MjModel:
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
  from mjlab.entity import Entity

  return Entity(get_g1_robot_cfg()).spec.compile()


@dataclass
class G1Kinematics:
  """Kinematic tree of the robot (bodies after the world body), as torch tensors."""

  body_names: tuple
  parent: list  # parent index within robot bodies (-1 for root)
  levels: list  # list of index tensors grouped by tree depth (root excluded)
  levels_py: tuple  # same as levels, as tuples of ints
  body_pos: torch.Tensor  # (NB, 3) offset in parent frame
  body_quat: torch.Tensor  # (NB, 4)
  jnt_axis: torch.Tensor  # (NB, 3) hinge axis in body frame (zeros for root)
  jnt_pos: torch.Tensor  # (NB, 3) hinge anchor in body frame
  jnt_qidx: torch.Tensor  # (NB,) index into the 29 joint positions (-1 for root)
  track_idx: torch.Tensor  # (NB_TRACK,) indices of tracked bodies
  default_joint_pos: torch.Tensor  # (29,)
  action_scale: torch.Tensor  # (29,)
  soft_limits: torch.Tensor  # (29, 2) soft joint position limits (factor 0.9)

  @property
  def nb(self) -> int:
    return len(self.body_names)

  def to(self, device) -> "G1Kinematics":
    out = {}
    for k, v in self.__dict__.items():
      if isinstance(v, torch.Tensor):
        out[k] = v.to(device)
      elif k == "levels":
        out[k] = [x.to(device) for x in v]
      else:
        out[k] = v
    return G1Kinematics(**out)


def build_kinematics(model: mujoco.MjModel | None = None) -> G1Kinematics:
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTION_SCALE,
    KNEES_BENT_KEYFRAME,
  )
  import re

  m = model if model is not None else compile_g1_model()
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
  first = names.index("pelvis")
  robot_ids = list(range(first, m.nbody))
  body_names = tuple(names[b] for b in robot_ids)
  nb = len(robot_ids)
  local = {b: i for i, b in enumerate(robot_ids)}
  parent = [local.get(int(m.body_parentid[b]), -1) for b in robot_ids]
  depth = [0] * nb
  for i in range(nb):
    depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1
  levels = [torch.tensor([i for i in range(nb) if depth[i] == d]) for d in range(1, max(depth) + 1)]
  for lv in levels:  # parents must be resolved before children
    assert all(depth[parent[int(i)]] < depth[int(i)] for i in lv)

  jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
  body_pos = np.zeros((nb, 3))
  body_quat = np.zeros((nb, 4))
  axis = np.zeros((nb, 3))
  jpos = np.zeros((nb, 3))
  qidx = -np.ones(nb, dtype=np.int64)
  for i, b in enumerate(robot_ids):
    body_pos[i] = m.body_pos[b]
    body_quat[i] = m.body_quat[b]
    js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
    if i == 0:
      assert len(js) == 1 and m.jnt_type[js[0]] == mujoco.mjtJoint.mjJNT_FREE
      continue
    assert len(js) == 1 and m.jnt_type[js[0]] == mujoco.mjtJoint.mjJNT_HINGE, body_names[i]
    axis[i] = m.jnt_axis[js[0]]
    jpos[i] = m.jnt_pos[js[0]]
    qidx[i] = JOINT_NAMES.index(jnames[js[0]])

  def match(table: dict, name: str, default: float) -> float:
    for pat, val in table.items():
      if re.fullmatch(pat, name):
        return val
    return default

  default_q = np.array([match(KNEES_BENT_KEYFRAME.joint_pos, n, 0.0) for n in JOINT_NAMES])
  scale = np.array([match(G1_ACTION_SCALE, n, np.nan) for n in JOINT_NAMES])
  assert not np.isnan(scale).any()
  lim = np.zeros((NJ, 2))
  for j in range(m.njnt):
    if jnames[j] in JOINT_NAMES:
      lo, hi = m.jnt_range[j]
      mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9
      lim[JOINT_NAMES.index(jnames[j])] = (mid - half, mid + half)

  f = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
  return G1Kinematics(
    body_names=body_names,
    parent=parent,
    levels=levels,
    levels_py=tuple(tuple(int(i) for i in lv) for lv in levels),
    body_pos=f(body_pos),
    body_quat=f(body_quat),
    jnt_axis=f(axis),
    jnt_pos=f(jpos),
    jnt_qidx=torch.tensor(qidx),
    track_idx=torch.tensor([body_names.index(n) for n in TRACKED_BODIES]),
    default_joint_pos=f(default_q),
    action_scale=f(scale),
    soft_limits=f(lim),
  )
