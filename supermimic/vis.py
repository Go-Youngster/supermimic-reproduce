"""Visualization scene for the G1 tracking task (offscreen MP4 rendering and the interactive viewer).

The scene is purely kinematic: states produced by the GPU simulator (MuJoCo Warp) or by the learned
world model are written into qpos and MuJoCo only places the bodies. Copies of the robot are attached
with name prefixes: the policy-controlled robot ("robot_") is opaque, reference / world-model copies are
translucent ghosts. Collisions are disabled for every copy (nothing is simulated here).
"""

import mujoco
import numpy as np

from .g1 import JOINT_NAMES

REF_RGBA = (0.30, 0.70, 1.00, 0.30)  # reference motion (LAFAN1 retargeted to G1)
WM_RGBA = (1.00, 0.55, 0.10, 0.45)  # world-model imagination
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def _g1_spec(ghost_rgba=None) -> mujoco.MjSpec:
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  s = get_spec()
  for g in s.geoms:
    g.contype = 0
    g.conaffinity = 0
    if ghost_rgba is not None and g.group == 2:  # visual meshes
      g.material = ""
      g.rgba = list(ghost_rgba)
  if ghost_rgba is not None:  # ghosts carry no lights / cameras
    for x in list(s.lights) + list(s.cameras):
      s.delete(x)
  return s


def build_scene(copies=(("robot_", None), ("ref_", REF_RGBA)), width: int = 960, height: int = 540,
                floor: bool = True) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  spec.modelname = "supermimic_vis"
  vis = spec.visual
  vis.global_.offwidth, vis.global_.offheight = width, height
  vis.headlight.ambient = [0.32, 0.32, 0.32]
  vis.headlight.diffuse = [0.45, 0.45, 0.45]
  vis.headlight.specular = [0.05, 0.05, 0.05]
  vis.quality.shadowsize = 2048  # OSMesa is software GL: 4096 + floor reflection costs ~2x per frame
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.42, 0.56, 0.72], rgb2=[0.06, 0.08, 0.11], width=512, height=3072)
  if floor:
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.25, 0.28, 0.32], rgb2=[0.20, 0.23, 0.26],
                     markrgb=[0.55, 0.60, 0.66], width=512, height=512)
    tex = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"
    spec.add_material(name="grid", textures=tex, texrepeat=[1, 1], texuniform=True, reflectance=0.0)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[80, 80, 0.05], material="grid",
                            contype=0, conaffinity=0)
  for prefix, rgba in copies:
    spec.attach(_g1_spec(rgba), prefix=prefix, frame=spec.worldbody.add_frame())
  return spec.compile()


def prefixes(m: mujoco.MjModel) -> list[str]:
  out = []
  for j in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if n and n.endswith("floating_base_joint"):
      out.append(n[: -len("floating_base_joint")])
  return out


def qpos_index(m: mujoco.MjModel, prefix: str) -> np.ndarray:
  """qpos addresses (36,) of [root pos 3, root quat 4, JOINT_NAMES 29] for the robot copy `prefix`."""
  adr = int(m.jnt_qposadr[m.joint(prefix + "floating_base_joint").id])
  return np.array(list(range(adr, adr + 7)) + [int(m.jnt_qposadr[m.joint(prefix + n).id]) for n in JOINT_NAMES])


def state_to_qpos(s):
  """71-dim task state (pos, quat, lin, ang, jp, jv) -> 36-dim qpos in JOINT_NAMES order."""
  if isinstance(s, np.ndarray):
    return np.concatenate((s[..., :7], s[..., 13:42]), -1)
  import torch

  return torch.cat((s[..., :7], s[..., 13:42]), -1)


HIDDEN = np.concatenate(([0.0, 0.0, -50.0, 1.0, 0.0, 0.0, 0.0], np.zeros(len(JOINT_NAMES))))


def _yaw(q) -> float:
  return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _qmul(a, b) -> np.ndarray:
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2))


def align_ref(ref_q: np.ndarray, robot_q: np.ndarray) -> np.ndarray:
  """Reference placed like the body targets of the tracking reward (task.tracking_reward).

  Root at (robot xy, reference height), rotated by the yaw of robot_root * ref_root^-1. Taking the yaw of the
  relative rotation stays well defined when lying down; the yaw of each root alone does not (pelvis x-axis
  vertical), which made the ghost spin in fall/get-up clips.
  """
  out = np.array(ref_q, dtype=np.float64)
  rel = _qmul(robot_q[3:7], np.asarray(ref_q[3:7]) * np.array([1.0, -1.0, -1.0, -1.0]))
  d = _yaw(rel)
  out[3:7] = _qmul((np.cos(d / 2), 0.0, 0.0, np.sin(d / 2)), ref_q[3:7])
  out[0:2] = robot_q[0:2]
  return out


class Panel:
  """Offscreen renderer with a free camera that smoothly follows the robot root."""

  def __init__(self, model: mujoco.MjModel, width: int, height: int, distance: float = 3.6,
               azimuth: float = 140.0, elevation: float = -14.0, smooth: float = 0.15):
    self.m, self.d = model, mujoco.MjData(model)
    self.renderer = mujoco.Renderer(model, height, width)
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
    self.idx = {p: qpos_index(model, p) for p in prefixes(model)}
    self.smooth, self.look = smooth, None
    for i in self.idx.values():
      self.d.qpos[i] = HIDDEN

  def render(self, qpos: dict, follow: str = "robot_") -> np.ndarray:
    for p, q in qpos.items():
      self.d.qpos[self.idx[p]] = HIDDEN if q is None else q
    mujoco.mj_forward(self.m, self.d)
    target = np.array([qpos[follow][0], qpos[follow][1], 0.78])
    self.look = target if self.look is None else self.look + self.smooth * (target - self.look)
    self.cam.lookat[:] = self.look
    self.renderer.update_scene(self.d, self.cam)
    return self.renderer.render()

  def close(self):
    self.renderer.close()


_FONTS: dict = {}


def font(size: int):
  from PIL import ImageFont

  if size not in _FONTS:
    _FONTS[size] = ImageFont.truetype(FONT, size)
  return _FONTS[size]


def overlay(img: np.ndarray, lines, *, xy=(12, 10), size: int = 19, badge: str | None = None,
            legend=None) -> np.ndarray:
  """Caption box (top-left), optional red badge (top-right) and colour legend (bottom-left)."""
  from PIL import Image, ImageDraw

  im = Image.fromarray(img)
  dr = ImageDraw.Draw(im, "RGBA")
  f = font(size)
  if lines:
    w = max(dr.textlength(t, font=f) for t in lines)
    h = len(lines) * (size + 6) + 8
    dr.rounded_rectangle((xy[0] - 6, xy[1] - 4, xy[0] + w + 8, xy[1] + h - 4), 6, fill=(10, 14, 20, 150))
    for k, t in enumerate(lines):
      dr.text((xy[0], xy[1] + k * (size + 6)), t, font=f, fill=(235, 240, 245, 255))
  if badge:
    fb = font(size + 3)
    w = dr.textlength(badge, font=fb)
    x0 = im.width - w - 26
    dr.rounded_rectangle((x0, 10, im.width - 10, 10 + size + 16), 6, fill=(200, 40, 40, 215))
    dr.text((x0 + 8, 14), badge, font=fb, fill=(255, 255, 255, 255))
  if legend:
    fs = font(size - 3)
    y = im.height - 12 - len(legend) * (size + 2)
    w = max(dr.textlength(t, font=fs) for _, t in legend) + 34
    dr.rounded_rectangle((6, y - 6, 6 + w, im.height - 8), 6, fill=(10, 14, 20, 140))
    for k, (rgb, t) in enumerate(legend):
      yy = y + k * (size + 2)
      dr.rectangle((14, yy + 4, 28, yy + 4 + size - 8), fill=tuple(int(255 * c) for c in rgb[:3]) + (255,))
      dr.text((36, yy), t, font=fs, fill=(230, 235, 240, 255))
  return np.array(im)
