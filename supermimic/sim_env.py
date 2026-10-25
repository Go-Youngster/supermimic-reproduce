"""Motion-tracking simulator on MuJoCo Warp (via mjlab) with analytic reward/termination.

mjlab builds the G1 model, actuators and per-world randomizable fields; stepping and resets
bypass the manager-based env loop and read qpos/qvel directly (no extra forward pass).
"""

import re

import torch

from .g1 import JOINT_NAMES, G1Kinematics, build_kinematics
from .math3d import quat_apply, quat_apply_inv, quat_mul
from .motion_lib import MotionLibrary
from .task import (
  CTX_DIM,
  OBS_NOISE,
  R_MAX,
  STATE_DIM,
  body_states,
  goal_obs,
  proprio_obs,
  reference,
  termination,
  tracking_reward,
)

FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"
FRICTION_RANGE = (0.3, 1.2)
BIAS_RANGE = (-0.01, 0.01)
COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))


def make_env_cfg(num_envs: int):
  from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.envs import mdp as envs_mdp
  from mjlab.envs.mdp import dr as mdr
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.scene import SceneCfg
  from mjlab.sim import MujocoCfg, SimulationCfg
  from mjlab.terrains import TerrainEntityCfg

  # Startup events only declare which model fields become per-world; values are resampled per episode.
  events = {
    "base_com": EventTermCfg(
      mode="startup",
      func=mdr.body_com_offset,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add", "ranges": {0: (0.0, 0.0)}},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdr.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs", "ranges": (1.0, 1.0), "shared_random": True},
    ),
  }
  return ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=num_envs, entities={"robot": get_g1_robot_cfg()}),
    observations={
      "policy": ObservationGroupCfg(
        terms={"jp": ObservationTermCfg(func=envs_mdp.joint_pos_rel)}, concatenate_terms=True, enable_corruption=False
      )
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True
      )
    },
    events=events,
    sim=SimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    episode_length_s=1.0e6,
    auto_reset=False,
  )


def _post_step(kin, lib, s1, frames, action, prev_action):
  nxt = torch.minimum(frames + 1, lib.last_frame[frames])
  ref = reference(lib, nxt)
  bodies = body_states(kin, s1)
  r = tracking_reward(kin, bodies, ref, action, prev_action, s1[:, 13:42])
  terminated, _ = termination(s1, ref)
  return nxt, r, terminated


class TrackingSim:
  """Batched G1 tracking environment. Actions/joints use JOINT_NAMES order."""

  def __init__(
    self,
    lib: MotionLibrary,
    num_envs: int,
    device: str = "cuda:0",
    *,
    train: bool = True,
    max_episode_steps: int = 500,
    goal_k: int = 10,
    hist_len: int = 10,
    seed: int = 0,
    compile: bool = True,
  ):
    from mjlab.envs import ManagerBasedRlEnv

    self.lib, self.num_envs, self.device, self.train = lib, num_envs, device, train
    self.goal_k, self.hist_len, self.max_episode_steps = goal_k, hist_len, max_episode_steps
    cfg = make_env_cfg(num_envs)
    cfg.seed = seed
    self.env = ManagerBasedRlEnv(cfg, device=device)
    self.sim = self.env.sim
    self.robot = self.env.scene["robot"]
    assert tuple(self.robot.joint_names) == JOINT_NAMES, self.robot.joint_names
    self.kin: G1Kinematics = build_kinematics().to(device)
    assert torch.allclose(self.robot.data.default_joint_pos[0], self.kin.default_joint_pos, atol=1e-5)
    assert torch.allclose(self.robot.data.soft_joint_pos_limits[0], self.kin.soft_limits, atol=1e-4)
    self.action_term = self.env.action_manager.get_term("joint_pos")
    tgt = [self.robot.joint_names[i] for i in self.action_term._target_ids.tolist()]
    assert tuple(tgt) == JOINT_NAMES, tgt
    idx = self.robot.indexing
    self.fq, self.fv = idx.free_joint_q_adr.long(), idx.free_joint_v_adr.long()
    self.jq, self.jv = idx.joint_q_adr.long(), idx.joint_v_adr.long()
    self.origins = self.env.scene.env_origins.clone()
    gnames = list(self.robot.geom_names)
    self.foot_geoms = idx.geom_ids[[i for i, n in enumerate(gnames) if re.fullmatch(FOOT_GEOMS, n)]].long()
    self.torso_body = int(idx.body_ids[list(self.robot.body_names).index("torso_link")])
    assert self.sim.model.geom_friction.dim() == 3 and self.sim.model.body_ipos.dim() == 3
    self.default_ipos = self.sim.model.body_ipos[0, self.torso_body].clone()

    n = num_envs
    self.frames = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_return = torch.zeros(n, device=device)
    self.ep_disc_return = torch.zeros(n, device=device)
    self.ep_start_bin = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_start_frame = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_id = torch.zeros(n, dtype=torch.long, device=device)
    self.xi = torch.zeros(n, CTX_DIM, device=device)
    self.xi[:, 0] = 1.0
    self.prev_action = torch.zeros(n, len(JOINT_NAMES), device=device)
    self.obs_hist = torch.zeros(n, hist_len, OBS_NOISE.numel(), device=device)
    self.obs_noise = OBS_NOISE.to(device) if train else torch.zeros_like(OBS_NOISE, device=device)
    self._post = torch.compile(_post_step, dynamic=False) if compile else _post_step
    self.reset(torch.arange(n, device=device))

  # ---- state access -------------------------------------------------------
  def state(self) -> torch.Tensor:
    qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
    quat = qpos[:, self.fq[3:7]]
    return torch.cat(
      (
        qpos[:, self.fq[0:3]] - self.origins,
        quat,
        qvel[:, self.fv[0:3]],
        quat_apply(quat, qvel[:, self.fv[3:6]]),
        qpos[:, self.jq],
        qvel[:, self.jv],
      ),
      -1,
    )

  def context(self) -> torch.Tensor:
    return self.xi

  def observe(self, s=None) -> torch.Tensor:
    s = self.state() if s is None else s
    o = proprio_obs(self.kin, s, self.xi, self.prev_action)
    return o + (torch.rand_like(o) * 2 - 1) * self.obs_noise

  def goal(self, s=None) -> torch.Tensor:
    s = self.state() if s is None else s
    return goal_obs(self.lib, s, self.frames, self.goal_k)

  # ---- reset / step -------------------------------------------------------
  def _randomize(self, env_ids: torch.Tensor):
    n = env_ids.numel()
    u = lambda lo, hi, shape: torch.rand(shape, device=self.device) * (hi - lo) + lo  # noqa: E731
    if self.train:
      fr = u(*FRICTION_RANGE, (n,))
      bias = u(*BIAS_RANGE, (n, len(JOINT_NAMES)))
      com = torch.stack([u(lo, hi, (n,)) for lo, hi in COM_RANGE], -1)
    else:
      fr = torch.ones(n, device=self.device)
      bias = torch.zeros(n, len(JOINT_NAMES), device=self.device)
      com = torch.zeros(n, 3, device=self.device)
    m = self.sim.model
    m.geom_friction[env_ids[:, None], self.foot_geoms[None, :], 0] = fr[:, None]
    m.body_ipos[env_ids, self.torso_body] = self.default_ipos + com
    self.robot.data.encoder_bias[env_ids] = bias
    self.xi[env_ids] = torch.cat((fr[:, None], bias, com), -1)

  def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None, perturb: bool | None = None):
    if env_ids.numel() == 0:
      return
    perturb = self.train if perturb is None else perturb
    if frames is None:
      frames = self.lib.sample_starts(env_ids.numel())
    n = env_ids.numel()
    self.sim.reset(env_ids)
    self._randomize(env_ids)
    ref = reference(self.lib, frames)
    root_pos = ref["pos"][:, 0] + self.origins[env_ids]
    root_quat = ref["quat"][:, 0]
    lin, ang = ref["lin"][:, 0], ref["ang"][:, 0]
    jp, jv = ref["jpos"], ref["jvel"]
    if perturb:  # BeyondMimic reset randomization
      u = lambda lo, hi, shape: torch.rand(shape, device=self.device) * (hi - lo) + lo  # noqa: E731
      root_pos = root_pos + torch.cat((u(-0.05, 0.05, (n, 2)), u(-0.01, 0.01, (n, 1))), -1)
      rpy = torch.cat((u(-0.1, 0.1, (n, 2)), u(-0.2, 0.2, (n, 1))), -1)
      root_quat = quat_mul(_quat_from_rpy(rpy), root_quat)
      lin = lin + torch.cat((u(-0.5, 0.5, (n, 2)), u(-0.2, 0.2, (n, 1))), -1)
      ang = ang + torch.cat((u(-0.52, 0.52, (n, 2)), u(-0.78, 0.78, (n, 1))), -1)
      jp = jp + u(-0.1, 0.1, jp.shape)
    lim = self.kin.soft_limits
    jp = torch.clamp(jp, lim[:, 0], lim[:, 1])
    d = self.sim.data
    e = env_ids[:, None]
    d.qpos[e, self.fq[None, :]] = torch.cat((root_pos, root_quat), -1)
    d.qvel[e, self.fv[None, :]] = torch.cat((lin, quat_apply_inv(root_quat, ang)), -1)
    d.qpos[e, self.jq[None, :]] = jp
    d.qvel[e, self.jv[None, :]] = jv
    # Actuator targets are fully rewritten by the next physics_step, so no robot/action reset is needed.
    self.frames[env_ids] = frames
    self.ep_steps[env_ids] = 0
    self.ep_return[env_ids] = 0
    self.ep_disc_return[env_ids] = 0
    self.ep_start_frame[env_ids] = frames
    self.ep_start_bin[env_ids] = self.lib.frame_bins(frames)
    self.ep_id[env_ids] += 1
    self.prev_action[env_ids] = 0
    s = self.state()[env_ids]
    o = proprio_obs(self.kin, s, self.xi[env_ids], self.prev_action[env_ids])
    o = o + (torch.rand_like(o) * 2 - 1) * self.obs_noise
    self.obs_hist[env_ids] = o.unsqueeze(1).expand(-1, self.hist_len, -1)

  def physics_step(self, action: torch.Tensor):
    self.action_term.process_actions(action)
    self.action_term.apply_actions()
    self.env.scene.write_data_to_sim()
    for _ in range(4):
      self.sim.step()

  def step(self, action: torch.Tensor, gamma: float = 0.99):
    """Steps physics; returns next state (pre-reset), reward and done flags. Does not reset."""
    self.physics_step(action)
    s1 = self.state()
    nxt, r, terminated = self._post(self.kin, self.lib, s1, self.frames, action, self.prev_action)
    self.frames = nxt
    self.ep_steps += 1
    self.ep_disc_return += r * gamma ** (self.ep_steps - 1).float()
    self.ep_return += r
    clip_end = (self.frames >= self.lib.last_frame[self.frames]) & ~terminated
    timeout = (self.ep_steps >= self.max_episode_steps) & ~terminated & ~clip_end
    self.prev_action = action.clone()
    return dict(next_state=s1, reward=r, terminated=terminated, clip_end=clip_end, timeout=timeout)

  def finish_episodes(self, done: torch.Tensor, gamma: float = 0.99):
    """Updates return-based sampling statistics for finished episodes and resets them."""
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return ids
    start = self.ep_start_frame[ids]
    remaining = (self.lib.last_frame[start] - start).clamp_min(1).float()
    horizon = torch.minimum(remaining, torch.full_like(remaining, float(self.max_episode_steps)))
    g_max = R_MAX * (1 - gamma**horizon) / (1 - gamma)
    self.lib.update_bin_returns(self.ep_start_bin[ids], (self.ep_disc_return[ids] / g_max).clamp(0, 1))
    self.reset(ids)
    return ids

  def push_obs(self):
    o = self.observe()
    self.obs_hist = torch.cat((self.obs_hist[:, 1:], o.unsqueeze(1)), 1)
    return o


def _quat_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
  r, p, y = rpy.unbind(-1)
  cr, sr, cp, sp, cy, sy = (torch.cos(r / 2), torch.sin(r / 2), torch.cos(p / 2), torch.sin(p / 2), torch.cos(y / 2), torch.sin(y / 2))
  return torch.stack(
    (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1
  )


__all__ = ["TrackingSim", "STATE_DIM", "CTX_DIM"]
