"""SuperMimic training loop (Algorithm A.1): collect -> fit world model -> BPTT policy + critic."""

import argparse
import dataclasses
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .evaluate import Evaluator
from .g1 import NJ
from .motion_lib import MotionLibrary
from .networks import AMP, Critic, Policy, WorldModel, state_delta, state_error
from .replay import Replay
from .runlog import RunLogger
from .sim_env import TrackingSim
from .task import (
  CTX_DIM,
  GOAL_FRAME_DIM,
  OBS_DIM,
  R_MAX,
  STATE_DIM,
  body_states,
  critic_features,
  goal_noise_mask,
  goal_obs,
  proprio_obs,
  reference,
  termination,
  tracking_reward,
)


@dataclass
class Config:
  motion_file: str = "data/motions/lafan1.npz"
  clips: str = ""  # comma-separated clip-name substrings to keep (empty = all)
  run_dir: str = "runs/debug"
  seed: int = 0
  num_envs: int = 4096
  replay_steps: int = 1000
  max_episode_steps: int = 500
  goal_k: int = 10
  hist: int = 10
  action_noise: float = 0.1
  utd: float = 0.4
  # world model
  wm_k: int = 8
  wm_batches: int = 10
  wm_batch: int = 2048
  wm_hidden: int = 2048
  # policy / critic
  horizon: int = 32
  imag_batch: int = 4096
  gamma: float = 0.99
  lam: float = 0.95
  use_critic: bool = True
  critic_epochs: int = 5
  critic_minibatches: int = 4
  lr: float = 2e-4
  lr_decay_iters: int = 250_000
  lr_final_frac: float = 0.03
  total_iters: int = 320_000
  max_hours: float = 12.0  # wall-clock training budget (evaluation time excluded)
  warmup_env_steps: int = 24
  wm_pretrain_steps: int = 300
  # bookkeeping
  log_every: int = 50
  eval_every_min: float = 20.0
  ckpt_every: int = 1000
  snapshot_every: int = 4000
  compile: bool = True
  bf16: bool = True  # bf16 autocast for MLP matmuls (A100 substitute for the paper's TF32 on H100)
  cudagraphs: bool = False
  cpp_wrapper: bool = False
  final_eval: bool = True
  stop_at_return: float = 0.0  # stop once eval return reaches this (0 = never)
  # PPO-WM ablation (Fig. 5b): replace the BPTT policy update by PPO on imagined rollouts
  policy_algo: str = "bptt"
  ppo_rollouts: int = 16384
  ppo_steps: int = 32
  ppo_epochs: int = 5
  ppo_minibatches: int = 4
  ppo_clip: float = 0.2
  ppo_entropy: float = 0.005
  ppo_lr: float = 1e-3
  ppo_desired_kl: float = 0.01
  ppo_init_std: float = 1.0


class SuperMimic:
  def __init__(self, cfg: Config):
    self.cfg = cfg
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 32
    AMP["bf16"] = cfg.bf16
    if cfg.cpp_wrapper:
      import torch._inductor.config as inductor_config

      inductor_config.cpp_wrapper = True
    dev = self.dev = "cuda:0"
    lib_ids = None
    if cfg.clips:
      import numpy as _np

      names = [str(n) for n in _np.load(cfg.motion_file)["names"]]
      keys = [k.strip() for k in cfg.clips.split(",") if k.strip()]
      lib_ids = [i for i, n in enumerate(names) if any(k in n for k in keys)]
    self.lib = MotionLibrary(cfg.motion_file, dev, clip_ids=lib_ids)
    self.lib_ids = lib_ids
    self.sim = TrackingSim(self.lib, cfg.num_envs, dev, max_episode_steps=cfg.max_episode_steps,
                           goal_k=cfg.goal_k, hist_len=cfg.hist, seed=cfg.seed, compile=cfg.compile)
    self.kin = self.sim.kin
    goal_dim = GOAL_FRAME_DIM * cfg.goal_k
    self.goal_noise = goal_noise_mask(cfg.goal_k, dev)
    self.policy = Policy(OBS_DIM, cfg.hist, goal_dim).to(dev)
    self.wm = WorldModel(self.kin, hidden=(cfg.wm_hidden,) * 3).to(dev)
    crit_dim = 68 + 7 * self.kin.nb + goal_dim + 3 * cfg.goal_k + CTX_DIM
    self.critic = Critic(crit_dim).to(dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr, fused=True)
    self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95), fused=True)
    self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, fused=True)
    self.ppo = cfg.policy_algo == "ppo"
    if self.ppo:
      self.std = nn.Parameter(torch.full((NJ,), cfg.ppo_init_std, device=dev))
      self.ppo_params = list(self.policy.parameters()) + [self.std] + list(self.critic.parameters())
      self.opt_ppo = torch.optim.Adam(self.ppo_params, lr=cfg.ppo_lr)
      self.ppo_lr = cfg.ppo_lr
    self.replay = Replay(cfg.replay_steps, cfg.num_envs, STATE_DIM, NJ, OBS_DIM, CTX_DIM, dev)
    self.log = RunLogger(cfg.run_dir, dataclasses.asdict(cfg), "ppo_wm" if self.ppo else "supermimic",
                         dict(motion_hours=round(self.lib.hours, 4), num_clips=self.lib.num_clips))
    self.it = 0
    self.train_time = 0.0
    self.env_steps = 0
    self.imag_steps = 0
    self.best_eval = -1.0
    self.utd_acc = 0.0
    self.ep_stats = []
    mode = "reduce-overhead" if cfg.cudagraphs else None
    if cfg.cudagraphs:
      self._mark_static()
    comp = (lambda f: torch.compile(f, dynamic=False, mode=mode)) if cfg.compile else (lambda f: f)
    self._wm_step = comp(self._wm_step_impl)
    self._wm_rollout = self._wm_rollout_impl
    self._imag_step = comp(self._imag_step_impl)
    self._crit_feat = comp(self._crit_feat_impl)
    self._ppo_step = comp(self._ppo_step_impl)
    self._evaluator = None
    self._resume()

  # ---------------------------------------------------------------- helpers
  def _mark_static(self):
    from .kinematics_fast import tables

    T = tables(self.kin)
    ts = [v for v in vars(self.lib).values() if isinstance(v, torch.Tensor)]
    ts += [v for v in vars(self.kin).values() if isinstance(v, torch.Tensor)] + list(self.kin.levels)
    ts += [T.A, T.parent, T.qidx, T.has_joint, *T.jumps, self.goal_noise, self.sim.obs_noise]
    for t in ts:
      torch._dynamo.mark_static_address(t)

  def lr_at(self, it: int) -> float:
    c = self.cfg
    x = min(it, c.lr_decay_iters) / c.lr_decay_iters
    return c.lr * (c.lr_final_frac + (1 - c.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * x)))

  def _set_lr(self):
    lr = self.lr_at(self.it)
    for opt in (self.opt_wm, self.opt_pi, self.opt_v):
      for g in opt.param_groups:
        g["lr"] = lr
    return lr

  def act(self, obs_hist, goal):
    return self.policy(obs_hist, goal)

  # ---------------------------------------------------------------- collect
  @torch.no_grad()
  def collect_step(self):
    sim, c = self.sim, self.cfg
    s = sim.state()
    xi = sim.xi.clone()
    f = sim.frames.clone()
    o = sim.obs_hist[:, -1].clone()
    g = sim.goal(s)
    g = g + (torch.rand_like(g) * 2 - 1) * self.goal_noise
    self.policy.obs_norm.update(o)
    self.policy.goal_norm.update(g)
    a = self.policy(sim.obs_hist, g)
    a = a + c.action_noise * torch.randn_like(a)
    out = sim.step(a, c.gamma)
    self.replay.add(s, a, out["next_state"], o, xi, f, sim.ep_id, sim.ep_steps - 1, out["terminated"], out["reward"])
    done = out["terminated"] | out["clip_end"] | out["timeout"]
    if bool(done.any()):
      ids = done.nonzero().squeeze(-1)
      steps = sim.ep_steps[ids].float()
      start = sim.ep_start_frame[ids]
      intended = torch.clamp(self.lib.last_frame[start] - start, max=c.max_episode_steps).clamp_min(1).float()
      self.ep_stats.append(torch.stack((sim.ep_return[ids] / (R_MAX * intended), steps, out["terminated"][ids].float()), -1))
    sim.finish_episodes(done, c.gamma)
    sim.push_obs()
    self.env_steps += c.num_envs

  # ---------------------------------------------------------------- world model
  def _wm_step_impl(self, s, a, xi, s_target):
    s1 = self.wm(s, a, xi)
    e = state_error(s1, s_target) / self.wm.nominal
    return s1, (e * e).mean(-1)

  def _wm_rollout_impl(self, s0, acts, s_next, xi):
    s = s0
    errs = []
    for j in range(acts.shape[1]):
      s, err = self._wm_step(s, acts[:, j], xi, s_next[:, j])
      errs.append(err)
    return torch.stack(errs, 1)  # (B, k)

  def wm_update(self, n_steps: int):
    c = self.cfg
    losses = []
    for _ in range(n_steps):
      if c.cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
      s0, acts, s1, xi = self.replay.sample_windows(c.wm_batch, c.wm_k)
      with torch.no_grad():
        self.wm.in_norm.update(self.wm.features(s0, acts[:, 0], xi, body_states(self.kin, s0)))
        self.wm.out_norm.update(state_delta(s0, s1[:, 0]))
      loss = self._wm_rollout(s0, acts, s1, xi).mean()
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
      self.opt_wm.step()
      losses.append(loss.detach())
    return torch.stack(losses).mean()

  @torch.no_grad()
  def wm_online_error(self):
    """8-step prediction error on the most recent policy rollouts (normalized by NOMINAL_ERR)."""
    c = self.cfg
    if c.cudagraphs:
      torch.compiler.cudagraph_mark_step_begin()
    s0, acts, s1, xi = self.replay.sample_windows(c.wm_batch, c.wm_k, recent=4 * c.wm_k)
    return self._wm_rollout(s0, acts, s1, xi).mean(0)

  # ---------------------------------------------------------------- imagination
  def _imag_step_impl(self, s, xi, f, hist, prev_a, bodies):
    lib, kin = self.lib, self.kin
    g_clean = goal_obs(lib, s, f, self.cfg.goal_k)
    g = g_clean + (torch.rand_like(g_clean) * 2 - 1) * self.goal_noise
    a = self.policy(hist, g)
    s1 = self.wm(s, a, xi, bodies)
    last = lib.last_frame[f]
    f1 = torch.minimum(f + 1, last)
    ref = reference(lib, f1)
    bodies1 = body_states(kin, s1)
    r = tracking_reward(kin, bodies1, ref, a, prev_a, s1[:, 13 : 13 + NJ])
    _, d = termination(s1, ref)
    valid = (f < last).float()
    end = (f1 >= last).float()
    o1 = proprio_obs(kin, s1, xi, a)
    o1 = o1 + (torch.rand_like(o1) * 2 - 1) * self.sim.obs_noise
    hist1 = torch.cat((hist[:, 1:], o1.unsqueeze(1)), 1)
    return a, s1, f1, bodies1, r * valid, d * valid, end, hist1

  def _crit_feat_impl(self, s, xi, f, bodies):
    g = goal_obs(self.lib, s, f, self.cfg.goal_k)
    return critic_features(self.kin, self.lib, s, xi, f, bodies, g, self.cfg.goal_k)

  # ---------------------------------------------------------------- PPO-WM
  def _ppo_step_impl(self, s, xi, f, hist, prev_a, bodies, std):
    lib, kin = self.lib, self.kin
    g_clean = goal_obs(lib, s, f, self.cfg.goal_k)
    g = g_clean + (torch.rand_like(g_clean) * 2 - 1) * self.goal_noise
    x = critic_features(kin, lib, s, xi, f, bodies, g_clean, self.cfg.goal_k)
    mu = self.policy(hist, g)
    a = mu + std * torch.randn_like(mu)
    logp = (-0.5 * ((a - mu) / std) ** 2 - torch.log(std) - 0.9189385332).sum(-1)
    v = self.critic(x)
    s1 = self.wm(s, a, xi, bodies)
    last = lib.last_frame[f]
    f1 = torch.minimum(f + 1, last)
    ref = reference(lib, f1)
    bodies1 = body_states(kin, s1)
    r = tracking_reward(kin, bodies1, ref, a, prev_a, s1[:, 13 : 13 + NJ])
    _, d = termination(s1, ref)
    valid = (f < last).float()
    end = (f1 >= last).float()
    o1 = proprio_obs(kin, s1, xi, a)
    o1 = o1 + (torch.rand_like(o1) * 2 - 1) * self.sim.obs_noise
    hist1 = torch.cat((hist[:, 1:], o1.unsqueeze(1)), 1)
    return g, x, a, mu, logp, v, s1, f1, bodies1, r * valid, d * valid, end, hist1

  def ppo_wm_update(self):
    c = self.cfg
    N, T = c.ppo_rollouts, c.ppo_steps
    with torch.no_grad():
      s, xi, f, hist = self.replay.sample_starts(N, c.hist)
      prev_a = hist[:, -1, 2 * NJ + 6 :].clone()
      bodies = body_states(self.kin, s)
      std = self.std.detach().clamp_min(1e-3)
      buf = dict(hist=[], goal=[], crit=[], act=[], mu=[], logp=[], val=[], rew=[], cont=[])
      for _ in range(T):
        buf["hist"].append(hist)
        g, x, a, mu, logp, v, s, f, bodies, r, d, end, hist = self._ppo_step(s, xi, f, hist, prev_a, bodies, std)
        prev_a = a
        for k, val in (("goal", g), ("crit", x), ("act", a), ("mu", mu), ("logp", logp), ("val", v), ("rew", r), ("cont", (1 - d) * (1 - end))):
          buf[k].append(val)
      last_v = self.critic(self._crit_feat(s, xi, f, bodies))
      B = {k: torch.stack(v) for k, v in buf.items()}
      self.policy.obs_norm.update(B["hist"][:, :, -1])
      self.policy.goal_norm.update(B["goal"])
      self.critic.norm.update(B["crit"])
      adv = torch.zeros_like(B["rew"])
      gae = torch.zeros(N, device=self.dev)
      for t in reversed(range(T)):
        nv = last_v if t == T - 1 else B["val"][t + 1]
        delta = B["rew"][t] + c.gamma * B["cont"][t] * nv - B["val"][t]
        gae = delta + c.gamma * c.lam * B["cont"][t] * gae
        adv[t] = gae
      B["ret"] = adv + B["val"]
      B["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = N * T
    flat = {k: v.reshape(n, *v.shape[2:]) for k, v in B.items()}
    mb = n // c.ppo_minibatches
    stats = []
    for _ in range(c.ppo_epochs):
      perm = torch.randperm(n, device=self.dev)
      for k in range(c.ppo_minibatches):
        idx = perm[k * mb : (k + 1) * mb]
        mu = self.policy(flat["hist"][idx], flat["goal"][idx])
        sd = self.std.clamp_min(1e-3)
        a = flat["act"][idx]
        logp = (-0.5 * ((a - mu) / sd) ** 2 - torch.log(sd) - 0.9189385332).sum(-1)
        ent = (0.5 + 0.9189385332 + torch.log(sd)).sum()
        with torch.no_grad():
          kl = (torch.log(sd / std) + (std**2 + (flat["mu"][idx] - mu) ** 2) / (2 * sd**2) - 0.5).sum(-1).mean()
          if kl > 2 * c.ppo_desired_kl:
            self.ppo_lr = max(1e-5, self.ppo_lr / 1.5)
          elif kl < c.ppo_desired_kl / 2:
            self.ppo_lr = min(1e-2, self.ppo_lr * 1.5)
          for g in self.opt_ppo.param_groups:
            g["lr"] = self.ppo_lr
        ratio = torch.exp(logp - flat["logp"][idx])
        adv = flat["adv"][idx]
        surr = -torch.min(ratio * adv, ratio.clamp(1 - c.ppo_clip, 1 + c.ppo_clip) * adv).mean()
        v = self.critic(flat["crit"][idx])
        v_old = flat["val"][idx]
        v_clip = v_old + (v - v_old).clamp(-c.ppo_clip, c.ppo_clip)
        ret = flat["ret"][idx]
        v_loss = torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
        loss = surr + v_loss - c.ppo_entropy * ent
        self.opt_ppo.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.ppo_params, 1.0)
        self.opt_ppo.step()
        stats.append(torch.stack((surr.detach(), v_loss.detach(), kl)))
    self.imag_steps += N * T
    st = torch.stack(stats).mean(0)
    return dict(imag_reward=B["rew"].mean() / R_MAX, surrogate=st[0], value_loss=st[1], kl=st[2],
                std=self.std.detach().mean(), ppo_lr=torch.tensor(self.ppo_lr))

  def policy_update(self):
    c = self.cfg
    if c.cudagraphs:
      torch.compiler.cudagraph_mark_step_begin()
    s, xi, f, hist = self.replay.sample_starts(c.imag_batch, c.hist)
    prev_a = hist[:, -1, 2 * NJ + 6 :].clone()
    self.wm.requires_grad_(False)
    self.critic.requires_grad_(False)
    bodies = body_states(self.kin, s)
    feats, rews, dones, ends = [], [], [], []
    J = torch.zeros(s.shape[0], device=self.dev)
    surv = torch.ones_like(J)
    disc = 1.0
    for t in range(c.horizon):
      feats.append((s.detach(), f, tuple(b.detach() for b in bodies)))
      a, s, f, bodies, r, d, end, hist = self._imag_step(s, xi, f, hist, prev_a, bodies)
      J = J + disc * surv * r
      surv = surv * (1 - d) * (1 - end)
      disc *= c.gamma
      prev_a = a
      rews.append(r.detach())
      dones.append(d.detach())
      ends.append(end)
    feats.append((s.detach(), f, tuple(b.detach() for b in bodies)))
    if c.use_critic:
      v_last = self.critic(self._crit_feat(s, xi, f, bodies))
      J = J + disc * surv * v_last
    loss = -J.mean()
    self.opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
    nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
    self.opt_pi.step()
    self.wm.requires_grad_(True)
    self.critic.requires_grad_(True)
    self.imag_steps += c.imag_batch * c.horizon
    out = dict(J=J.mean().detach(), pi_grad_norm=gn.detach(), imag_reward=torch.stack(rews).mean() / R_MAX,
               imag_survival=surv.mean().detach())
    if c.use_critic:
      out.update(self.critic_update(feats, xi, rews, dones, ends))
    return out

  def critic_update(self, feats, xi, rews, dones, ends):
    c = self.cfg
    with torch.no_grad():
      X = torch.stack([self._crit_feat(s, xi, f, b) for s, f, b in feats])  # (H+1, B, D)
      self.critic.norm.update(X[:-1])
      V = self.critic(X)
      R = V[-1]
      targets = []
      for t in reversed(range(c.horizon)):
        R = rews[t] + c.gamma * (1 - dones[t]) * (1 - ends[t]) * ((1 - c.lam) * V[t + 1] + c.lam * R)
        targets.append(R)
      targets = torch.stack(targets[::-1])  # (H, B)
    Xf = X[:-1].reshape(-1, X.shape[-1])
    Tf = targets.reshape(-1)
    n = Tf.numel()
    mb = n // c.critic_minibatches
    losses = []
    for _ in range(c.critic_epochs):
      perm = torch.randperm(n, device=self.dev)
      for k in range(c.critic_minibatches):
        idx = perm[k * mb : (k + 1) * mb]
        loss = ((self.critic(Xf[idx]) - Tf[idx]) ** 2).mean()
        self.opt_v.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_v.step()
        losses.append(loss.detach())
    return dict(critic_loss=torch.stack(losses).mean(), value_mean=V[:-1].mean(), target_mean=Tf.mean())

  # ---------------------------------------------------------------- eval / ckpt
  def evaluate(self) -> dict:
    if self._evaluator is None:
      self._evaluator = Evaluator(self.lib, self.dev)
    self.policy.eval()
    res = self._evaluator.run(lambda h, g: self.policy(h, g))
    self.policy.train()
    return res

  def state_dict(self):
    sd = dict(it=self.it, train_time=self.train_time, env_steps=self.env_steps, imag_steps=self.imag_steps,
              best_eval=self.best_eval, policy=self.policy.state_dict(), wm=self.wm.state_dict(),
              critic=self.critic.state_dict(), opt_wm=self.opt_wm.state_dict(), opt_pi=self.opt_pi.state_dict(),
              opt_v=self.opt_v.state_dict(), lib=self.lib.state_dict(), cfg=dataclasses.asdict(self.cfg))
    if self.ppo:
      sd.update(std=self.std.detach().cpu(), opt_ppo=self.opt_ppo.state_dict(), ppo_lr=self.ppo_lr)
    return sd

  def _resume(self):
    sd = self.log.load("latest")
    if sd is None:
      return
    self.policy.load_state_dict(sd["policy"])
    self.wm.load_state_dict(sd["wm"])
    self.critic.load_state_dict(sd["critic"])
    self.opt_wm.load_state_dict(sd["opt_wm"])
    self.opt_pi.load_state_dict(sd["opt_pi"])
    self.opt_v.load_state_dict(sd["opt_v"])
    self.lib.load_state_dict({k: v.to(self.dev) for k, v in sd["lib"].items()})
    self.it, self.train_time = sd["it"], sd["train_time"]
    self.env_steps, self.imag_steps, self.best_eval = sd["env_steps"], sd["imag_steps"], sd["best_eval"]
    if self.ppo and "std" in sd:
      self.std.data.copy_(sd["std"].to(self.dev))
      self.opt_ppo.load_state_dict(sd["opt_ppo"])
      self.ppo_lr = sd["ppo_lr"]
    print(f"[resume] it={self.it} train_time={self.train_time / 3600:.2f}h")

  # ---------------------------------------------------------------- main loop
  def train(self):
    c = self.cfg
    resumed = self.it > 0
    t_wall0 = time.time()
    # Fill the replay buffer before the first update (also after a resume, since it is not checkpointed).
    t0 = time.time()
    for _ in range(max(c.warmup_env_steps, c.wm_k + 2)):
      self.collect_step()
    if not resumed:
      self.wm_update(c.wm_pretrain_steps)
    torch.cuda.synchronize()
    self.train_time += time.time() - t0
    last_eval_time = self.train_time if resumed else -1e9
    timers = dict(collect=0.0, wm=0.0, policy=0.0)
    acc_logs = []
    while self.it < c.total_iters and self.train_time < c.max_hours * 3600:
      t_it = time.time()
      lr = self._set_lr()
      self.utd_acc += 1.0 / c.utd
      n_col = int(self.utd_acc)
      self.utd_acc -= n_col
      ta = time.time()
      for _ in range(n_col):
        self.collect_step()
      tb = time.time()
      wm_loss = self.wm_update(c.wm_batches)
      tc = time.time()
      pol = self.ppo_wm_update() if self.ppo else self.policy_update()
      td = time.time()
      timers["collect"] += tb - ta
      timers["wm"] += tc - tb
      timers["policy"] += td - tc
      self.it += 1
      self.train_time += time.time() - t_it
      acc_logs.append(dict(wm_loss=wm_loss, **pol))
      if self.it % c.log_every == 0:
        torch.cuda.synchronize()
        rec = dict(it=self.it, train_h=self.train_time / 3600, wall_h=(time.time() - t_wall0) / 3600, lr=lr,
                   env_steps=self.env_steps, imag_steps=self.imag_steps)
        for k in acc_logs[0]:
          rec[k] = float(torch.stack([x[k].float() for x in acc_logs]).mean())
        acc_logs = []
        pe = self.wm_online_error()
        rec["wm_pred_err8"] = float(pe.mean())
        rec["wm_pred_err1"] = float(pe[0])
        if self.ep_stats:
          st = torch.cat(self.ep_stats)
          rec.update(train_return=float(st[:, 0].mean()), ep_len=float(st[:, 1].mean()), term_frac=float(st[:, 2].mean()),
                     episodes=int(st.shape[0]))
          self.ep_stats = []
        tot = sum(timers.values())
        rec.update({f"t_{k}": v / c.log_every for k, v in timers.items()}, it_per_s=c.log_every / max(tot, 1e-9))
        timers = dict(collect=0.0, wm=0.0, policy=0.0)
        rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self.log.log(rec)
        self.log.status(state="running", it=self.it, train_h=rec["train_h"], last=rec)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
      if self.it % c.ckpt_every == 0:
        self.log.save(self.state_dict())
      if self.it % c.snapshot_every == 0:
        self.log.save(dict(policy=self.policy.state_dict(), it=self.it, train_time=self.train_time), f"it{self.it:06d}")
      if self.train_time - last_eval_time >= c.eval_every_min * 60:
        last_eval_time = self.train_time
        if self._run_eval():
          break
    self.log.save(self.state_dict())
    if c.final_eval:
      self._run_eval()
    self.log.status(state="finished", it=self.it, train_h=self.train_time / 3600)

  def _run_eval(self) -> bool:
    te = time.time()
    res = self.evaluate()
    res.update(it=self.it, train_h=self.train_time / 3600, env_steps=self.env_steps, imag_steps=self.imag_steps,
               eval_seconds=time.time() - te)
    self.log.log_eval(res)
    print(f"[eval] it={self.it} train_h={res['train_h']:.2f} return={res['norm_return']:.4f} failed={res['failed']}/{res['num_motions']}", flush=True)
    if res["norm_return"] > self.best_eval:
      self.best_eval = res["norm_return"]
      self.log.save(dict(policy=self.policy.state_dict(), it=self.it, train_time=self.train_time, eval=res), "best")
    return self.cfg.stop_at_return > 0 and res["norm_return"] >= self.cfg.stop_at_return


def parse_config(argv=None) -> Config:
  ap = argparse.ArgumentParser()
  for f in dataclasses.fields(Config):
    if f.type is bool or f.type == "bool":
      ap.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=f.default)
    else:
      ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
  return Config(**vars(ap.parse_args(argv)))


if __name__ == "__main__":
  SuperMimic(parse_config()).train()
