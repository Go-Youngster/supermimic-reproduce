"""PPO-Sim baseline: PPO in the simulator with SuperMimic's motions, observations, reward, termination,
domain randomization, network sizes and evaluation (BeyondMimic PPO hyperparameters)."""

import argparse
import dataclasses
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .evaluate import Evaluator
from .g1 import NJ
from .motion_lib import MotionLibrary
from .networks import Critic, Policy
from .runlog import RunLogger
from .sim_env import TrackingSim
from .task import CTX_DIM, GOAL_FRAME_DIM, OBS_DIM, R_MAX, body_states, critic_features, goal_noise_mask


@dataclass
class PPOConfig:
  motion_file: str = "data/motions/lafan1.npz"
  clips: str = ""
  run_dir: str = "runs/ppo_debug"
  seed: int = 0
  num_envs: int = 8192
  steps_per_env: int = 24
  max_episode_steps: int = 500
  goal_k: int = 10
  hist: int = 10
  lr: float = 1e-3
  desired_kl: float = 0.01
  clip: float = 0.2
  entropy_coef: float = 0.005
  value_coef: float = 1.0
  epochs: int = 5
  minibatches: int = 4
  gamma: float = 0.99
  lam: float = 0.95
  max_grad_norm: float = 1.0
  init_std: float = 1.0
  max_hours: float = 18.0
  total_iters: int = 10**9
  log_every: int = 10
  eval_every_min: float = 30.0
  ckpt_every: int = 200
  final_eval: bool = True


class PPOSim:
  def __init__(self, cfg: PPOConfig):
    self.cfg = cfg
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 32
    dev = self.dev = "cuda:0"
    lib_ids = None
    if cfg.clips:
      names = [str(n) for n in np.load(cfg.motion_file)["names"]]
      keys = [k.strip() for k in cfg.clips.split(",") if k.strip()]
      lib_ids = [i for i, n in enumerate(names) if any(k in n for k in keys)]
    self.lib = MotionLibrary(cfg.motion_file, dev, clip_ids=lib_ids)
    self.sim = TrackingSim(self.lib, cfg.num_envs, dev, max_episode_steps=cfg.max_episode_steps,
                           goal_k=cfg.goal_k, hist_len=cfg.hist, seed=cfg.seed)
    self.kin = self.sim.kin
    goal_dim = GOAL_FRAME_DIM * cfg.goal_k
    self.goal_noise = goal_noise_mask(cfg.goal_k, dev)
    self.actor = Policy(OBS_DIM, cfg.hist, goal_dim).to(dev)
    self.std = nn.Parameter(torch.full((NJ,), cfg.init_std, device=dev))
    crit_dim = 68 + 7 * self.kin.nb + goal_dim + 3 * cfg.goal_k + CTX_DIM
    self.critic = Critic(crit_dim).to(dev)
    self.params = list(self.actor.parameters()) + [self.std] + list(self.critic.parameters())
    self.opt = torch.optim.Adam(self.params, lr=cfg.lr)
    self.lr = cfg.lr
    self.log = RunLogger(cfg.run_dir, dataclasses.asdict(cfg), "ppo_sim",
                         dict(motion_hours=round(self.lib.hours, 4), num_clips=self.lib.num_clips))
    self.it, self.train_time, self.env_steps = 0, 0.0, 0
    self.best_eval = -1.0
    self.ep_stats = []
    self._evaluator = None
    self._crit = torch.compile(self._crit_impl, dynamic=False)
    sd = self.log.load("latest")
    if sd is not None:
      self.actor.load_state_dict(sd["actor"])
      self.critic.load_state_dict(sd["critic"])
      self.std.data.copy_(sd["std"].to(dev))
      self.opt.load_state_dict(sd["opt"])
      self.lib.load_state_dict({k: v.to(dev) for k, v in sd["lib"].items()})
      self.it, self.train_time, self.env_steps, self.lr = sd["it"], sd["train_time"], sd["env_steps"], sd["lr"]
      self.best_eval = sd.get("best_eval", -1.0)
      print(f"[resume] it={self.it} train_time={self.train_time / 3600:.2f}h")

  def critic_input(self, s, goal_clean):
    return self._crit(s, self.sim.xi, self.sim.frames, goal_clean)

  def _crit_impl(self, s, xi, frames, goal_clean):
    return critic_features(self.kin, self.lib, s, xi, frames, body_states(self.kin, s), goal_clean, self.cfg.goal_k)

  @torch.no_grad()
  def rollout(self):
    c, sim, T, N = self.cfg, self.sim, self.cfg.steps_per_env, self.cfg.num_envs
    self.std_old = self.std.detach().clamp_min(1e-3).clone()
    buf = dict(hist=[], goal=[], crit=[], act=[], logp=[], val=[], rew=[], done=[], mu=[])
    for _ in range(T):
      s = sim.state()
      g_clean = sim.goal(s)
      g = g_clean + (torch.rand_like(g_clean) * 2 - 1) * self.goal_noise
      x = self.critic_input(s, g_clean)
      self.actor.obs_norm.update(sim.obs_hist[:, -1])
      self.actor.goal_norm.update(g)
      self.critic.norm.update(x)
      mu = self.actor(sim.obs_hist, g)
      std = self.std.clamp_min(1e-3)
      a = mu + std * torch.randn_like(mu)
      logp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
      v = self.critic(x)
      buf["hist"].append(sim.obs_hist.clone())
      buf["goal"].append(g)
      buf["crit"].append(x)
      buf["act"].append(a)
      buf["logp"].append(logp)
      buf["val"].append(v)
      buf["mu"].append(mu)
      out = sim.step(a, c.gamma)
      r = out["reward"].clone()
      if bool(out["timeout"].any()):  # bootstrap truncated episodes
        s1 = out["next_state"]
        v1 = self.critic(self.critic_input(s1, sim.goal(s1)))
        r = r + c.gamma * v1 * out["timeout"].float()
      done = out["terminated"] | out["clip_end"] | out["timeout"]
      buf["rew"].append(r)
      buf["done"].append(done.float())
      if bool(done.any()):
        ids = done.nonzero().squeeze(-1)
        start = sim.ep_start_frame[ids]
        intended = torch.clamp(self.lib.last_frame[start] - start, max=c.max_episode_steps).clamp_min(1).float()
        self.ep_stats.append(torch.stack((sim.ep_return[ids] / (R_MAX * intended), sim.ep_steps[ids].float(),
                                          out["terminated"][ids].float()), -1))
      sim.finish_episodes(done, c.gamma)
      sim.push_obs()
      self.env_steps += N
    s = sim.state()
    last_v = self.critic(self.critic_input(s, sim.goal(s)))
    B = {k: torch.stack(v) for k, v in buf.items()}
    adv = torch.zeros_like(B["rew"])
    gae = torch.zeros(N, device=self.dev)
    for t in reversed(range(T)):
      nv = last_v if t == T - 1 else B["val"][t + 1]
      delta = B["rew"][t] + c.gamma * nv * (1 - B["done"][t]) - B["val"][t]
      gae = delta + c.gamma * c.lam * (1 - B["done"][t]) * gae
      adv[t] = gae
    B["ret"] = adv + B["val"]
    B["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return B

  def update(self, B):
    c = self.cfg
    n = B["act"].shape[0] * B["act"].shape[1]
    flat = {k: v.reshape(n, *v.shape[2:]) for k, v in B.items()}
    mb = n // c.minibatches
    stats = []
    for _ in range(c.epochs):
      perm = torch.randperm(n, device=self.dev)
      for k in range(c.minibatches):
        idx = perm[k * mb : (k + 1) * mb]
        mu = self.actor(flat["hist"][idx], flat["goal"][idx])
        std = self.std.clamp_min(1e-3)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(flat["act"][idx]).sum(-1)
        ent = dist.entropy().sum(-1).mean()
        with torch.no_grad():  # adaptive learning rate (rsl_rl)
          old_mu, so = flat["mu"][idx], self.std_old
          kl = (torch.log(std / so) + (so**2 + (old_mu - mu) ** 2) / (2 * std**2) - 0.5).sum(-1).mean()
          if kl > 2 * c.desired_kl:
            self.lr = max(1e-5, self.lr / 1.5)
          elif kl < c.desired_kl / 2:
            self.lr = min(1e-2, self.lr * 1.5)
          for g in self.opt.param_groups:
            g["lr"] = self.lr
        ratio = torch.exp(logp - flat["logp"][idx])
        adv = flat["adv"][idx]
        surr = -torch.min(ratio * adv, torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv).mean()
        v = self.critic(flat["crit"][idx])
        v_old = flat["val"][idx]
        v_clip = v_old + (v - v_old).clamp(-c.clip, c.clip)
        ret = flat["ret"][idx]
        v_loss = torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
        loss = surr + c.value_coef * v_loss - c.entropy_coef * ent
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, c.max_grad_norm)
        self.opt.step()
        stats.append(torch.stack((surr.detach(), v_loss.detach(), ent.detach(), kl)))
    s = torch.stack(stats).mean(0)
    return dict(surrogate=s[0], value_loss=s[1], entropy=s[2], kl=s[3])

  def evaluate(self):
    if self._evaluator is None:
      self._evaluator = Evaluator(self.lib, self.dev)
    return self._evaluator.run(lambda h, g: self.actor(h, g))

  def _run_eval(self):
    te = time.time()
    res = self.evaluate()
    res.update(it=self.it, train_h=self.train_time / 3600, env_steps=self.env_steps, eval_seconds=time.time() - te)
    self.log.log_eval(res)
    print(f"[eval] it={self.it} train_h={res['train_h']:.2f} return={res['norm_return']:.4f} failed={res['failed']}", flush=True)
    if res["norm_return"] > self.best_eval:
      self.best_eval = res["norm_return"]
      self.log.save(dict(actor=self.actor.state_dict(), it=self.it, train_time=self.train_time, eval=res), "best")

  def state_dict(self):
    return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), std=self.std.detach().cpu(),
                opt=self.opt.state_dict(), lib=self.lib.state_dict(), it=self.it, train_time=self.train_time,
                env_steps=self.env_steps, lr=self.lr, best_eval=self.best_eval, cfg=dataclasses.asdict(self.cfg))

  def train(self):
    c = self.cfg
    last_eval = self.train_time if self.it > 0 else -1e9
    t_wall0 = time.time()
    acc = []
    while self.it < c.total_iters and self.train_time < c.max_hours * 3600:
      t0 = time.time()
      B = self.rollout()
      t1 = time.time()
      st = self.update(B)
      self.it += 1
      t2 = time.time()
      self.train_time += t2 - t0
      acc.append(dict(t_rollout=t1 - t0, t_update=t2 - t1, **st))
      if self.it % c.log_every == 0:
        rec = dict(it=self.it, train_h=self.train_time / 3600, wall_h=(time.time() - t_wall0) / 3600, env_steps=self.env_steps,
                   lr=self.lr, std=float(self.std.mean()))
        for k in acc[0]:
          rec[k] = float(np.mean([float(x[k]) for x in acc]))
        acc = []
        if self.ep_stats:
          s = torch.cat(self.ep_stats)
          rec.update(train_return=float(s[:, 0].mean()), ep_len=float(s[:, 1].mean()), term_frac=float(s[:, 2].mean()),
                     episodes=int(s.shape[0]))
          self.ep_stats = []
        rec["fps"] = c.num_envs * c.steps_per_env / max(rec["t_rollout"] + rec["t_update"], 1e-9)
        rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self.log.log(rec)
        self.log.status(state="running", it=self.it, train_h=rec["train_h"], last=rec)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
      if self.it % c.ckpt_every == 0:
        self.log.save(self.state_dict())
      if self.train_time - last_eval >= c.eval_every_min * 60:
        last_eval = self.train_time
        self._run_eval()
    self.log.save(self.state_dict())
    if c.final_eval:
      self._run_eval()
    self.log.status(state="finished", it=self.it, train_h=self.train_time / 3600)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  for f in dataclasses.fields(PPOConfig):
    if f.type is bool or f.type == "bool":
      ap.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=f.default)
    else:
      ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
  PPOSim(PPOConfig(**vars(ap.parse_args()))).train()
