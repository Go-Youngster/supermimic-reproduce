"""Interactive MuJoCo viewer (browser, viser) for the trained tracking policies.

The selected policy runs live in the training simulator (MuJoCo Warp via mjlab, one world, evaluation
settings: no domain randomisation, no noise) and the robot is streamed to a viser web scene, together with
a translucent ghost of the reference motion and, optionally, an orange ghost of the world model's
closed-loop imagination (re-synchronised to the simulator every H steps, as in training). Stepping pauses
while no browser is connected.

  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/sim_viewer.py --port 8800
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mjviser import ViserMujocoScene  # noqa: E402
from mjviser.conversions import merge_geoms  # noqa: E402

from supermimic.motion_lib import MotionLibrary  # noqa: E402
from supermimic.networks import AMP, Policy, WorldModel, state_error  # noqa: E402
from supermimic.sim_env import TrackingSim  # noqa: E402
from supermimic.task import GOAL_FRAME_DIM, OBS_DIM, R_MAX, goal_obs, proprio_obs  # noqa: E402
from supermimic.vis import align_ref, build_scene, qpos_index, state_to_qpos  # noqa: E402

H_IMAG = 32
KIND = {"supermimic": "SuperMimic", "ppo_sim": "PPO-Sim", "ppo_wm": "PPO-WM"}


def discover(main_run: str, ppo_run: str) -> dict:
  """label -> (run, ckpt); main/PPO best first, then every other run's best."""
  out = {}
  runs = sorted(p for p in (ROOT / "runs").iterdir() if (p / "config.json").exists())
  runs.sort(key=lambda p: (p.name != main_run, p.name != ppo_run, p.name))
  for p in runs:
    meta = json.loads((p / "config.json").read_text()).get("meta", {})
    ev = [json.loads(x) for x in (p / "eval.jsonl").read_text().splitlines() if x.strip()] if (p / "eval.jsonl").exists() else []
    best = max(ev, key=lambda e: e["norm_return"]) if ev else None
    kind = KIND.get(meta.get("kind"), meta.get("kind", "?"))
    for ck in ("best", "latest"):
      if not (p / "ckpt" / f"{ck}.pt").exists() or (ck == "latest" and p.name != main_run):
        continue
      tag = f" (it {best['it']}, 评估 {best['norm_return']:.3f})" if ck == "best" and best else ""
      out[f"{kind} · {p.name} · {'最佳' if ck == 'best' else '最新'}{tag}"] = (p.name, ck)
  return out


class Viewer:
  def __init__(self, a):
    self.a = a
    self.dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    self.lib = MotionLibrary(str(ROOT / "data/motions/lafan1.npz"), self.dev)
    # compile=True fuses the reward/termination kernels (~6 ms -> ~1 ms per step with one world on a shared GPU).
    self.sim = TrackingSim(self.lib, 1, self.dev, train=False, max_episode_steps=10**9, seed=0, compile=True)
    self.kin = self.sim.kin
    self.ids = torch.zeros(1, dtype=torch.long, device=self.dev)
    self.policies = discover(a.main_run, a.ppo_run)
    self.cache: dict = {}
    self.wm, self.wm_bf16, self.wm_info = self._load_wm(a.main_run)
    self.lock = threading.Lock()
    self.pending: list = []
    self.playing, self.speed, self.follow, self.respawn, self.wm_on = True, 1.0, True, True, False
    self.clip = "dance2_subject1" if "dance2_subject1" in self.lib.names else self.lib.names[0]
    self.start_s = 0.0
    self.last_push = None
    self.cam_xy = np.zeros(2)
    self._build_scene()
    self._build_gui()
    self.label = next(iter(self.policies))
    self._set_policy(self.label)
    with torch.no_grad():  # trigger torch.compile before the first browser connects
      self.sim.step(torch.zeros(1, len(self.kin.default_joint_pos), device=self.dev))
    self._reset()

  # ------------------------------------------------------------------ loading
  def _load(self, label):
    if label not in self.cache:
      run, ck = self.policies[label]
      cfg = json.loads((ROOT / f"runs/{run}/config.json").read_text())["config"]
      sd = torch.load(ROOT / f"runs/{run}/ckpt/{ck}.pt", map_location=self.dev, weights_only=False)
      pol = Policy(OBS_DIM, cfg["hist"], GOAL_FRAME_DIM * cfg["goal_k"]).to(self.dev).eval()
      pol.load_state_dict(sd["policy"] if "policy" in sd else sd["actor"])
      self.cache[label] = (pol, bool(cfg.get("bf16", False)), int(sd.get("it", -1)))
    return self.cache[label]

  def _load_wm(self, run):
    p = ROOT / f"runs/{run}/ckpt/latest.pt"
    if not p.exists():
      return None, False, None
    cfg = json.loads((ROOT / f"runs/{run}/config.json").read_text())["config"]
    sd = torch.load(p, map_location=self.dev, weights_only=False)
    if "wm" not in sd:
      return None, False, None
    wm = WorldModel(self.kin, hidden=(cfg["wm_hidden"],) * 3).to(self.dev).eval()
    wm.load_state_dict(sd["wm"])
    return wm, bool(cfg.get("bf16", False)), dict(run=run, it=int(sd.get("it", -1)))

  # ------------------------------------------------------------------ scene
  def _build_scene(self):
    s = self.server = viser.ViserServer(host=self.a.host, port=self.a.port, label="SuperMimic · MuJoCo 仿真可视化")
    s.gui.configure_theme(control_width="medium", dark_mode=True, show_logo=False, show_share_button=False,
                          brand_color=(46, 160, 120))
    s.scene.set_up_direction("+z")
    s.initial_camera.position = (2.6, -2.9, 1.0)
    s.initial_camera.look_at = (0.0, 0.0, 0.0)
    self.m = build_scene(copies=(("robot_", None),))
    self.d = mujoco.MjData(self.m)
    self.qi = qpos_index(self.m, "robot_")
    self.scene = ViserMujocoScene(s, self.m, num_envs=1)
    self.scene.camera_tracking_enabled = False
    self.mg = build_scene(copies=(("g_", None),), floor=False)
    self.qg = qpos_index(self.mg, "g_")
    self.dg = mujoco.MjData(self.mg)
    self.ghost_ref = self._ghost("/ghost_ref", (77, 179, 255), 0.35)
    self.ghost_wm = self._ghost("/ghost_wm", (255, 140, 26), 0.5)
    for _, h in self.ghost_wm:
      h.visible = False

  def _ghost(self, root, color, opacity):
    out = []
    for b in range(1, self.mg.nbody):
      geoms = [g for g in range(self.mg.ngeom) if self.mg.geom_bodyid[g] == b and self.mg.geom_group[g] == 2]
      if not geoms:
        continue
      mesh = merge_geoms(self.mg, geoms)
      h = self.server.scene.add_mesh_simple(f"{root}/b{b}", np.asarray(mesh.vertices, np.float32),
                                            np.asarray(mesh.faces, np.int32), color=color, opacity=opacity,
                                            cast_shadow=False, receive_shadow=False)
      out.append((b, h))
    return out

  def _place_ghost(self, handles, q, off):
    self.dg.qpos[self.qg] = q
    mujoco.mj_kinematics(self.mg, self.dg)
    for b, h in handles:
      h.position = self.dg.xpos[b] + off
      h.wxyz = self.dg.xquat[b]

  # ------------------------------------------------------------------ gui
  def _build_gui(self):
    g = self.server.gui
    g.add_markdown("**SuperMimic 复现 · 策略在 MuJoCo Warp 中实时运行**  \n灰色实体 = 策略控制的 G1；蓝色半透明 = 参考动作；"
                   "橙色 = 世界模型想象(可选)。拖动鼠标旋转视角，滚轮缩放。")
    with g.add_folder("回放"):
      self.g_policy = g.add_dropdown("策略", options=list(self.policies), initial_value=next(iter(self.policies)))
      self.g_clip = g.add_dropdown("动作片段", options=list(self.lib.names), initial_value=self.clip)
      self.g_start = g.add_slider("起始时间 (s)", min=0.0, max=self._clip_len_s(self.clip), step=0.5, initial_value=0.0)
      self.g_play = g.add_button("暂停")
      self.g_reset = g.add_button("从起始时间重新开始")
      self.g_speed = g.add_slider("播放速度 (× 实时)", min=0.1, max=2.0, step=0.05, initial_value=1.0)
    with g.add_folder("显示"):
      self.g_ghost = g.add_checkbox("参考动作幽灵 (蓝)", initial_value=True)
      self.g_align = g.add_checkbox("参考根部对齐到机器人 (与奖励一致)", initial_value=True)
      self.g_wm = g.add_checkbox("世界模型想象幽灵 (橙)", initial_value=False, disabled=self.wm is None)
      self.g_follow = g.add_checkbox("相机跟随机器人", initial_value=True)
      self.g_respawn = g.add_checkbox("跌倒后在参考位置重生", initial_value=True)
    with g.add_folder("外力扰动"):
      self.g_push_v = g.add_slider("推力 (根速度增量 m/s)", min=0.3, max=3.0, step=0.1, initial_value=1.2)
      self.g_push = g.add_button("随机水平方向推一下")
    with g.add_folder("状态"):
      self.g_stats = g.add_markdown("")

    self.g_policy.on_update(lambda _: self._queue(("policy", self.g_policy.value)))
    self.g_clip.on_update(lambda _: self._queue(("clip", self.g_clip.value)))
    self.g_start.on_update(lambda _: self._queue(("start", float(self.g_start.value))))
    self.g_reset.on_click(lambda _: self._queue(("reset", None)))
    self.g_push.on_click(lambda _: self._queue(("push", float(self.g_push_v.value))))

    @self.g_play.on_click
    def _(_):
      self.playing = not self.playing
      self.g_play.label = "暂停" if self.playing else "继续"

    @self.g_speed.on_update
    def _(_):
      self.speed = float(self.g_speed.value)

    @self.g_ghost.on_update
    def _(_):
      for _b, h in self.ghost_ref:
        h.visible = bool(self.g_ghost.value)

    @self.g_wm.on_update
    def _(_):
      self.wm_on = bool(self.g_wm.value) and self.wm is not None
      for _b, h in self.ghost_wm:
        h.visible = self.wm_on
      self._queue(("resync", None))

    @self.g_follow.on_update
    def _(_):
      self.follow = bool(self.g_follow.value)

    @self.g_respawn.on_update
    def _(_):
      self.respawn = bool(self.g_respawn.value)

  def _queue(self, item):
    with self.lock:
      self.pending.append(item)

  def _clip_len_s(self, name) -> float:
    c = self.lib.names.index(name)
    return max(0.5, float(self.lib.clip_len[c] - 2) / 50.0)

  # ------------------------------------------------------------------ sim control
  def _set_policy(self, label):
    self.label = label
    self.pol, self.bf16, self.it = self._load(label)

  def _reset(self, frame: int | None = None):
    c = self.lib.names.index(self.clip)
    start = int(self.lib.clip_start[c])
    if frame is None:
      frame = start + int(round(self.start_s * 50))
    frame = min(max(frame, start), start + int(self.lib.clip_len[c]) - 2)
    self.sim.reset(self.ids, frames=torch.tensor([frame], device=self.dev), perturb=False)
    self.fallen_at = None
    self.rew_hist: list = []
    self.ep_rew, self.ep_steps = 0.0, 0
    if frame == start + int(round(self.start_s * 50)):
      self.falls = 0
    self.k_imag = 0
    self.im = None
    self.wm_err = float("nan")
    pos = self.sim.state()[0, :2].cpu().numpy()
    self.cam_xy = pos.copy()

  def _apply_pending(self):
    with self.lock:
      items, self.pending = self.pending, []
    for kind, v in items:
      if kind == "policy":
        self._set_policy(v)
        self._reset()
      elif kind == "clip":
        self.clip = v
        self.start_s = 0.0
        self.g_start.max = self._clip_len_s(v)
        self.g_start.value = 0.0
        self._reset()
      elif kind == "start":
        if abs(v - self.start_s) > 1e-6:
          self.start_s = v
          self._reset()
      elif kind == "reset":
        self._reset()
      elif kind == "resync":
        self.k_imag = 0
      elif kind == "push":
        ang = np.random.uniform(0, 2 * np.pi)
        qv = self.sim.sim.data.qvel
        qv[0, self.sim.fv[0]] += v * float(np.cos(ang))
        qv[0, self.sim.fv[1]] += v * float(np.sin(ang))
        self.last_push = (v, float(np.degrees(ang)), time.time())

  @torch.no_grad()
  def step(self):
    sim = self.sim
    s = sim.state()
    if self.wm_on:
      if self.k_imag % H_IMAG == 0:
        self.im = [s.clone(), sim.obs_hist.clone(), sim.frames.clone()]
      else:
        self.wm_err = float(((state_error(self.im[0], s) / self.wm.nominal) ** 2).mean())
      AMP["bf16"] = self.wm_bf16
      xi = sim.context()
      a_im = self.pol(self.im[1], goal_obs(self.lib, self.im[0], self.im[2], sim.goal_k))
      s_im = self.wm(self.im[0], a_im, xi)
      o = proprio_obs(self.kin, s_im, xi, a_im)
      self.im = [s_im, torch.cat((self.im[1][:, 1:], o.unsqueeze(1)), 1),
                 torch.minimum(self.im[2] + 1, self.lib.last_frame[self.im[2]])]
      self.k_imag += 1
    AMP["bf16"] = self.bf16
    out = sim.step(self.pol(sim.obs_hist, sim.goal(s)))
    sim.push_obs()
    flags = torch.stack((out["reward"][0] / R_MAX, out["terminated"][0].float(), out["clip_end"][0].float())).cpu()
    r, terminated, clip_end = float(flags[0]), bool(flags[1] > 0), bool(flags[2] > 0)  # one sync per step
    self.rew_hist = (self.rew_hist + [r])[-50:]
    self.ep_rew += r
    self.ep_steps += 1
    if terminated and self.fallen_at is None:
      self.fallen_at = self.ep_steps
      self.falls += 1
    if self.fallen_at is not None and self.ep_steps - self.fallen_at >= 50 and self.respawn:
      self._respawn()
    elif clip_end:
      self._reset()

  def _respawn(self):
    falls = self.falls
    self._reset(frame=int(self.sim.frames[0]))
    self.falls = falls

  def render(self):
    f = self.sim.frames[0]
    parts = [state_to_qpos(self.sim.state()[0]), self.lib.body_pos[f, 0], self.lib.body_quat[f, 0], self.lib.joint_pos[f]]
    wm = self.wm_on and self.im is not None
    if wm:
      parts.append(state_to_qpos(self.im[0][0]))
    arr = torch.cat(parts).double().cpu().numpy()  # one device->host copy
    q, ref = arr[:36], arr[36:72]
    if self.follow:
      self.cam_xy += 0.2 * (q[:2] - self.cam_xy)
    off = np.array([-self.cam_xy[0], -self.cam_xy[1], 0.0])
    qd = q.copy()
    qd[:2] -= self.cam_xy
    self.d.qpos[self.qi] = qd
    mujoco.mj_kinematics(self.m, self.d)
    self.scene.update_from_mjdata(self.d)
    self.scene.fixed_bodies_frame.position = off
    if self.g_ghost.value:
      self._place_ghost(self.ghost_ref, align_ref(ref, q) if self.g_align.value else ref, off)
    if wm:
      self._place_ghost(self.ghost_wm, arr[72:108], off)

  def stats(self, step_ms: float, rate: float):
    c = self.lib.names.index(self.clip)
    f = int(self.sim.frames[0]) - int(self.lib.clip_start[c])
    status = "**已跌倒 (回合终止条件触发)**" if self.fallen_at is not None else "跟踪中"
    lines = [
      f"**片段** {self.clip} · {f / 50:.1f} s / {float(self.lib.clip_len[c]) / 50:.1f} s",
      f"**策略** {self.label.split(' (')[0]} · it {self.it}",
      f"**跟踪奖励** 1 s 均值 {np.mean(self.rew_hist) if self.rew_hist else float('nan'):.2f} · 本段均值 "
      f"{self.ep_rew / max(self.ep_steps, 1):.2f}",
      f"**状态** {status} · 累计跌倒 {self.falls}",
      f"**速度** {rate:.2f}× 实时 · 每控制步 {step_ms:.1f} ms",
    ]
    if self.wm_on:
      lines.append(f"**世界模型** 想象第 {self.k_imag % H_IMAG}/{H_IMAG} 步 · 归一化误差 {self.wm_err:.3f}"
                   f" (WM 来自 {self.wm_info['run']} it {self.wm_info['it']})")
    if self.last_push and time.time() - self.last_push[2] < 4:
      lines.append(f"**推力** 根速度 +{self.last_push[0]:.1f} m/s · 水平方向 {self.last_push[1]:.0f}°")
    if not self.server.get_clients():
      lines.append("_无浏览器连接，仿真已暂停_")
    self.g_stats.content = "  \n".join(lines)

  def run(self):
    last_stats, last_render, t_hist = 0.0, 0.0, []
    while True:
      self._apply_pending()
      if not self.playing or not self.server.get_clients():
        if time.time() - last_stats > 1.0:
          self.stats(float("nan"), 0.0)
          last_stats = time.time()
        time.sleep(0.05)
        continue
      t0 = time.perf_counter()
      self.step()
      if t0 - last_render >= 1 / 30:  # stream at <= 30 Hz; physics/policy still run every 20 ms step
        self.render()
        last_render = t0
      el = time.perf_counter() - t0
      target = 0.02 / max(self.speed, 0.05)
      if el < target:
        time.sleep(target - el)
      t_hist = (t_hist + [(el, time.perf_counter() - t0)])[-50:]
      if time.time() - last_stats > 0.5:
        step_ms = 1000 * float(np.mean([x[0] for x in t_hist]))
        rate = 0.02 / float(np.mean([x[1] for x in t_hist]))
        self.stats(step_ms, rate)
        last_stats = time.time()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--port", type=int, default=8800)
  ap.add_argument("--main_run", default="sm_lafan1_main")
  ap.add_argument("--ppo_run", default="ppo_lafan1_1gpu")
  a = ap.parse_args()
  v = Viewer(a)
  print(f"[viewer] http://{a.host}:{a.port}  policies={len(v.policies)}", flush=True)
  v.run()


if __name__ == "__main__":
  main()
