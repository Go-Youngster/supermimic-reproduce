"""Render MuJoCo videos of the trained policies (media/ is git-ignored).

Stage 1 (GPU): every selected LAFAN1 clip is played from its first frame in the training simulator
(MuJoCo Warp, evaluation settings: no domain randomisation, no noise) by three policy groups:
  sm  : SuperMimic best checkpoint      ppo : PPO-Sim best checkpoint
  wm  : SuperMimic latest checkpoint, with its world model imagining next to the simulator in closed loop
        (re-synchronised to the true state every H steps, as in training-time imagination).
Also writes results/sim_info.json (physics / actuator / contact parameters of the compiled model).
Stage 2 (CPU pool): MuJoCo OpenGL rendering through OSMesa, PIL captions, H.264 MP4 via ffmpeg.
Outputs media/videos/<id>.mp4|jpg and media/index.json.

  CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/render_videos.py            # both stages
  .venv/bin/python scripts/render_videos.py --stage render                      # re-render from cache
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEDIA = ROOT / "media"
VIDEOS = MEDIA / "videos"
CACHE = MEDIA / "cache" / "rollouts.npz"
FPS = 25  # every other 50 Hz control step
H_IMAG = 32

SHOWCASE = ["dance2_subject1", "jumps1_subject1", "fallAndGetUp1_subject1", "fightAndSports1_subject1",
            "sprint1_subject2", "run1_subject2"]
COMPARE = ["fallAndGetUp1_subject1", "dance2_subject1", "jumps1_subject2"]  # fallback if no eval files
WM_CLIPS = ["walk1_subject1", "dance2_subject1"]
GROUPS = ("sm", "ppo", "wm")


def pick_compare(sm_run: str, ppo_run: str, names: list, n: int = 3) -> list:
  """Clips that PPO-Sim fails (lowest per-clip return first) at its best evaluation while SuperMimic does not."""
  def best(run):
    p = ROOT / f"runs/{run}/eval.jsonl"
    ev = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return max(ev, key=lambda e: e["norm_return"]) if ev else None

  sm, ppo = best(sm_run), best(ppo_run)
  if not sm or not ppo or len(ppo.get("per_clip", [])) != len(names):
    return COMPARE
  sm_failed = set(sm.get("failed_clips", []))
  order = sorted(range(len(names)), key=lambda i: ppo["per_clip"][i])
  failed = [names[i] for i in order if names[i] in set(ppo.get("failed_clips", [])) and names[i] not in sm_failed]
  rest = [names[i] for i in order if names[i] not in failed and names[i] not in sm_failed]
  return (failed + rest)[:n]


# ----------------------------------------------------------------------------- stage 1: rollouts (GPU)
def sim_info(sim) -> dict:
  import mujoco

  from supermimic.g1 import JOINT_NAMES
  from supermimic.sim_env import BIAS_RANGE, COM_RANGE, FOOT_GEOMS, FRICTION_RANGE

  m = sim.sim.mj_model
  name = lambda t, i: (mujoco.mj_id2name(m, t, i) or "").split("/")[-1]  # noqa: E731
  jid = {name(mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
  act_of = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
  kin = sim.kin
  joints = []
  for k, n in enumerate(JOINT_NAMES):
    j = jid[n]
    a, dof = act_of[j], int(m.jnt_dofadr[j])
    joints.append(dict(
      name=n, kp=round(float(m.actuator_gainprm[a, 0]), 2), kd=round(float(-m.actuator_biasprm[a, 2]), 3),
      effort=round(float(m.actuator_forcerange[a, 1]), 1), armature=round(float(m.dof_armature[dof]), 5),
      range=[round(float(x), 3) for x in m.jnt_range[j]], default=round(float(kin.default_joint_pos[k]), 3),
      action_scale=round(float(kin.action_scale[k]), 4),
    ))
  robot_bodies = [b for b in range(1, m.nbody) if m.body_mass[b] > 0]
  feet = [g for g in range(m.ngeom) if re.fullmatch(FOOT_GEOMS, name(mujoco.mjtObj.mjOBJ_GEOM, g))]
  floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
  coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
  gtype = lambda g: mujoco.mjtGeom(int(m.geom_type[g])).name.replace("mjGEOM_", "").lower()  # noqa: E731
  coll_types: dict = {}
  for g in coll:
    coll_types[gtype(g)] = coll_types.get(gtype(g), 0) + 1
  o = m.opt
  return dict(
    engine=dict(mujoco=mujoco.__version__, backend="MuJoCo Warp (GPU, batched worlds via mjlab)"),
    option=dict(
      timestep=float(o.timestep), decimation=4, control_hz=round(1 / (4 * o.timestep)),
      integrator=mujoco.mjtIntegrator(int(o.integrator)).name.replace("mjINT_", ""),
      solver=mujoco.mjtSolver(int(o.solver)).name.replace("mjSOL_", ""),
      cone=mujoco.mjtCone(int(o.cone)).name.replace("mjCONE_", ""),
      iterations=int(o.iterations), ls_iterations=int(o.ls_iterations), tolerance=float(o.tolerance),
      gravity=[float(x) for x in o.gravity], impratio=float(o.impratio), nconmax=35, njmax=250,
    ),
    model=dict(nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), nbody=int(m.nbody), ngeom=int(m.ngeom), nmesh=int(m.nmesh),
               collision_geoms=len(coll), collision_types=coll_types,
               total_mass=round(float(sum(m.body_mass[b] for b in robot_bodies)), 2)),
    bodies=[dict(name=name(mujoco.mjtObj.mjOBJ_BODY, b), mass=round(float(m.body_mass[b]), 3)) for b in robot_bodies],
    joints=joints,
    feet=dict(count=len(feet), type=gtype(feet[0]) if feet else None,
              radius=round(float(m.geom_size[feet[0], 0]), 4) if feet else None,
              half_lengths=[round(float(m.geom_size[g, 1]), 4) for g in feet[: len(feet) // 2]],
              priority=int(m.geom_priority[feet[0]]) if feet else None, condim=int(m.geom_condim[feet[0]]) if feet else None,
              friction=[round(float(x), 3) for x in m.geom_friction[feet[0]]] if feet else None),
    floor=dict(friction=[round(float(x), 3) for x in m.geom_friction[floor[0]]] if floor else None,
               solref=[float(x) for x in m.geom_solref[floor[0]]] if floor else None,
               solimp=[float(x) for x in m.geom_solimp[floor[0]]] if floor else None),
    randomization=dict(foot_friction=list(FRICTION_RANGE), encoder_bias=list(BIAS_RANGE),
                       torso_com_offset=[list(r) for r in COM_RANGE],
                       reset=dict(root_xy=0.05, root_z=0.01, roll_pitch=0.1, yaw=0.2, lin_vel_xy=0.5, lin_vel_z=0.2,
                                  ang_vel_xy=0.52, ang_vel_z=0.78, joint_pos=0.1)),
    termination=dict(root_height_error=0.25, gravity_z_error=0.8),
    generated=time.strftime("%Y-%m-%d %H:%M:%S"),
  )


def rollout(a) -> None:
  import torch

  from supermimic.motion_lib import MotionLibrary
  from supermimic.networks import AMP, Policy, WorldModel, state_error
  from supermimic.sim_env import TrackingSim
  from supermimic.task import GOAL_FRAME_DIM, OBS_DIM, R_MAX, goal_obs, proprio_obs
  from supermimic.vis import state_to_qpos

  dev = "cuda:0"
  torch.backends.cuda.matmul.allow_tf32 = True
  lib = MotionLibrary(str(ROOT / "data/motions/lafan1.npz"), dev)
  compare = pick_compare(a.sm_run, a.ppo_run, lib.names)
  print(f"[rollout] compare clips: {compare}", flush=True)
  clips = sorted(set(SHOWCASE + compare + WM_CLIPS), key=lib.names.index)
  ids = torch.tensor([lib.names.index(c) for c in clips], device=dev)
  nc = len(clips)
  sim = TrackingSim(lib, nc * len(GROUPS), dev, train=False, max_episode_steps=10**9, seed=7, compile=False)
  kin = sim.kin
  (ROOT / "results").mkdir(exist_ok=True)
  info = sim_info(sim)
  (ROOT / "results/sim_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))

  def load(run, ckpt):
    cfg = json.loads((ROOT / f"runs/{run}/config.json").read_text())["config"]
    assert cfg["hist"] == sim.hist_len and cfg["goal_k"] == sim.goal_k, (run, cfg["hist"], cfg["goal_k"])
    sd = torch.load(ROOT / f"runs/{run}/ckpt/{ckpt}.pt", map_location=dev, weights_only=False)
    pol = Policy(OBS_DIM, cfg["hist"], GOAL_FRAME_DIM * cfg["goal_k"]).to(dev).eval()
    pol.load_state_dict(sd["policy"] if "policy" in sd else sd["actor"])
    wm = None
    if "wm" in sd:
      wm = WorldModel(kin, hidden=(cfg["wm_hidden"],) * 3).to(dev).eval()
      wm.load_state_dict(sd["wm"])
    ev = [json.loads(x) for x in (ROOT / f"runs/{run}/eval.jsonl").read_text().splitlines() if x.strip()]
    ev_it = {e["it"]: e for e in ev}
    e = ev_it.get(sd.get("it")) or (sd.get("eval") or {})
    r = e.get("norm_return")
    meta = dict(run=run, ckpt=ckpt, it=int(sd.get("it", -1)), train_h=round(float(sd.get("train_time", 0.0)) / 3600, 2),
                eval=None if r is None else round(float(r), 4), failed=e.get("failed"))
    return pol, wm, bool(cfg.get("bf16", False)), meta

  pols = [load(a.sm_run, "best"), load(a.ppo_run, "best"), load(a.sm_run, "latest")]
  wm, wm_bf16 = pols[2][1], pols[2][2]
  N = sim.num_envs
  starts = lib.clip_start[ids].repeat(len(GROUPS))
  sim.reset(torch.arange(N, device=dev), frames=starts, perturb=False)
  lens = lib.clip_len[ids]
  T = int(lens.max())
  Q = torch.zeros(T, N, 36, device=dev)
  REFQ = torch.zeros(T, nc, 36, device=dev)
  WMQ = torch.zeros(T, nc, 36, device=dev)
  WERR = torch.zeros(T, nc, device=dev)
  R = torch.zeros(T, N, device=dev)
  TERM = torch.zeros(T, N, dtype=torch.bool, device=dev)
  sl = slice(2 * nc, 3 * nc)
  xi = sim.context()[sl].clone()
  im = None
  t0 = time.time()
  with torch.no_grad():
    for t in range(T):
      s = sim.state()
      f = sim.frames[:nc]
      Q[t] = state_to_qpos(s)
      REFQ[t] = torch.cat((lib.body_pos[f, 0], lib.body_quat[f, 0], lib.joint_pos[f]), -1)
      if t % H_IMAG == 0:
        im = [s[sl].clone(), sim.obs_hist[sl].clone(), sim.frames[sl].clone()]
      WMQ[t] = state_to_qpos(im[0])
      WERR[t] = ((state_error(im[0], s[sl]) / wm.nominal) ** 2).mean(-1)
      if t == T - 1:
        break
      g = sim.goal(s)
      acts = []
      for k, (pol, _, bf16, _) in enumerate(pols):
        AMP["bf16"] = bf16
        acts.append(pol(sim.obs_hist[k * nc:(k + 1) * nc], g[k * nc:(k + 1) * nc]))
      AMP["bf16"] = wm_bf16
      a_im = pols[2][0](im[1], goal_obs(lib, im[0], im[2], sim.goal_k))
      s_im = wm(im[0], a_im, xi)
      o = proprio_obs(kin, s_im, xi, a_im)
      im = [s_im, torch.cat((im[1][:, 1:], o.unsqueeze(1)), 1), torch.minimum(im[2] + 1, lib.last_frame[im[2]])]
      out = sim.step(torch.cat(acts))
      R[t] = out["reward"] / R_MAX
      TERM[t] = out["terminated"]
      sim.push_obs()
      if t % 2000 == 0:
        print(f"[rollout] t={t}/{T} {time.time() - t0:.0f}s", flush=True)
  CACHE.parent.mkdir(parents=True, exist_ok=True)
  np.savez(CACHE, Q=Q.cpu().numpy(), REFQ=REFQ.cpu().numpy(), WMQ=WMQ.cpu().numpy(), WERR=WERR.cpu().numpy(),
           R=R.cpu().numpy(), TERM=TERM.cpu().numpy(), lens=lens.cpu().numpy(),
           meta=json.dumps(dict(clips=clips, groups=GROUPS, policies=[p[3] for p in pols], dt=0.02, h_imag=H_IMAG,
                                showcase=SHOWCASE, compare=compare, wm=WM_CLIPS,
                                created=time.strftime("%Y-%m-%d %H:%M:%S"))))
  print(f"[rollout] done T={T} envs={N} in {time.time() - t0:.0f}s -> {CACHE}", flush=True)


# ----------------------------------------------------------------------------- stage 2: render (CPU pool)
def _window_activity(refq: np.ndarray, n: int) -> int:
  d = np.abs(np.diff(refq[:, 7:], axis=0)).sum(-1) + 3 * np.abs(np.diff(refq[:, :3], axis=0)).sum(-1)
  if len(d) <= n:
    return 0
  c = np.concatenate(([0.0], np.cumsum(d)))
  return int(np.argmax(c[n:] - c[:-n]))


def plan(d: dict, meta: dict) -> list[dict]:
  clips, nc = meta["clips"], len(meta["clips"])
  pol = {g: p for g, p in zip(GROUPS, meta["policies"])}
  jobs = []
  for c_name in meta.get("showcase", SHOWCASE):
    c = clips.index(c_name)
    n, L = 15 * 50, int(d["lens"][c])
    t0 = _window_activity(d["REFQ"][:L, c], n)
    t1 = min(L, t0 + n)
    e = c
    jobs.append(dict(
      id=f"showcase_{c_name}", kind="showcase", clip=c_name, t0=t0, t1=t1, policy=pol["sm"],
      robot=d["Q"][t0:t1, e], ref=d["REFQ"][t0:t1, c], rew=d["R"][max(0, t0 - 50):t1, e], rew_off=min(t0, 50),
      term=d["TERM"][:t1, e],
      title=f"SuperMimic 跟踪 {c_name}", category="SuperMimic 主实验最佳策略",
      desc=f"最佳检查点 (it {pol['sm']['it']}, 评估 {pol['sm']['eval']}) 在 MuJoCo Warp 中从片段起点开始完整回放，截取动作最剧烈的 15 s。",
    ))
  for c_name in meta.get("compare", COMPARE):
    c = clips.index(c_name)
    L = int(d["lens"][c])
    e_sm, e_ppo = c, nc + c
    fall = np.flatnonzero(d["TERM"][:L, e_ppo])
    if len(fall):
      tf = int(fall[0])
      t0, t1 = max(0, tf - 10 * 50), min(L, tf + 5 * 50)
    else:
      tf = None
      r = d["R"][:L, e_ppo]
      n = 15 * 50
      cs = np.concatenate(([0.0], np.cumsum(r)))
      t0 = int(np.argmin(cs[n:] - cs[:-n])) if L > n else 0
      t1 = min(L, t0 + n)
    jobs.append(dict(
      id=f"compare_{c_name}", kind="compare", clip=c_name, t0=t0, t1=t1, policy=pol["sm"], policy2=pol["ppo"],
      robot=d["Q"][t0:t1, e_sm], robot2=d["Q"][t0:t1, e_ppo], ref=d["REFQ"][t0:t1, c],
      rew=d["R"][max(0, t0 - 50):t1, e_sm], rew2=d["R"][max(0, t0 - 50):t1, e_ppo], rew_off=min(t0, 50),
      term=d["TERM"][:t1, e_sm], term2=d["TERM"][:t1, e_ppo], ppo_fall_s=None if tf is None else round(tf / 50, 2),
      title=f"SuperMimic vs PPO-Sim · {c_name}", category="与 PPO-Sim 基线对比",
      desc=("同一片段、同一仿真器、均从片段起点回放；" + (f"PPO-Sim 在 {tf / 50:.1f} s 处跌倒(回合终止)，截取其前 10 s 与后 5 s。"
             if tf is not None else "PPO-Sim 本次未跌倒，截取其奖励最低的 15 s。")),
    ))
  for c_name in meta.get("wm", WM_CLIPS):
    c = clips.index(c_name)
    L, n = int(d["lens"][c]), 10 * 50
    t0 = (_window_activity(d["REFQ"][:L, c], n) // H_IMAG) * H_IMAG
    t1 = min(L, t0 + n)
    e = 2 * nc + c
    jobs.append(dict(
      id=f"wm_{c_name}", kind="wm", clip=c_name, t0=t0, t1=t1, policy=pol["wm"],
      robot=d["Q"][t0:t1, e], wmq=d["WMQ"][t0:t1, c], werr=d["WERR"][t0:t1, c],
      title=f"世界模型想象 vs 仿真 · {c_name}", category="世界模型可视化",
      desc=(f"最新检查点 (it {pol['wm']['it']}) 的策略分别在 MuJoCo 仿真与学到的世界模型中闭环运行；"
            f"每 {H_IMAG} 步(0.64 s)把想象状态重置为仿真真实状态，与训练时想象窗口一致。"),
    ))
  return jobs


def _rew_ma(rew: np.ndarray, off: int, i: int) -> float:
  k = off + i
  return float(rew[max(0, k - 49):k + 1].mean()) if k >= 0 and len(rew) else float("nan")


def render_job(job: dict) -> dict:
  import imageio_ffmpeg
  from PIL import Image

  from supermimic.vis import REF_RGBA, WM_RGBA, Panel, align_ref, build_scene, overlay

  t_start = time.time()
  kind, clip, t0 = job["kind"], job["clip"], job["t0"]
  steps = range(0, job["t1"] - t0, 2)
  gray = (0.85, 0.85, 0.85)
  if kind == "compare":
    w, h = 640, 540
    model = build_scene(copies=(("robot_", None), ("ref_", REF_RGBA)), width=w, height=h)
    panels = [Panel(model, w, h, distance=3.4), Panel(model, w, h, distance=3.4)]
    size = (2 * w, h)
  elif kind == "wm":
    w, h = 960, 540
    panels = [Panel(build_scene(copies=(("robot_", None), ("wm_", WM_RGBA)), width=w, height=h), w, h)]
    size = (w, h)
  else:
    w, h = 960, 540
    panels = [Panel(build_scene(width=w, height=h), w, h)]
    size = (w, h)
  VIDEOS.mkdir(parents=True, exist_ok=True)
  out = VIDEOS / f"{job['id']}.mp4"
  tmp = VIDEOS / f".{job['id']}.tmp.mp4"
  writer = imageio_ffmpeg.write_frames(
    str(tmp), size, fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2, ffmpeg_log_level="error",
    output_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"])
  writer.send(None)
  poster_at = int(len(steps) * 0.4)
  poster = None
  for n, i in enumerate(steps):
    tc = (t0 + i) * 0.02
    if kind == "showcase":
      p = job["policy"]
      ev = "-" if p["eval"] is None else f"{p['eval']:.3f}"
      fell = bool(job["term"][: t0 + i + 1].any())
      img = panels[0].render({"robot_": job["robot"][i], "ref_": align_ref(job["ref"][i], job["robot"][i])})
      img = overlay(img, [f"SuperMimic(本复现) · {clip}",
                          f"检查点 it {p['it']} · 训练 {p['train_h']:.1f} h · 评估回报 {ev}",
                          f"片段时间 {tc:6.1f} s    跟踪奖励(1 s 均值) {_rew_ma(job['rew'], job['rew_off'], i):.2f}"],
                    badge="跌倒 · 回合终止" if fell else None,
                    legend=[(gray, "实体: 策略控制的 G1 (MuJoCo Warp, 50 Hz)"),
                            (REF_RGBA, "蓝色半透明: 参考动作 (LAFAN1, 按奖励的方式对齐到机器人根部)")])
    elif kind == "compare":
      halves = []
      for k, (name, rob, rew, term) in enumerate((("SuperMimic (世界模型 + 解析梯度)", "robot", "rew", "term"),
                                                   ("PPO-Sim (BeyondMimic 超参)", "robot2", "rew2", "term2"))):
        fell = bool(job[term][: t0 + i + 1].any())
        im = panels[k].render({"robot_": job[rob][i], "ref_": align_ref(job["ref"][i], job[rob][i])})
        halves.append(overlay(im, [name, f"跟踪奖励(1 s 均值) {_rew_ma(job[rew], job['rew_off'], i):.2f}"], size=17,
                              badge="跌倒 · 回合终止" if fell else None))
      img = np.concatenate(halves, 1)
      img = overlay(img, None, size=17, legend=[(gray, f"{clip} · 片段时间 {tc:6.1f} s · 同一仿真器/同一起点"),
                                                 (REF_RGBA, "蓝色半透明: 参考动作 (按奖励的方式对齐到各自机器人根部)")])
      img[:, w - 1:w + 1] = 20
    else:
      k = (t0 + i) % H_IMAG
      img = panels[0].render({"robot_": job["robot"][i], "wm_": job["wmq"][i]})
      img = overlay(img, [f"世界模型闭环想象 vs MuJoCo 仿真 · {clip}",
                          f"每 {H_IMAG} 步 (0.64 s) 用仿真真实状态重置想象 · 想象第 {k:2d}/{H_IMAG} 步",
                          f"片段时间 {tc:6.1f} s    归一化状态误差 {float(job['werr'][i]):.3f}"],
                    legend=[(gray, "实体: MuJoCo Warp 仿真 (策略闭环)"), (WM_RGBA, "橙色: 世界模型想象 (同一策略闭环)")])
    writer.send(np.ascontiguousarray(img))
    if n == poster_at:
      poster = img
  writer.close()
  os.replace(tmp, out)
  Image.fromarray(poster).resize((size[0] // 2, size[1] // 2)).save(VIDEOS / f"{job['id']}.jpg", quality=82)
  for p in panels:
    p.close()
  entry = {k: job[k] for k in ("id", "kind", "clip", "title", "category", "desc")}
  entry.update(file=f"videos/{job['id']}.mp4", poster=f"videos/{job['id']}.jpg", start_s=round(t0 * 0.02, 2),
               dur_s=round((job["t1"] - t0) * 0.02, 2), width=size[0], height=size[1], fps=FPS,
               size_mb=round(out.stat().st_size / 2**20, 2), render_s=round(time.time() - t_start, 1),
               policy=job["policy"], policy2=job.get("policy2"), ppo_fall_s=job.get("ppo_fall_s"))
  if kind == "showcase":
    entry["mean_reward"] = round(float(job["rew"][job["rew_off"]:].mean()), 3)
  if kind == "compare":
    entry["mean_reward"] = round(float(job["rew"][job["rew_off"]:].mean()), 3)
    entry["mean_reward2"] = round(float(job["rew2"][job["rew_off"]:].mean()), 3)
  if kind == "wm":
    entry["mean_err"] = round(float(job["werr"].mean()), 4)
  print(f"[render] {job['id']} {entry['dur_s']}s video in {entry['render_s']}s ({entry['size_mb']} MB)", flush=True)
  return entry


def render(a) -> None:
  z = np.load(CACHE)
  d = {k: z[k] for k in ("Q", "REFQ", "WMQ", "WERR", "R", "TERM", "lens")}
  meta = json.loads(str(z["meta"]))
  jobs = plan(d, meta)
  if a.only:
    jobs = [j for j in jobs if re.search(a.only, j["id"])]
  jobs.sort(key=lambda j: -(j["t1"] - j["t0"]) * (2 if j["kind"] == "compare" else 1))
  with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
    entries = pool.map(render_job, jobs, chunksize=1)
  idx_path = MEDIA / "index.json"
  old = json.loads(idx_path.read_text()) if idx_path.exists() and a.only else {}
  keep = {e["id"]: e for e in old.get("videos", [])}
  keep.update({e["id"]: e for e in entries})
  if not a.only:  # full re-render: drop videos of clips that are no longer selected
    for p in VIDEOS.glob("*"):
      if p.suffix in (".mp4", ".jpg") and p.stem not in keep:
        p.unlink()
  order = {"showcase": 0, "compare": 1, "wm": 2}
  videos = sorted(keep.values(), key=lambda e: (order.get(e["kind"], 9), e["id"]))
  idx = dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), rollout=meta, videos=videos)
  tmp = idx_path.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
  tmp.replace(idx_path)
  print(f"[render] {len(entries)} videos -> {idx_path}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", choices=("all", "rollout", "render"), default="all")
  ap.add_argument("--sm_run", default="sm_lafan1_main")
  ap.add_argument("--ppo_run", default="ppo_lafan1_1gpu")
  ap.add_argument("--workers", type=int, default=12)
  ap.add_argument("--only", default="", help="regex on video ids (render stage)")
  a = ap.parse_args()
  os.chdir(ROOT)
  if a.stage in ("all", "rollout"):
    rollout(a)
  if a.stage in ("all", "render"):
    render(a)


if __name__ == "__main__":
  main()
