"""GPU motion library with return-based reset sampling (Appendix D.1)."""

import numpy as np
import torch


class MotionLibrary:
  def __init__(self, path: str, device: str, clip_ids=None, bin_seconds: float = 1.0):
    d = np.load(path, allow_pickle=False)
    starts, lens = d["clip_start"], d["clip_len"]
    names = d["names"]
    if clip_ids is None:
      clip_ids = np.arange(len(lens))
    clip_ids = np.asarray(clip_ids)
    sel = np.concatenate([np.arange(starts[c], starts[c] + lens[c]) for c in clip_ids])
    self.names = [str(names[c]) for c in clip_ids]
    self.fps = float(d["fps"])
    t = lambda x: torch.tensor(np.ascontiguousarray(x), device=device)  # noqa: E731
    self.joint_pos = t(d["joint_pos"][sel])
    self.joint_vel = t(d["joint_vel"][sel])
    self.body_pos = t(d["body_pos_w"][sel])
    self.body_quat = t(d["body_quat_w"][sel])
    self.body_lin_vel = t(d["body_lin_vel_w"][sel])
    self.body_ang_vel = t(d["body_ang_vel_w"][sel])
    lens = lens[clip_ids]
    new_starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    self.num_clips = len(lens)
    self.num_frames = int(lens.sum())
    self.clip_start = t(new_starts).long()
    self.clip_len = t(lens).long()
    clip_of_frame = np.repeat(np.arange(len(lens)), lens)
    self.clip_of_frame = t(clip_of_frame).long()
    self.last_frame = (self.clip_start + self.clip_len - 1)[self.clip_of_frame]
    self.device = device

    # 1 s bins over the whole timeline; a bin never crosses a clip boundary.
    bin_len = int(round(bin_seconds * self.fps))
    nb_per_clip = np.ceil(lens / bin_len).astype(np.int64)
    bin_clip = np.repeat(np.arange(len(lens)), nb_per_clip)
    bin_off = np.concatenate([np.arange(n) for n in nb_per_clip]) * bin_len
    bin_start = new_starts[bin_clip] + bin_off
    bin_end = np.minimum(bin_start + bin_len, new_starts[bin_clip] + lens[bin_clip])
    self.num_bins = len(bin_start)
    self.bin_start = t(bin_start).long()
    self.bin_size = t(bin_end - bin_start).long()
    self.bin_remaining = t(new_starts[bin_clip] + lens[bin_clip] - bin_start).long()
    self.bin_return = torch.zeros(self.num_bins, device=device)  # normalized running return G_b/G_max,b
    self.bin_visits = torch.zeros(self.num_bins, device=device)
    self.past_weight = torch.full((self.num_bins,), 1.0 / self.num_bins, device=device)

  @property
  def hours(self) -> float:
    return self.num_frames / self.fps / 3600

  def frame_bins(self, frames: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(self.bin_start, frames, right=True) - 1

  def sample_starts(self, n: int, mix=(0.5, 0.4, 0.1), temp: float = 25.0) -> torch.Tensor:
    """Mixture of uniform / softmax(-25 G_b/G_max) / past-softmax-weight over bins."""
    soft = torch.softmax(-temp * self.bin_return, dim=0)
    self.past_weight.mul_(0.999).add_(0.001 * soft)
    u = torch.rand(n, device=self.device)
    probs_hard = soft
    probs_past = self.past_weight / self.past_weight.sum()
    b_uni = torch.randint(0, self.num_bins, (n,), device=self.device)
    b_hard = torch.multinomial(probs_hard, n, replacement=True)
    b_past = torch.multinomial(probs_past, n, replacement=True)
    b = torch.where(u < mix[0], b_uni, torch.where(u < mix[0] + mix[1], b_hard, b_past))
    off = (torch.rand(n, device=self.device) * self.bin_size[b]).long()
    frames = self.bin_start[b] + off
    # Leave at least one transition before the clip end.
    return torch.minimum(frames, self.last_frame[frames] - 1).clamp_min(self.clip_start[self.clip_of_frame[frames]])

  def update_bin_returns(self, bins: torch.Tensor, norm_returns: torch.Tensor, alpha: float = 0.1):
    if bins.numel() == 0:
      return
    s = torch.zeros(self.num_bins, device=self.device).index_add_(0, bins, norm_returns)
    c = torch.zeros(self.num_bins, device=self.device).index_add_(0, bins, torch.ones_like(norm_returns))
    m = c > 0
    first = m & (self.bin_visits == 0)
    mean = s[m] / c[m]
    upd = self.bin_return[m] * (1 - alpha) + alpha * mean
    upd = torch.where(first[m], mean, upd)
    self.bin_return[m] = upd
    self.bin_visits += c

  def future_frames(self, frames: torch.Tensor, k: int) -> torch.Tensor:
    """Global indices of frames t+1..t+k, clamped to the clip's last frame. Shape (..., k)."""
    off = torch.arange(1, k + 1, device=frames.device)
    f = frames.unsqueeze(-1) + off
    return torch.minimum(f, self.last_frame[frames].unsqueeze(-1))

  def state_dict(self):
    return {"bin_return": self.bin_return, "bin_visits": self.bin_visits, "past_weight": self.past_weight}

  def load_state_dict(self, sd):
    for k, v in sd.items():
      getattr(self, k).copy_(v)
