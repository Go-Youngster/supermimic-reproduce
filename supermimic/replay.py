"""Time-major ring replay buffer of simulator transitions, stored on GPU."""

import torch


class Replay:
  def __init__(self, capacity: int, num_envs: int, state_dim: int, act_dim: int, obs_dim: int, ctx_dim: int, device: str):
    T, N = capacity, num_envs
    z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
    self.s, self.s1 = z(T, N, state_dim), z(T, N, state_dim)
    self.a = z(T, N, act_dim)
    self.o = z(T, N, obs_dim)
    self.xi = z(T, N, ctx_dim)
    self.r = z(T, N)
    self.f = z(T, N, dtype=torch.long)
    self.ep = z(T, N, dtype=torch.long)
    self.step = z(T, N, dtype=torch.long)
    self.term = z(T, N, dtype=torch.bool)
    self.T, self.N, self.device = T, N, device
    self.ptr, self.size = 0, 0

  def add(self, s, a, s1, o, xi, f, ep, step, term, r):
    t = self.ptr
    self.s[t], self.a[t], self.s1[t], self.o[t], self.xi[t] = s, a, s1, o, xi
    self.f[t], self.ep[t], self.step[t], self.term[t], self.r[t] = f, ep, step, term, r
    self.ptr = (t + 1) % self.T
    self.size = min(self.size + 1, self.T)

  @property
  def oldest(self) -> int:
    return (self.ptr - self.size) % self.T

  def num_transitions(self) -> int:
    return self.size * self.N

  def sample_windows(self, batch: int, k: int, recent: int | None = None):
    """k-step windows within one episode: start state, actions (B,k,A), next states (B,k,S), context."""
    span = self.size if recent is None else min(recent, self.size)
    assert span >= k
    base = (self.ptr - span) % self.T
    out_t, out_e = [], []
    need = batch
    for _ in range(20):
      n = need * 2 + 64
      off = torch.randint(0, span - k + 1, (n,), device=self.device)
      e = torch.randint(0, self.N, (n,), device=self.device)
      t0 = (base + off) % self.T
      t1 = (t0 + k - 1) % self.T
      ok = self.ep[t0, e] == self.ep[t1, e]
      out_t.append(t0[ok])
      out_e.append(e[ok])
      need -= int(ok.sum())
      if need <= 0:
        break
    t0 = torch.cat(out_t)[:batch]
    e = torch.cat(out_e)[:batch]
    tt = (t0[:, None] + torch.arange(k, device=self.device)) % self.T
    ee = e[:, None].expand_as(tt)
    return self.s[t0, e], self.a[tt, ee], self.s1[tt, ee], self.xi[t0, e]

  def sample_starts(self, batch: int, hist: int):
    """Start states for imagination with the observation history the policy saw."""
    off = torch.randint(0, self.size, (batch,), device=self.device)
    e = torch.randint(0, self.N, (batch,), device=self.device)
    t = (self.oldest + off) % self.T
    jmax = torch.minimum(self.step[t, e], off)
    j = torch.arange(hist - 1, -1, -1, device=self.device)
    tt = (t[:, None] - torch.minimum(j[None, :], jmax[:, None])) % self.T
    obs_hist = self.o[tt, e[:, None].expand_as(tt)]
    return self.s[t, e], self.xi[t, e], self.f[t, e], obs_hist
