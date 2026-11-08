"""PCISPH paint solver: You et al. 2013, Sec. 4 (fluidity, diffusion, absorption).

One step:
  1. neighbours (hash grid), density (Eq. 1)
  2. elastic strain update for binder particles (Appendix A)
  3. non-pressure accelerations: gravity, viscosity (Eq. 4), elasticity (Eq. 5)
  4. pigment diffusion (Eq. 6-8), on the start-of-step state
  5. PCISPH pressure loop (Eq. 3; Solenthaler & Pajarola 2009, ref. [7])
  6. symplectic Euler advection (pigment is advected with the particles, Appendix B)
  7. absorption into the paper (Eq. 9-10), pigment deposited on the paper
"""
from __future__ import annotations

import json
import math
import os
import time as _time

import numpy as np

from . import kernels as K
from .grid import HashGrid
from .params import BINDER, Config, Liquid

_ELASTIC = {"literal": 0, "difference": 1, "symmetric": 2}
_CURV = {"abs": 0, "literal": 1, "constant": 2}
_PAIR = {"particle": 0, "pair": 1}


class Simulation:
    def __init__(self, cfg: Config, liquids: list[Liquid], pigment_rgb):
        self.cfg = cfg
        self.liquids = list(liquids)
        self.pigment_rgb = np.asarray(pigment_rgb, np.float32).reshape(-1, 3)
        self.K = self.pigment_rgb.shape[0]
        h = cfg.h
        self.cp6, self.csp, self.clap = K.coefs(h)
        L = self.liquids
        self.muv = np.array([q.mu_v for q in L], np.float64)
        self.mue = np.array([q.mu_e for q in L], np.float64)
        self.elastic = np.array([1 if (q.kind == BINDER and q.mu_e > 0) else 0 for q in L], np.int32)
        self.omega = np.array([q.omega_D for q in L], np.float64)
        self.rc = np.array([q.r_c for q in L], np.float64)
        self.gamma = np.array([q.gamma for q in L], np.float64)
        self.theta = np.array([q.theta for q in L], np.float64)
        self.alpha = np.array([q.alpha for q in L], np.float64)
        self.yld = np.array([q.yield_ for q in L], np.float64)
        self.kind_of = np.array([q.kind for q in L], np.int8)

        self.m, self.delta1 = self._prototype()
        # liquid state
        self.x = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.lid = np.zeros(0, np.int32)
        self.c = np.zeros((0, self.K))
        self.eps = np.zeros((0, 6))
        self.l = np.zeros(0)
        self.pid = np.zeros(0, np.int64)
        self._next_id = 0
        # boundary (paper + walls)
        self.bx = np.zeros((0, 3))
        self.bpaper = np.zeros(0, np.bool_)
        self.bpsi = np.zeros(0)
        self.bgrid = None
        self.paper_c = np.zeros((0, self.K))
        self.paper_stain = np.zeros((0, self.K))
        self.paper_touch = np.zeros(0, np.int8)
        self.paper_area = cfg.d0 * cfg.d0
        self.paper_bounds = None

        self.t = 0.0
        self.emitters = []
        self.stats = dict(steps=0, absorbed=0, killed=0, clamped=0, overflow=0, pcisph_iters=0, last_iters=0,
                          last_err=0.0, last_dt=cfg.dt, deposited=np.zeros(self.K), emitted=0)

    # ------------------------------------------------------------------ setup
    def _prototype(self):
        """Particle mass from a filled cubic lattice (rest density = rho0) and the PCISPH stiffness delta
        (Solenthaler & Pajarola 2009, Eq. 8) for dt = 1; delta(dt) = delta1 / dt^2."""
        cfg = self.cfg
        h, d0 = cfg.h, cfg.d0
        n = int(math.ceil(h / d0)) + 1
        r = np.arange(-n, n + 1) * d0
        P = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        rr = np.linalg.norm(P, axis=1)
        inside = rr < h
        W = self.cp6 * (h * h - rr[inside] ** 2) ** 3
        m = cfg.rho0 / W.sum()
        nb = inside & (rr > 1e-12)
        q = P[nb]
        rn = rr[nb][:, None]
        # grad_i W_ij for r_ij = x_i - x_j = -q   (prototype particle at the origin)
        g_rho = (-6.0 * self.cp6 * (h * h - rn ** 2) ** 2) * (-q)          # poly6 gradient (density)
        g_p = (self.csp * (h - rn) ** 2 / rn) * (-q)                        # spiky gradient (pressure)
        s1 = g_rho.sum(0) @ g_p.sum(0)
        s2 = np.einsum("ij,ij->", g_rho, g_p)
        beta1 = 2.0 * (m / cfg.rho0) ** 2           # beta = 2 (dt m / rho0)^2 with dt = 1
        delta1 = -1.0 / (beta1 * (-s1 - s2))
        return m, delta1

    def add_particles(self, x, liquid: int, c, v=None):
        x = np.asarray(x, np.float64).reshape(-1, 3)
        n = x.shape[0]
        if n == 0:
            return
        cv = np.zeros((n, self.K))
        cv[:] = np.asarray(c, np.float64).reshape(-1, self.K) if np.ndim(c) == 2 else np.asarray(c, np.float64)
        vv = np.zeros((n, 3)) if v is None else np.broadcast_to(np.asarray(v, np.float64), (n, 3))
        self.x = np.vstack([self.x, x])
        self.v = np.vstack([self.v, vv])
        self.lid = np.concatenate([self.lid, np.full(n, liquid, np.int32)])
        self.c = np.vstack([self.c, cv])
        self.eps = np.vstack([self.eps, np.zeros((n, 6))])
        self.l = np.concatenate([self.l, np.zeros(n)])
        self.pid = np.concatenate([self.pid, np.arange(self._next_id, self._next_id + n)])
        self._next_id += n
        self.stats["emitted"] += n

    def add_boundary(self, bx, paper: bool):
        bx = np.asarray(bx, np.float64).reshape(-1, 3)
        self.bx = np.vstack([self.bx, bx])
        self.bpaper = np.concatenate([self.bpaper, np.full(bx.shape[0], paper, np.bool_)])
        self.bgrid = None

    def finalize_boundary(self):
        cfg = self.cfg
        if self.bx.shape[0] == 0:
            self.bx = np.array([[1e6, 1e6, 1e6]])
            self.bpaper = np.zeros(1, np.bool_)
        self.bgrid = HashGrid(self.bx, cfg.h)
        bn, bc, _ = self.bgrid.neighbors(self.bx, cfg.max_boundary_neighbors * 2, same_set=True)
        self.bpsi = K.boundary_psi(self.bx, bn, bc, cfg.h, self.cp6, cfg.rho0)
        nb = self.bx.shape[0]
        self.paper_c = np.zeros((nb, self.K))
        self.paper_stain = np.zeros((nb, self.K))
        self.paper_touch = np.zeros(nb, np.int8)
        top = self.bpaper
        if top.any():
            p = self.bx[top]
            self.paper_bounds = (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())

    # ------------------------------------------------------------------ helpers
    def _remove(self, keep):
        for name in ("x", "v", "lid", "c", "eps", "l", "pid"):
            setattr(self, name, getattr(self, name)[keep])

    def _dt(self, rho_lam_visc=None):
        cfg = self.cfg
        if not cfg.adaptive or self.x.shape[0] == 0:
            return cfg.dt
        vmax = float(np.sqrt((self.v ** 2).sum(1).max())) if self.x.shape[0] else 0.0
        dt = cfg.dt
        if vmax > 1e-9:
            dt = min(dt, cfg.cfl * cfg.h / vmax)
        mu_max = float(self.muv[np.unique(self.lid)].max()) if self.x.shape[0] else 0.0
        if mu_max > 0:
            lam = 15.0 / (cfg.h ** 2 * cfg.rho0) * 1.6   # sum m/(rho rho) lap W incl. paper, with margin
            dt = min(dt, 1.0 / (mu_max * lam))
        return dt

    def _diffuse(self, rho, nbr, cnt, dt):
        """Eq. (6)-(8), explicit, sub-stepped so that dt_sub * |D| * lam <= diffusion_safety."""
        cfg = self.cfg
        n = self.x.shape[0]
        D = np.zeros((n, self.K))
        lam = np.zeros(n)
        kappa_max = 1.0 / cfg.d0 if cfg.kappa_max is None else cfg.kappa_max
        mode, pair = _CURV[cfg.curvature], _PAIR[cfg.diffusion_pairing]
        t_left = dt
        subs = 0
        cn = np.empty_like(self.c)
        while t_left > 1e-15 and subs < cfg.max_diffusion_substeps:
            K.diffusivity(self.x, rho, self.lid, self.c, nbr, cnt, self.m, cfg.h, self.csp, self.clap, self.omega,
                          mode, cfg.grad_eps, kappa_max, cfg.curvature_ops == "paper", D, lam)
            rate = float((np.abs(D).max(1) * lam).max()) if n else 0.0
            ds = t_left if rate <= 0 else min(t_left, cfg.diffusion_safety / rate)
            K.diffuse(self.x, rho, self.c, D, nbr, cnt, self.m, cfg.h, self.clap, pair, ds, cn)
            self.c, cn = cn, self.c
            t_left -= ds
            subs += 1
        self.stats["diff_substeps"] = subs

    # ------------------------------------------------------------------ one step
    def step(self, dt=None):
        cfg = self.cfg
        if self.bgrid is None:
            self.finalize_boundary()
        for e in self.emitters:
            e.emit(self, self.t, dt if dt is not None else cfg.dt)
        if dt is None:
            dt = self._dt()
        n = self.x.shape[0]
        if n == 0:
            self.t += dt
            return dt
        h = cfg.h
        g = cfg.g
        grid = HashGrid(self.x, h)
        nbr, cnt, ov = grid.neighbors(self.x, cfg.max_neighbors, same_set=True)
        bnbr, bcnt, ov2 = self.bgrid.neighbors(self.x, cfg.max_boundary_neighbors)
        self.stats["overflow"] += int(ov + ov2)

        rho = np.empty(n)
        K.density(self.x, nbr, cnt, self.bx, bnbr, bcnt, self.bpsi, self.m, h, self.cp6, rho)

        if self.elastic.any():
            K.update_strain(self.x, self.v, rho, self.lid, self.eps, nbr, cnt, self.m, h, self.csp,
                            self.elastic, self.alpha, self.yld, dt)

        a_np = np.empty((n, 3))
        K.nonpressure_acc(self.x, self.v, rho, self.lid, self.eps, nbr, cnt, self.bx, bnbr, bcnt, self.bpsi,
                          self.m, h, self.csp, self.clap, self.muv, self.mue, self.elastic, g[0], g[1], g[2],
                          _ELASTIC[cfg.elastic_form], cfg.boundary_viscosity, cfg.rho0, a_np)

        # pigment diffusion, Eq. (6)-(8)
        if self.K > 0 and self.omega.max() > 0:
            self._diffuse(rho, nbr, cnt, dt)

        # PCISPH, Eq. (3)
        delta = self.delta1 / (dt * dt)
        p = np.zeros(n)
        a_p = np.zeros((n, 3))
        rhos = np.empty(n)
        it = 0
        err_max = 0.0
        while True:
            vs = self.v + dt * (a_np + a_p)
            xs = self.x + dt * vs
            K.density(xs, nbr, cnt, self.bx, bnbr, bcnt, self.bpsi, self.m, h, self.cp6, rhos)
            err = rhos - cfg.rho0
            p = np.maximum(p + delta * err, 0.0)
            K.pressure_acc(self.x, rhos, p, nbr, cnt, self.bx, bnbr, bcnt, self.bpsi, self.m, h, self.csp, a_p)
            it += 1
            err_max = float(np.maximum(err, 0).max()) / cfg.rho0
            if (it >= cfg.min_iter and err_max < cfg.eta) or it >= cfg.max_iter:
                break
        self.stats["pcisph_iters"] += it
        self.stats["last_iters"] = it
        self.stats["last_err"] = err_max

        self.v = self.v + dt * (a_np + a_p)
        self.x = self.x + dt * self.v

        # keep particles out of the paper (numerical safeguard against tunnelling)
        if self.paper_bounds is not None:
            x0, x1, y0, y1 = self.paper_bounds
            over = (self.x[:, 0] >= x0) & (self.x[:, 0] <= x1) & (self.x[:, 1] >= y0) & (self.x[:, 1] <= y1)
            low = over & (self.x[:, 2] < cfg.contact_floor * cfg.d0)
            if low.any():
                self.x[low, 2] = cfg.contact_floor * cfg.d0
                self.v[low, 2] = np.maximum(self.v[low, 2], 0.0)
                self.stats["clamped"] += int(low.sum())

        # absorption, Eq. (9)-(10)
        if self.rc.max() > 0 and self.bpaper.any():
            H = np.zeros(n)
            if cfg.hydrostatic:
                K.column_heights(self.x, cfg.d0, H)
            absorbed = np.zeros(n, np.bool_)
            K.absorption(self.x, self.lid, self.l, H, bnbr, bcnt, self.bpaper, self.rc, self.gamma, self.theta,
                         self.muv, cfg.rho0, abs(g[2]), dt, cfg.contact_height, cfg.absorb_depth, cfg.hydrostatic,
                         absorbed)
            if absorbed.any():
                idx = np.nonzero(absorbed)[0]
                ab_nbr, ab_cnt, _ = self.bgrid.neighbors(self.x[idx], cfg.max_boundary_neighbors)
                before = self.paper_c.sum(0).copy()
                K.deposit(self.x[idx], self.c[idx], self.m / cfg.rho0, self.bx, ab_nbr, ab_cnt, self.bpaper, h,
                          self.cp6, self.paper_area, cfg.deposit_fraction, self.paper_c)
                self.stats["deposited"] += self.paper_c.sum(0) - before
                self.stats["absorbed"] += int(idx.size)
                self._remove(~absorbed)
                bnbr, bcnt = bnbr[~absorbed], bcnt[~absorbed]

        if self.bpaper.any() and self.x.shape[0]:
            K.contact_stain(self.x, self.c, self.bx, bnbr, bcnt, self.bpaper, h, self.cp6, cfg.contact_height,
                            self.paper_stain, self.paper_touch)

        dead = (self.x[:, 2] < cfg.kill_z) | ~np.isfinite(self.x).all(1)
        if dead.any():
            self.stats["killed"] += int(dead.sum())
            self._remove(~dead)

        self.t += dt
        self.stats["steps"] += 1
        self.stats["last_dt"] = dt
        return dt

    # ------------------------------------------------------------------ output
    def frame_dict(self):
        cfg = self.cfg
        return dict(
            x=self.x.astype(np.float32), v=self.v.astype(np.float32),
            kind=self.kind_of[self.lid].astype(np.int8) if self.x.shape[0] else np.zeros(0, np.int8),
            lid=self.lid.astype(np.int16), c=self.c.astype(np.float32),
            absorbed=np.zeros(self.x.shape[0], np.bool_), pid=self.pid,
            paper_x=self.bx[self.bpaper].astype(np.float32), paper_c=self.paper_c[self.bpaper].astype(np.float32),
            paper_stain=self.paper_stain[self.bpaper].astype(np.float32),
            paper_touch=self.paper_touch[self.bpaper].astype(np.int8),
            pigment_rgb=self.pigment_rgb, d0=np.float32(cfg.d0), h=np.float32(cfg.h), time=np.float32(self.t),
            mass=np.float32(self.m), rho0=np.float32(cfg.rho0),
        )

    def save_frame(self, path):
        np.savez_compressed(path, **self.frame_dict())

    def run(self, duration, out_dir, fps=30, log_every=1, on_frame=None):
        os.makedirs(out_dir, exist_ok=True)
        frame_dt = 1.0 / fps
        nframes = int(round(duration * fps))
        t_start = _time.time()
        if self.bgrid is None:
            self.finalize_boundary()
        self.save_frame(os.path.join(out_dir, "frame_%05d.npz" % 0))
        log = []
        for f in range(1, nframes + 1):
            t_target = f * frame_dt
            t0 = _time.time()
            while self.t < t_target - 1e-12:
                dt = self._dt()
                dt = min(dt, t_target - self.t)
                self.step(dt)
            self.save_frame(os.path.join(out_dir, "frame_%05d.npz" % f))
            rec = dict(frame=f, t=round(self.t, 5), n=int(self.x.shape[0]), steps=self.stats["steps"],
                       iters=self.stats["last_iters"], err=round(self.stats["last_err"], 5),
                       dt=self.stats["last_dt"], absorbed=self.stats["absorbed"], killed=self.stats["killed"],
                       clamped=self.stats["clamped"], overflow=self.stats["overflow"],
                       sec=round(_time.time() - t0, 2))
            log.append(rec)
            if on_frame:
                on_frame(self, f)
            if log_every and f % log_every == 0:
                print(json.dumps(rec), flush=True)
        with open(os.path.join(out_dir, "run_log.json"), "w") as fh:
            json.dump(dict(log=log, total_sec=round(_time.time() - t_start, 1),
                           cfg={k: (v if not isinstance(v, tuple) else list(v)) for k, v in vars(self.cfg).items()},
                           liquids=[vars(q) for q in self.liquids], mass=self.m), fh, indent=1)
