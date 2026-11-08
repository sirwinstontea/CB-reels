"""Scene building blocks: particle blocks, paper sheet, walls, nozzles (pouring / dripping)."""
from __future__ import annotations

import math

import numpy as np


def lattice_box(lo, hi, d0, jitter=0.0, seed=0):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    axes = [np.arange(lo[k] + 0.5 * d0, hi[k] - 0.25 * d0, d0) for k in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    if jitter:
        P += np.random.default_rng(seed).uniform(-jitter, jitter, P.shape) * d0
    return P


def dedupe(existing, new, d0, frac=0.8):
    """Drop points of `new` closer than frac*d0 to `existing` or to each other (avoids exploding overlaps)."""
    from scipy.spatial import cKDTree
    new = np.asarray(new, float).reshape(-1, 3)
    if new.shape[0] == 0:
        return new
    keep = np.ones(new.shape[0], bool)
    if existing is not None and len(existing):
        d, _ = cKDTree(existing).query(new, k=1)
        keep &= d >= frac * d0
    t = cKDTree(new)
    for i, j in sorted(t.query_pairs(frac * d0)):
        if keep[i] and keep[j]:
            keep[j] = False
    return new[keep]


def lattice_sphere(center, radius, d0):
    c = np.asarray(center, float)
    P = lattice_box(c - radius, c + radius, d0)
    return P[np.linalg.norm(P - c, axis=1) <= radius]


def lattice_mask(mask, pixel, origin_xy, z0, height, d0):
    """Extrude a boolean image mask (rows = -y, cols = +x) into a block of particles: used for the CASA letters."""
    mask = np.asarray(mask, bool)
    ny, nx = mask.shape
    w, hgt = nx * pixel, ny * pixel
    P = lattice_box((origin_xy[0], origin_xy[1], z0), (origin_xy[0] + w, origin_xy[1] + hgt, z0 + height), d0)
    col = np.clip(((P[:, 0] - origin_xy[0]) / pixel).astype(int), 0, nx - 1)
    row = np.clip(ny - 1 - ((P[:, 1] - origin_xy[1]) / pixel).astype(int), 0, ny - 1)
    return P[mask[row, col]]


def paper_sheet(x0, x1, y0, y1, d0, layers=2):
    """Paper particles: `layers` lattice layers whose top surface is z = 0 (centres at -d0/2, -3d0/2, ...)."""
    xs = np.arange(x0 + 0.5 * d0, x1, d0)
    ys = np.arange(y0 + 0.5 * d0, y1, d0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    out = []
    for k in range(layers):
        out.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, -(k + 0.5) * d0)], 1))
    top = out[0]
    rest = np.vstack(out[1:]) if layers > 1 else np.zeros((0, 3))
    return top, rest


def wall_box(x0, x1, y0, y1, z1, d0, layers=2):
    """Side walls (no floor) standing on the paper, for a solvent pool."""
    pts = []
    zs = np.arange(0.5 * d0, z1, d0)
    for k in range(layers):
        o = (k + 0.5) * d0
        xs = np.arange(x0 - o, x1 + o + 1e-9, d0)
        ys = np.arange(y0 - o, y1 + o + 1e-9, d0)
        for z in zs:
            pts += [np.stack([xs, np.full_like(xs, y0 - o), np.full_like(xs, z)], 1),
                    np.stack([xs, np.full_like(xs, y1 + o), np.full_like(xs, z)], 1),
                    np.stack([np.full_like(ys, x0 - o), ys, np.full_like(ys, z)], 1),
                    np.stack([np.full_like(ys, x1 + o), ys, np.full_like(ys, z)], 1)]
    return np.vstack(pts)


def disk_pattern(radius, d0):
    """Hexagonal disk of offsets in the plane normal to z."""
    pts = []
    rows = int(radius / (d0 * math.sqrt(3) / 2)) + 1
    for r in range(-rows, rows + 1):
        y = r * d0 * math.sqrt(3) / 2
        off = 0.5 * d0 if r % 2 else 0.0
        for cidx in range(-int(radius / d0) - 1, int(radius / d0) + 2):
            x = cidx * d0 + off
            if x * x + y * y <= radius * radius + 1e-12:
                pts.append((x, y))
    if not pts:
        pts = [(0.0, 0.0)]
    return np.array(pts)


class Nozzle:
    """Emits a jet of paint/solvent: a disk of particles every time the stream has advanced one spacing.

    path(t) -> (3,) nozzle position; the jet leaves along `direction` with `speed`, plus the nozzle's own velocity
    (an artist's arm swinging over the canvas, as in action painting)."""

    def __init__(self, liquid, c, path, radius, speed, t0, t1, direction=(0, 0, -1), jitter=0.1, seed=0):
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.liquid = liquid
        self.c = np.asarray(c, float)
        self.path = path
        self.radius = radius
        self.speed = speed
        self.t0, self.t1 = t0, t1
        self.dir = np.asarray(direction, float) / np.linalg.norm(direction)
        self.acc = 0.0
        self.pattern = None
        self._prev = None

    def emit(self, sim, t, dt):
        if t < self.t0 or t > self.t1:
            self._prev = None
            return
        d0 = sim.cfg.d0
        if self.pattern is None:
            p2 = disk_pattern(self.radius, d0)
            # orthonormal basis around dir
            a = np.array([1.0, 0, 0]) if abs(self.dir[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(self.dir, a); u /= np.linalg.norm(u)
            w = np.cross(self.dir, u)
            self.pattern = p2[:, :1] * u + p2[:, 1:2] * w
        pos = np.asarray(self.path(t), float)
        vel_n = (pos - self._prev) / dt if self._prev is not None and dt > 0 else np.zeros(3)
        self._prev = pos
        self.acc += self.speed * dt
        while self.acc >= d0:
            self.acc -= d0
            x = pos + self.pattern + self.dir * self.acc
            if self.jitter:
                x = x + self.rng.uniform(-self.jitter, self.jitter, x.shape) * d0
            sim.add_particles(x, self.liquid, self.c, v=self.dir * self.speed + vel_n)
