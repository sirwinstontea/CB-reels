"""Scene set-ups for the paper's figures.

Scale: d0 = 0.01 m, h = 0.02 m.  With the paper's mu_v = 80 the explicit viscous limit then gives dt ~ 2e-4 s,
the time step the paper reports for Fig. 6, and the reported particle counts (e.g. 37 180 solvent particles in a
pool, 20-23k paper particles) correspond to metre-scale canvases at this spacing.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..params import (CASA_BINDER, CASA_SOLVENT, FIG5_BASIC, FIG5_VISCOELASTIC, FIG5_VISCOUS, FLOWER_BINDER,
                      FLOWER_SOLVENT, POLLOCK_BINDER, Config)
from ..scene import Nozzle, lattice_box, lattice_mask, lattice_sphere, paper_sheet, wall_box
from ..solver import Simulation

RED = (0.80, 0.08, 0.06)
GREEN = (0.10, 0.55, 0.15)
YELLOW = (0.95, 0.80, 0.08)
BLUE = (0.08, 0.22, 0.75)
BLACK = (0.04, 0.04, 0.05)
WHITE = (0.93, 0.92, 0.88)
TEAL = (0.05, 0.45, 0.50)
MAGENTA = (0.75, 0.10, 0.45)
ORANGE = (0.92, 0.42, 0.05)


def _paper(sim, x0, x1, y0, y1):
    top, rest = paper_sheet(x0, x1, y0, y1, sim.cfg.d0, layers=2)
    sim.add_boundary(top, True)
    sim.add_boundary(rest, False)


def base_config(**kw):
    kw.setdefault("d0", 0.01)
    kw.setdefault("h", 0.02)
    kw.setdefault("dt", 2e-4)
    kw.setdefault("adaptive", True)
    return Config(**kw)


# ----------------------------------------------------------------------------------------------- Fig. 5
def fig5(cfg: Config | None = None):
    """Fig. 5: identical pours of viscoelastic (red), viscous (green) and basic (yellow) paint onto canvas."""
    cfg = cfg or base_config()
    sim = Simulation(cfg, [FIG5_VISCOELASTIC, FIG5_VISCOUS, FIG5_BASIC], [RED, GREEN, YELLOW])
    _paper(sim, -0.45, 0.45, -0.2, 0.2)
    for k, xo in enumerate((-0.25, 0.0, 0.25)):
        c = np.zeros(3)
        c[k] = 1.0
        sim.emitters.append(Nozzle(k, c, lambda t, xo=xo: (xo, 0.0, 0.35), radius=0.016, speed=0.6,
                                   t0=0.0, t1=1.2))
    return sim, 3.0


# ----------------------------------------------------------------------------------------------- Fig. 6
def fig6_pollock(cfg: Config | None = None, seed=3):
    """Fig. 6: action painting.  Nozzles swing over the canvas dripping viscoelastic acrylic (Table 1 row 1)."""
    cfg = cfg or base_config()
    pig = [BLACK, WHITE, YELLOW, TEAL]
    sim = Simulation(cfg, [POLLOCK_BINDER], pig)
    W, H = 0.9, 0.6
    _paper(sim, -W / 2, W / 2, -H / 2, H / 2)
    rng = np.random.default_rng(seed)
    t = 0.0
    for k in range(9):
        pk = k % len(pig)
        c = np.zeros(len(pig)); c[pk] = 1.0
        a = rng.uniform(0.15, 0.32); b = rng.uniform(0.08, 0.2)
        ph = rng.uniform(0, 2 * math.pi); wv = rng.uniform(2.5, 4.5) * rng.choice([-1, 1])
        cx, cy = rng.uniform(-0.12, 0.12), rng.uniform(-0.06, 0.06)
        dur = rng.uniform(0.6, 1.0)

        def path(tt, a=a, b=b, ph=ph, wv=wv, cx=cx, cy=cy):
            return (cx + a * math.cos(wv * tt + ph), cy + b * math.sin(1.7 * wv * tt + ph), 0.30)
        sim.emitters.append(Nozzle(0, c, path, radius=rng.uniform(0.008, 0.013), speed=0.5, t0=t, t1=t + dur))
        t += dur * 0.55
    return sim, t + 2.0


# ----------------------------------------------------------------------------------------------- Fig. 7
def fig7_flower(cfg: Config | None = None, petals=6):
    """Fig. 7: watercolour flower.  Binder paint petals (Table 1 row 2) are flooded with low-viscosity solvent
    (row 3) on absorbent paper: pigment diffuses into the solvent, the solvent is absorbed, the binder stays."""
    cfg = cfg or base_config(d0=0.006, h=0.012)
    pig = [MAGENTA, YELLOW, GREEN]
    sim = Simulation(cfg, [FLOWER_BINDER, FLOWER_SOLVENT], pig)
    d0 = cfg.d0
    _paper(sim, -0.22, 0.22, -0.22, 0.22)
    # one global lattice for the paint layer (3 particles thick) so the petals cannot overlap
    P = lattice_box((-0.2, -0.22, 0), (0.2, 0.2, 3 * d0), d0)
    xy = P[:, :2]
    center = np.linalg.norm(xy, axis=1) <= 0.025
    petal = np.zeros(len(P), bool)
    for k in range(petals):
        a = 2 * math.pi * k / petals
        u = np.array([math.cos(a), math.sin(a)])
        rel = xy - 0.075 * u
        along = rel @ u
        across = rel @ np.array([-u[1], u[0]])
        petal |= (along / 0.055) ** 2 + (across / 0.028) ** 2 <= 1.0
    petal &= ~center
    stem = (np.abs(xy[:, 0]) <= 0.006) & (xy[:, 1] > -0.21) & (xy[:, 1] < -0.13) & (P[:, 2] < 2 * d0) & ~petal
    sim.add_particles(P[petal], 0, [1, 0, 0])
    sim.add_particles(P[center], 0, [0, 1, 0])
    sim.add_particles(P[stem], 0, [0, 0, 1])
    # solvent: drops of clear "brush water" falling onto the petals (wet-in-wet)
    from ..scene import dedupe
    rng = np.random.default_rng(7)
    drops = []
    for k in range(14):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(0.03, 0.12)
        rad = rng.uniform(0.016, 0.024)
        cpos = (r * math.cos(a), r * math.sin(a), 3 * d0 + rad + d0 + rng.uniform(0.0, 0.05))
        drops.append(lattice_sphere(cpos, rad, d0))
    sim.add_particles(dedupe(sim.x, np.vstack(drops), d0), 1, [0, 0, 0])
    return sim, 2.5


# ----------------------------------------------------------------------------------------------- Fig. 8
def _casa_mask(scale=1):
    """'CASA' rasterised with PIL's default font, upscaled; rows = -y."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arialbd.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("L", (300, 80), 0)
    d = ImageDraw.Draw(img)
    d.text((4, 2), "CASA", fill=255, font=font)
    a = np.array(img) > 128
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return a


def fig8_casa(cfg: Config | None = None):
    """Fig. 8: the letters of 'CASA' (viscous paint, mu_v = 40) fall one after another into a solvent pool on
    absorbent paper (solvent r_c = 2.5e-4)."""
    cfg = cfg or base_config()
    pig = [RED, BLUE, YELLOW, GREEN]
    sim = Simulation(cfg, [CASA_BINDER, CASA_SOLVENT], pig)
    d0 = cfg.d0
    W, H = 1.0, 0.5
    _paper(sim, -W / 2 - 0.05, W / 2 + 0.05, -H / 2 - 0.05, H / 2 + 0.05)
    sim.add_boundary(wall_box(-W / 2, W / 2, -H / 2, H / 2, 0.12, d0), False)
    sim.add_particles(lattice_box((-W / 2, -H / 2, 0), (W / 2, H / 2, 0.05), d0), 1, np.zeros(4))
    mask = _casa_mask()
    pixel = 0.85 / mask.shape[1]
    ncol = mask.shape[1]
    # split columns into 4 letters by empty columns
    colsum = mask.sum(0)
    letters, inside, start = [], False, 0
    for i, s in enumerate(colsum):
        if s and not inside:
            inside, start = True, i
        if not s and inside:
            inside = False
            letters.append((start, i))
    if inside:
        letters.append((start, ncol))
    sim._pending = []
    for k, (a, b) in enumerate(letters[:4]):
        m = np.zeros_like(mask)
        m[:, a:b] = mask[:, a:b]
        P = lattice_mask(m, pixel, (-0.425, -mask.shape[0] * pixel / 2), 0.16, 0.03, d0)
        c = np.zeros(4); c[k % 4] = 1.0
        sim._pending.append((0.15 + 0.45 * k, P, c))

    class Dropper:
        def emit(self, s, t, dt):
            while s._pending and s._pending[0][0] <= t:
                _, P, c = s._pending.pop(0)
                s.add_particles(P, 0, c, v=(0, 0, -0.8))
    sim.emitters.append(Dropper())
    return sim, 3.5


SCENES = {"fig5": fig5, "fig6": fig6_pollock, "fig7": fig7_flower, "fig8": fig8_casa}
