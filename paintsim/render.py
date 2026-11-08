"""Ray-marched renderer for paintsim frames (You et al. 2013, Sec. 5).

The paper reconstructs the liquid surface with Zhu & Bridson's SDF (Eq. 11),
gathers pigment concentration onto a grid, gives the paper pigment density
where liquid has touched it, and renders both together in mental ray. Here the
same inputs go through a CPU ray-marcher (numba):

* liquid surface: Zhu & Bridson SDF on a grid (absorbed particles excluded),
  Schlick-Fresnel reflection of a soft studio environment (sky dome + window
  softbox) plus a Blinn-Phong highlight for the wet sheen;
* liquid body: refracting, transparent; pigment absorbs volumetrically with
  Beer-Lambert, absorbance per pigment A = -ln(max(pigment_rgb, 1e-3));
* paper: white with procedural tooth/fibre texture, tinted by the paper pigment
  map with Beer-Lambert down and back up through the layer, darker where wet,
  coloured shadows through the liquid, contact occlusion near blobs;
* key light from the upper left, linear -> sRGB output.

Public API
----------
render_frame(frame, camera='top', width=960, height=540, extent=None, **opts) -> (H, W, 3) uint8
render_sequence(frame_dir, out_mp4, fps=30, **kw) -> out_mp4
auto_extent(frame) -> (x0, x1, y0, y1); sequence_extent(paths) -> (x0, x1, y0, y1, zmax)
frame_mesh(frame, ...) -> (verts, faces, normals)
CLI: python -m paintsim.render <frame_dir_or_npz> [-o out] [--camera top|oblique] [--size 960x540]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
import time

import numpy as np
from numba import njit, prange

from .surface import (extract_mesh, paper_pigment_map, pigment_grid, splat_density_2d, write_obj,
                      zhu_bridson_sdf)

__all__ = ["render_frame", "render_sequence", "auto_extent", "sequence_extent", "content_height",
           "frame_mesh", "load_frame", "DEFAULTS"]

# Lengths are in units of the particle spacing d0 unless stated.
DEFAULTS = dict(
    spp=2,                 # samples per pixel per axis (2 -> 4 rays per pixel)
    radius=0.5,            # Zhu & Bridson particle radius r_i [d0]
    R=2.0,                 # Zhu & Bridson kernel radius [d0] (2x spacing, = h)
    sdf_cell=1.0 / 3.0,    # SDF grid spacing [d0]
    sdf_smooth=2,          # [1,2,1] smoothing passes on the SDF before tracing (0 = raw Eq. 11);
                           # damps particle-lattice bumps (wavelength ~d0) and keeps >= 2 d0 shapes
    max_cells=12_000_000,  # SDF grid cap (cell grows beyond it), 48 MB float32
    pigment_r=2.0,         # pigment gather radius r [d0] (= h)
    pigment_cell=0.5,      # pigment grid spacing [d0]
    paper_cell=0.25,       # paper map spacing [d0]
    paper_r=1.5,           # paper map splat radius [d0]
    absorb_scale=0.5,      # Beer-Lambert scale [1/d0]: one pass through d0 of c=1 liquid gives rgb^0.5
    paper_absorb_scale=None,  # same as absorb_scale unless set (paper_c is a thickness, same units)
    contact_stain=0.1,     # contact staining: stain = contact_stain * d0 * concentration (see surface.py)
    stain_source="auto",   # 'auto' (frame's paper_stain if present, else live), 'frame', 'live', 'none'
    touch_damp=0.0,        # dampness (0..1) of paper flagged by the frame's paper_touch; off by default
                           # because the flag is persistent and dried paper does not stay dark
    eta=1.38,              # refractive index of the liquid
    granulation=0.35,      # pigment settles into the paper tooth valleys
    tex_scale=1.0,         # paper texture feature size multiplier (features are ~0.1-1.6 d0)
    bump=0.035,            # paper tooth height [d0-texture units]
    paper_albedo=0.86,
    wet_darken=0.06,       # albedo loss of soaked paper
    key=0.85,              # key light irradiance (upper left)
    ambient=0.42,          # sky irradiance
    light_dir=(-0.45, 0.5, 0.74),
    shininess=180.0,
    spec=0.6,              # Blinn-Phong strength (multiplied by Fresnel)
    softbox=11.0,          # radiance of the window softbox seen in reflections
    ao=0.55,               # contact occlusion strength near liquid
    fov=26.0,              # vertical field of view [deg]
    elevation=36.0,        # oblique camera elevation [deg]
    azimuth=0.0,           # oblique camera azimuth [deg], 0 = looking along +y
    exposure=1.0,
    dither=True,
)

# indices into the float64 parameter vector passed to the kernels
(_ETA, _F0, _IK, _IA, _LX, _LY, _LZ, _SHINE, _KS, _BOXL, _ALB, _BUMP, _TEXS, _GRAN, _D0, _WETD,
 _AOK, _PIXANG, _HASLIQ, _FAR, _CAMX, _CAMY, _CAMZ, _T1X, _T1Y, _T1Z, _T2X, _T2Y, _T2Z, _NPAR) = range(30)


# ==========================================================================
# frame loading, extents, cameras
# ==========================================================================
def load_frame(src):
    """Frame as a dict of arrays (accepts a dict or a path to frame_*.npz)."""
    if isinstance(src, dict):
        return src
    with np.load(src, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _scalar(fr, key, default):
    v = fr.get(key, default)
    return float(np.asarray(v).reshape(-1)[0]) if v is not None else float(default)


def _live(fr):
    x = np.asarray(fr.get("x", np.zeros((0, 3))), np.float64).reshape(-1, 3)
    n = x.shape[0]
    c = np.asarray(fr.get("c", np.zeros((n, 1))), np.float64)
    if c.ndim == 1:
        c = c[:, None]
    ab = fr.get("absorbed")
    ab = np.zeros(n, bool) if ab is None else np.asarray(ab, bool).reshape(-1)
    return x, c, ab


def _inliers(p, d0):
    """Mask dropping stray particles (e.g. ejected by an unstable step) that sit
    far outside the bulk: keep points within one bulk-size (+10 d0) of the
    5th-95th percentile box."""
    p = np.asarray(p, np.float64)
    if p.shape[0] < 20:
        return np.ones(p.shape[0], bool)
    lo = np.percentile(p, 5, axis=0)
    hi = np.percentile(p, 95, axis=0)
    m = np.max(hi - lo) + 10 * d0
    return np.all((p >= lo - m) & (p <= hi + m), axis=1)


def auto_extent(frame, pad=0.18):
    """(x0, x1, y0, y1) around the live liquid, wet/absorbed liquid and deposits."""
    fr = load_frame(frame)
    d0 = _scalar(fr, "d0", 0.005)
    x, c, ab = _live(fr)
    pts = []
    if x.shape[0]:
        pts.append(x[_inliers(x, d0), :2])
    px = np.asarray(fr.get("paper_x", np.zeros((0, 3))), np.float64).reshape(-1, 3)
    pc = np.asarray(fr.get("paper_c", np.zeros((px.shape[0], 1))), np.float64)
    if px.shape[0] and pc.size:
        s = pc.reshape(px.shape[0], -1).sum(1)
        if s.max() > 0:
            pts.append(px[s > 0.02 * s.max(), :2])
    if not pts:
        if px.shape[0]:
            pts.append(px[:, :2])
        else:
            return (-0.1, 0.1, -0.1, 0.1)
    p = np.concatenate(pts, 0)
    lo, hi = p.min(0), p.max(0)
    size = np.maximum(hi - lo, 8 * d0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * size * (1 + 2 * pad) + 2 * d0
    return (mid[0] - half[0], mid[0] + half[0], mid[1] - half[1], mid[1] + half[1])


def content_height(frame):
    """Height of the liquid for framing (99th percentile of live z, so a few
    splash drops do not pull the camera back)."""
    fr = load_frame(frame)
    x, c, ab = _live(fr)
    live = x[~ab]
    live = live[_inliers(live, _scalar(fr, "d0", 0.005))]
    return float(max(np.percentile(live[:, 2], 99), 0.0)) if live.shape[0] else 0.0


def sequence_extent(paths, samples=8, pad=0.18):
    """Union of auto extents over up to ``samples`` evenly spaced frames, plus the
    largest content height: (x0, x1, y0, y1, zmax), one fixed camera for a clip."""
    paths = list(paths)
    if not paths:
        return (-0.1, 0.1, -0.1, 0.1, 0.0)
    idx = np.unique(np.linspace(0, len(paths) - 1, min(samples, len(paths))).round().astype(int))
    frs = [load_frame(paths[i]) for i in idx]
    ex = np.array([auto_extent(f, pad=pad) for f in frs])
    zmax = max(content_height(f) for f in frs)
    return (ex[:, 0].min(), ex[:, 1].max(), ex[:, 2].min(), ex[:, 3].max(), zmax)


def _fit_aspect(extent, aspect):
    x0, x1, y0, y1 = [float(v) for v in extent]
    w, h = x1 - x0, y1 - y0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def _camera(kind, extent, width, height, o):
    """Pinhole camera: (pos, fwd, right, up, tan_x, tan_y). 'top' shows exactly
    ``extent`` grown to the aspect ratio; 'oblique' frames the region from above
    at ``elevation`` degrees."""
    aspect = width / height
    ty = math.tan(math.radians(o["fov"]) / 2)
    tx = ty * aspect
    if kind == "top":
        x0, x1, y0, y1 = _fit_aspect(extent[:4], aspect)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        H = 0.5 * (y1 - y0) / ty
        pos = np.array([cx, cy, H])
        fwd = np.array([0.0, 0.0, -1.0])
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
    elif kind == "oblique":
        x0, x1, y0, y1 = [float(v) for v in extent[:4]]
        zmax = float(extent[4]) if len(extent) > 4 else 0.0
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        el = math.radians(o["elevation"])
        az = math.radians(o["azimuth"])
        ca, sa = math.cos(az), math.sin(az)
        # region size across and along the view direction
        w = abs(ca) * (x1 - x0) + abs(sa) * (y1 - y0)
        h = abs(sa) * (x1 - x0) + abs(ca) * (y1 - y0)
        # distance so the width fits and the foreshortened depth + height fits
        D = max(0.5 * w / tx, 0.5 * (h * math.sin(el) + zmax * math.cos(el)) / ty) * 1.05
        back = np.array([-sa * math.cos(el), -ca * math.cos(el), math.sin(el)])
        target = np.array([cx, cy, 0.5 * zmax])
        pos = target + D * back
        fwd = -back
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
    else:
        raise ValueError("camera must be 'top' or 'oblique'")
    return pos, fwd, right, up, tx, ty


def _ground_footprint(cam, extent):
    """xy box of the paper plane visible to the camera (clipped for grazing rays)."""
    pos, fwd, right, up, tx, ty = cam
    x0, x1, y0, y1 = extent[:4]
    span = max(x1 - x0, y1 - y0)
    pts = []
    for a in (-1, 1):
        for b in (-1, 1):
            d = fwd + a * tx * right + b * ty * up
            d /= np.linalg.norm(d)
            if d[2] < -1e-3:
                t = -pos[2] / d[2]
                p = pos + t * d
            else:
                p = pos + 10 * span * np.array([d[0], d[1], 0.0])
            pts.append(p[:2])
    p = np.array(pts)
    lo = np.maximum(p.min(0), [x0 - 1.0 * span, y0 - 1.0 * span])
    hi = np.minimum(p.max(0), [x1 + 1.0 * span, y1 + 1.0 * span])
    return lo, hi


# ==========================================================================
# numba helpers: interpolation, marching, texture, shading
# ==========================================================================
@njit(fastmath=True, cache=True)
def _tri(g, fx, fy, fz):
    nx, ny, nz = g.shape
    i = int(fx)
    j = int(fy)
    k = int(fz)
    if i > nx - 2:
        i = nx - 2
    if j > ny - 2:
        j = ny - 2
    if k > nz - 2:
        k = nz - 2
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    if k < 0:
        k = 0
    tx = fx - i
    ty = fy - j
    tz = fz - k
    c00 = g[i, j, k] * (1 - tx) + g[i + 1, j, k] * tx
    c10 = g[i, j + 1, k] * (1 - tx) + g[i + 1, j + 1, k] * tx
    c01 = g[i, j, k + 1] * (1 - tx) + g[i + 1, j, k + 1] * tx
    c11 = g[i, j + 1, k + 1] * (1 - tx) + g[i + 1, j + 1, k + 1] * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


@njit(fastmath=True, cache=True)
def _phi_at(sdf, sorg, ic, far, x, y, z):
    nx, ny, nz = sdf.shape
    fx = (x - sorg[0]) * ic
    fy = (y - sorg[1]) * ic
    fz = (z - sorg[2]) * ic
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > nx - 1 or fy > ny - 1 or fz > nz - 1:
        return far
    return _tri(sdf, fx, fy, fz)


@njit(fastmath=True, cache=True)
def _abs_at(ag, aorg, aic, x, y, z):
    """Trilinear sample of the (nx, ny, nz, 3) absorbance grid, 0 outside."""
    nx, ny, nz = ag.shape[0], ag.shape[1], ag.shape[2]
    fx = (x - aorg[0]) * aic
    fy = (y - aorg[1]) * aic
    fz = (z - aorg[2]) * aic
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx > nx - 1 or fy > ny - 1 or fz > nz - 1:
        return 0.0, 0.0, 0.0
    i = min(int(fx), nx - 2)
    j = min(int(fy), ny - 2)
    k = min(int(fz), nz - 2)
    tx = fx - i
    ty = fy - j
    tz = fz - k
    r = 0.0
    g = 0.0
    b = 0.0
    for a in range(2):
        wa = tx if a == 1 else 1.0 - tx
        for bb in range(2):
            wb = ty if bb == 1 else 1.0 - ty
            for cc in range(2):
                w = wa * wb * (tz if cc == 1 else 1.0 - tz)
                r += w * ag[i + a, j + bb, k + cc, 0]
                g += w * ag[i + a, j + bb, k + cc, 1]
                b += w * ag[i + a, j + bb, k + cc, 2]
    return r, g, b


@njit(fastmath=True, cache=True)
def _bil4(m, ox, oy, ic, x, y):
    """Bilinear sample of an (nx, ny, 4) map, zeros outside."""
    nx, ny = m.shape[0], m.shape[1]
    fx = (x - ox) * ic
    fy = (y - oy) * ic
    if fx < 0.0 or fy < 0.0 or fx > nx - 1 or fy > ny - 1:
        return 0.0, 0.0, 0.0, 0.0
    i = min(int(fx), nx - 2)
    j = min(int(fy), ny - 2)
    tx = fx - i
    ty = fy - j
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty
    a = w00 * m[i, j, 0] + w10 * m[i + 1, j, 0] + w01 * m[i, j + 1, 0] + w11 * m[i + 1, j + 1, 0]
    b = w00 * m[i, j, 1] + w10 * m[i + 1, j, 1] + w01 * m[i, j + 1, 1] + w11 * m[i + 1, j + 1, 1]
    c = w00 * m[i, j, 2] + w10 * m[i + 1, j, 2] + w01 * m[i, j + 1, 2] + w11 * m[i + 1, j + 1, 2]
    d = w00 * m[i, j, 3] + w10 * m[i + 1, j, 3] + w01 * m[i, j + 1, 3] + w11 * m[i + 1, j + 1, 3]
    return a, b, c, d


@njit(fastmath=True, cache=True)
def _ray_box(ox, oy, oz, dx, dy, dz, box):
    t0 = -1e30
    t1 = 1e30
    o = (ox, oy, oz)
    d = (dx, dy, dz)
    for a in range(3):
        lo = box[a]
        hi = box[a + 3]
        if abs(d[a]) < 1e-15:
            if o[a] < lo or o[a] > hi:
                return 1.0, 0.0
        else:
            inv = 1.0 / d[a]
            ta = (lo - o[a]) * inv
            tb = (hi - o[a]) * inv
            if ta > tb:
                ta, tb = tb, ta
            if ta > t0:
                t0 = ta
            if tb < t1:
                t1 = tb
    return t0, t1


@njit(fastmath=True, cache=True)
def _brick_exit(ox, oy, oz, dx, dy, dz, sorg, cell, B, bi, bj, bk):
    s = B * cell
    te = 1e30
    lo = (sorg[0] + bi * s, sorg[1] + bj * s, sorg[2] + bk * s)
    o = (ox, oy, oz)
    d = (dx, dy, dz)
    for a in range(3):
        if d[a] > 1e-15:
            t = (lo[a] + s - o[a]) / d[a]
        elif d[a] < -1e-15:
            t = (lo[a] - o[a]) / d[a]
        else:
            t = 1e30
        if t < te:
            te = t
    return te


@njit(fastmath=True, cache=True)
def _brick_of(bmin, B, fx, fy, fz):
    nbx, nby, nbz = bmin.shape
    bi = min(max(int(fx) // B, 0), nbx - 1)
    bj = min(max(int(fy) // B, 0), nby - 1)
    bk = min(max(int(fz) // B, 0), nbz - 1)
    return bi, bj, bk


@njit(fastmath=True, cache=True)
def _march_hit(sdf, bmin, B, sorg, cell, far, ox, oy, oz, dx, dy, dz, ta, tb):
    """First t in [ta, tb] where phi < 0 (sphere tracing + empty-brick skipping)."""
    ic = 1.0 / cell
    nx, ny, nz = sdf.shape
    eps = 1e-3 * cell
    t = ta
    tprev = ta
    for _ in range(40000):
        if t >= tb:
            return -1.0
        x = ox + t * dx
        y = oy + t * dy
        z = oz + t * dz
        fx = min(max((x - sorg[0]) * ic, 0.0), nx - 1.0)
        fy = min(max((y - sorg[1]) * ic, 0.0), ny - 1.0)
        fz = min(max((z - sorg[2]) * ic, 0.0), nz - 1.0)
        bi, bj, bk = _brick_of(bmin, B, fx, fy, fz)
        if bmin[bi, bj, bk] > 0.0:
            te = _brick_exit(ox, oy, oz, dx, dy, dz, sorg, cell, B, bi, bj, bk)
            if te < t:
                te = t
            tprev = te
            t = te + eps
            continue
        phi = _tri(sdf, fx, fy, fz)
        if phi < 0.0:
            a = tprev
            b = t
            for _k in range(10):
                m = 0.5 * (a + b)
                if _phi_at(sdf, sorg, ic, far, ox + m * dx, oy + m * dy, oz + m * dz) < 0.0:
                    b = m
                else:
                    a = m
            return b
        step = 0.8 * phi
        if step < 0.3 * cell:
            step = 0.3 * cell
        elif step > 1.2 * cell:
            step = 1.2 * cell
        tprev = t
        t += step
    return -1.0


@njit(fastmath=True, cache=True)
def _march_in(sdf, sorg, cell, far, ag, aorg, acell, ox, oy, oz, dx, dy, dz, tmax):
    """Integrate absorbance from a point inside the liquid until phi > 0 (code 0)
    or t = tmax (code 1). Returns (t, tau_r, tau_g, tau_b, code)."""
    ic = 1.0 / cell
    aic = 1.0 / acell
    ds = 0.5 * cell
    tr = 0.0
    tg = 0.0
    tb = 0.0
    t = 0.0
    phi = -ds
    for _ in range(200000):
        # deep inside, the (smooth) pigment field allows longer steps
        ds = -0.7 * phi
        if ds < 0.5 * cell:
            ds = 0.5 * cell
        elif ds > 1.5 * cell:
            ds = 1.5 * cell
        tn = t + ds
        if tn >= tmax:
            tn = tmax
            tm = 0.5 * (t + tn)
            ar, agg, ab = _abs_at(ag, aorg, aic, ox + tm * dx, oy + tm * dy, oz + tm * dz)
            L = tn - t
            return tn, tr + ar * L, tg + agg * L, tb + ab * L, 1
        phi = _phi_at(sdf, sorg, ic, far, ox + tn * dx, oy + tn * dy, oz + tn * dz)
        if phi > 0.0:
            a = t
            b = tn
            for _k in range(8):
                m = 0.5 * (a + b)
                if _phi_at(sdf, sorg, ic, far, ox + m * dx, oy + m * dy, oz + m * dz) > 0.0:
                    b = m
                else:
                    a = m
            tm = 0.5 * (t + a)
            ar, agg, ab = _abs_at(ag, aorg, aic, ox + tm * dx, oy + tm * dy, oz + tm * dz)
            L = a - t
            return a, tr + ar * L, tg + agg * L, tb + ab * L, 0
        tm = t + 0.5 * ds
        ar, agg, ab = _abs_at(ag, aorg, aic, ox + tm * dx, oy + tm * dy, oz + tm * dz)
        tr += ar * ds
        tg += agg * ds
        tb += ab * ds
        t = tn
    return t, tr, tg, tb, 1


@njit(fastmath=True, cache=True)
def _march_tau(sdf, bmin, B, sorg, cell, far, ag, aorg, acell, box, ox, oy, oz, dx, dy, dz):
    """Pigment optical depth along a ray (light/shadow rays), skipping empty bricks."""
    ta, tb = _ray_box(ox, oy, oz, dx, dy, dz, box)
    if ta < 0.0:
        ta = 0.0
    if ta >= tb:
        return 0.0, 0.0, 0.0
    ic = 1.0 / cell
    aic = 1.0 / acell
    nx, ny, nz = sdf.shape
    ds = 0.6 * cell
    tr = 0.0
    tg = 0.0
    tbb = 0.0
    t = ta
    phi = 0.0
    for _ in range(40000):
        if t >= tb:
            break
        # step from the last |phi|: long strides deep inside or well outside the liquid
        ds = 0.7 * abs(phi)
        if ds < 0.6 * cell:
            ds = 0.6 * cell
        elif ds > 1.5 * cell:
            ds = 1.5 * cell
        tm = t + 0.5 * ds
        x = ox + tm * dx
        y = oy + tm * dy
        z = oz + tm * dz
        fx = min(max((x - sorg[0]) * ic, 0.0), nx - 1.0)
        fy = min(max((y - sorg[1]) * ic, 0.0), ny - 1.0)
        fz = min(max((z - sorg[2]) * ic, 0.0), nz - 1.0)
        bi, bj, bk = _brick_of(bmin, B, fx, fy, fz)
        if bmin[bi, bj, bk] > 0.0:
            te = _brick_exit(ox, oy, oz, dx, dy, dz, sorg, cell, B, bi, bj, bk)
            t = max(te, t + 1e-3 * cell)
            phi = 0.0
            continue
        phi = _tri(sdf, fx, fy, fz)
        if phi < 0.5 * ds:
            w = min(0.5 - phi / ds, 1.0) * ds  # soft inside fraction (anti-aliased shadow edge)
            ar, agg, ab = _abs_at(ag, aorg, aic, x, y, z)
            tr += ar * w
            tg += agg * w
            tbb += ab * w
        t += ds
    return tr, tg, tbb


@njit(fastmath=True, cache=True)
def _normal(sdf, sorg, ic, cell, far, x, y, z):
    e = cell
    gx = _phi_at(sdf, sorg, ic, far, x + e, y, z) - _phi_at(sdf, sorg, ic, far, x - e, y, z)
    gy = _phi_at(sdf, sorg, ic, far, x, y + e, z) - _phi_at(sdf, sorg, ic, far, x, y - e, z)
    gz = _phi_at(sdf, sorg, ic, far, x, y, z + e) - _phi_at(sdf, sorg, ic, far, x, y, z - e)
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n < 1e-20:
        return 0.0, 0.0, 1.0
    return gx / n, gy / n, gz / n


# ---- procedural paper ----------------------------------------------------
@njit(fastmath=True, cache=True)
def _hash(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 1440662683) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h * (1.0 / 4294967296.0)


@njit(fastmath=True, cache=True)
def _vnoise(x, y, seed):
    xf = math.floor(x)
    yf = math.floor(y)
    ix = np.int64(xf)
    iy = np.int64(yf)
    fx = x - xf
    fy = y - yf
    u = fx * fx * (3.0 - 2.0 * fx)
    v = fy * fy * (3.0 - 2.0 * fy)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


@njit(fastmath=True, cache=True)
def _paper_h(X, Y, f):
    """Paper height in texture units (X, Y in units of d0*tex_scale; f = pixel
    footprint in the same units). Octaves below ~2 px fade out (no shimmer)."""
    h = 0.0
    lam = 0.9
    amp = 1.0
    for o in range(4):  # tooth: isotropic value-noise fbm
        w = lam / (2.0 * f) - 1.0
        if w > 0.0:
            if w > 1.0:
                w = 1.0
            h += amp * w * (_vnoise(X / lam + 17.3 * o, Y / lam - 9.1 * o, o) - 0.5)
        lam *= 0.5
        amp *= 0.55
    for m in range(3):  # fibres: thin elongated ridges in three directions
        ang = 0.6 + 1.9 * m
        ca = math.cos(ang)
        sa = math.sin(ang)
        U = ca * X + sa * Y
        V = -sa * X + ca * Y
        width = 0.07
        w = width / (0.7 * f) - 1.0
        if w > 0.0:
            if w > 1.0:
                w = 1.0
            n = _vnoise(U / 1.7 + 31.0 * m, V / width, 11 + m)
            rr = 1.0 - abs(2.0 * n - 1.0)
            rr = rr * rr
            rr = rr * rr * rr
            h += 0.30 * w * rr
    return h


@njit(fastmath=True, cache=True)
def _env(dx, dy, dz, P):
    """Soft studio environment: cool sky dome, warm-white floor (the paper), and a
    rectangular window softbox in the key-light direction."""
    if dz >= 0.0:
        s = 0.60 + 0.40 * math.sqrt(dz)
        r = 0.90 * s
        g = 0.94 * s
        b = 1.00 * s
    else:
        r = 0.80
        g = 0.79
        b = 0.77
    Lx, Ly, Lz = P[_LX], P[_LY], P[_LZ]
    c = dx * Lx + dy * Ly + dz * Lz
    if c > 0.5:
        a = (dx * P[_T1X] + dy * P[_T1Y] + dz * P[_T1Z]) / c
        bb = (dx * P[_T2X] + dy * P[_T2Y] + dz * P[_T2Z]) / c
        ea = min(max((0.17 - abs(a)) / 0.04, 0.0), 1.0)
        eb = min(max((0.12 - abs(bb)) / 0.04, 0.0), 1.0)
        w = ea * eb
        if w > 0.0:
            # faint mullion across the window
            mull = 1.0 - 0.6 * min(max((0.012 - abs(a)) / 0.006, 0.0), 1.0)
            L = P[_BOXL] * w * mull
            r += L
            g += 0.97 * L
            b += 0.92 * L
    return r, g, b


@njit(fastmath=True, cache=True)
def _shade_paper(x, y, fp, wet_in, P, sdf, sorg, scell, bmin, B, box, ag, aorg, acell, colm,
                 pmap, porg, pcell):
    S = P[_D0] * P[_TEXS]
    X = x / S
    Y = y / S
    f = fp / S
    h = _paper_h(X, Y, f)
    dl = max(0.35 * f, 0.02)
    hx = (_paper_h(X + dl, Y, f) - h) / dl
    hy = (_paper_h(X, Y + dl, f) - h) / dl
    bump = P[_BUMP]
    nx_ = -bump * hx
    ny_ = -bump * hy
    nn = 1.0 / math.sqrt(nx_ * nx_ + ny_ * ny_ + 1.0)
    nx_ *= nn
    ny_ *= nn
    nz_ = nn
    alb = P[_ALB] * (0.975 + 0.06 * h)
    # pigment on the paper (one-pass optical depth) and dampness from absorbed liquid
    pr, pg, pb, wet = _bil4(pmap, porg[0], porg[1], 1.0 / pcell, x, y)
    gran = 1.0 + P[_GRAN] * (-2.0 * h)
    if gran < 0.25:
        gran = 0.25
    Tr = math.exp(-2.0 * pr * gran)
    Tg = math.exp(-2.0 * pg * gran)
    Tb = math.exp(-2.0 * pb * gran)
    lr = 0.0
    lg = 0.0
    lb = 0.0
    ao = 1.0
    if P[_HASLIQ] > 0.0:
        cr, cg, cb, th = _bil4(colm, sorg[0], sorg[1], 1.0 / scell, x, y)
        if th > 0.05 * scell:
            wet_in = 1.0
        # ambient through the liquid column above
        ar_ = math.exp(-cr)
        ag_ = math.exp(-cg)
        ab_ = math.exp(-cb)
        # coloured shadow: light path through the liquid toward the key light
        lr, lg, lb = _march_tau(sdf, bmin, B, sorg, scell, P[_FAR], ag, aorg, acell, box,
                                x, y, 1e-4 * P[_D0], P[_LX], P[_LY], P[_LZ])
        if wet_in < 0.5:
            # contact occlusion from nearby liquid above the dry paper
            occ = 0.0
            ic = 1.0 / scell
            hh = 0.4 * P[_D0]
            wt = 0.5
            for _k in range(4):
                ph = _phi_at(sdf, sorg, ic, P[_FAR], x, y, hh)
                if ph < hh:
                    occ += wt * (hh - max(ph, 0.0)) / hh
                hh *= 2.0
                wt *= 0.5
            ao = 1.0 - P[_AOK] * min(occ, 1.0)
    else:
        ar_ = 1.0
        ag_ = 1.0
        ab_ = 1.0
    if wet > wet_in:
        wet_in = wet
    alb *= 1.0 - P[_WETD] * min(wet_in, 1.0)
    ndl = nx_ * P[_LX] + ny_ * P[_LY] + nz_ * P[_LZ]
    if ndl < 0.0:
        ndl = 0.0
    k = P[_IK] * ndl
    a = P[_IA] * ao * (0.6 + 0.4 * nz_)
    r = alb * Tr * (k * 1.00 * math.exp(-lr) + a * 0.92 * ar_)
    g = alb * Tg * (k * 0.97 * math.exp(-lg) + a * 0.96 * ag_)
    b = alb * Tb * (k * 0.92 * math.exp(-lb) + a * 1.00 * ab_)
    return r, g, b


@njit(fastmath=True, cache=True)
def _surface_light(nx_, ny_, nz_, dx, dy, dz, P):
    """Reflected radiance at a liquid surface (env reflection + Blinn-Phong)."""
    c = -(dx * nx_ + dy * ny_ + dz * nz_)
    rx = dx + 2 * c * nx_
    ry = dy + 2 * c * ny_
    rz = dz + 2 * c * nz_
    er, eg, eb = _env(rx, ry, rz, P)
    hx = P[_LX] - dx
    hy = P[_LY] - dy
    hz = P[_LZ] - dz
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    ndh = (nx_ * hx + ny_ * hy + nz_ * hz) / hn
    ndl = nx_ * P[_LX] + ny_ * P[_LY] + nz_ * P[_LZ]
    sp = 0.0
    if ndh > 0.0 and ndl > 0.0:
        s = P[_SHINE]
        sp = P[_KS] * P[_IK] * (s + 2.0) / (2.0 * math.pi) * ndh ** s * ndl
    return er + sp, eg + 0.98 * sp, eb + 0.95 * sp


@njit(fastmath=True, cache=True)
def _schlick(cosv, F0):
    m = 1.0 - cosv
    if m < 0.0:
        m = 0.0
    m2 = m * m
    return F0 + (1.0 - F0) * m2 * m2 * m


@njit(fastmath=True, cache=True)
def _trace(ox, oy, oz, dx, dy, dz, P, sdf, sorg, scell, bmin, B, box, ag, aorg, acell, colm,
           pmap, porg, pcell):
    far = P[_FAR]
    ic = 1.0 / scell
    eta = P[_ETA]
    F0 = P[_F0]
    has = P[_HASLIQ] > 0.0
    thr_r = 1.0
    thr_g = 1.0
    thr_b = 1.0
    Lr = 0.0
    Lg = 0.0
    Lb = 0.0
    inside = False
    off = 0.06 * scell
    for _ev in range(8):
        if dz < -1e-12 and oz > 0.0:
            tp = -oz / dz
        elif dz < -1e-12:
            tp = 0.0
        else:
            tp = 1e30
        if has:
            ta, tb = _ray_box(ox, oy, oz, dx, dy, dz, box)
        else:
            ta, tb = 1.0, 0.0
        if not inside:
            hit = -1.0
            if ta < tb and tb > 0.0:
                if ta < 0.0:
                    ta = 0.0
                tb2 = min(tb, tp)
                if ta < tb2:
                    hit = _march_hit(sdf, bmin, B, sorg, scell, far, ox, oy, oz, dx, dy, dz, ta, tb2)
            if hit < 0.0:
                if tp < 1e29:
                    px = ox + tp * dx
                    py = oy + tp * dy
                    ddx = px - P[_CAMX]
                    ddy = py - P[_CAMY]
                    ddz = -P[_CAMZ]
                    dist = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                    fp = dist * P[_PIXANG] / max(abs(dz), 0.25)
                    r, g, b = _shade_paper(px, py, fp, 0.0, P, sdf, sorg, scell, bmin, B, box, ag,
                                           aorg, acell, colm, pmap, porg, pcell)
                else:
                    r, g, b = _env(dx, dy, dz, P)
                return Lr + thr_r * r, Lg + thr_g * g, Lb + thr_b * b
            px = ox + hit * dx
            py = oy + hit * dy
            pz = oz + hit * dz
            nx_, ny_, nz_ = _normal(sdf, sorg, ic, scell, far, px, py, pz)
            cosi = -(dx * nx_ + dy * ny_ + dz * nz_)
            if cosi < 0.0:
                nx_ = -nx_
                ny_ = -ny_
                nz_ = -nz_
                cosi = -cosi
            F = _schlick(cosi, F0)
            sr, sg, sb = _surface_light(nx_, ny_, nz_, dx, dy, dz, P)
            Lr += thr_r * F * sr
            Lg += thr_g * F * sg
            Lb += thr_b * F * sb
            ri = 1.0 / eta
            k = 1.0 - ri * ri * (1.0 - cosi * cosi)
            sk = math.sqrt(max(k, 0.0))
            dx = ri * dx + (ri * cosi - sk) * nx_
            dy = ri * dy + (ri * cosi - sk) * ny_
            dz = ri * dz + (ri * cosi - sk) * nz_
            thr_r *= 1.0 - F
            thr_g *= 1.0 - F
            thr_b *= 1.0 - F
            ox = px - off * nx_
            oy = py - off * ny_
            oz = pz - off * nz_
            inside = True
        else:
            tmax = tp
            if ta < tb and tb < tmax:
                tmax = tb
            if tmax > 1e29:
                tmax = 1e29
            te, tr, tg, tbb, code = _march_in(sdf, sorg, scell, far, ag, aorg, acell, ox, oy, oz,
                                              dx, dy, dz, tmax)
            thr_r *= math.exp(-tr)
            thr_g *= math.exp(-tg)
            thr_b *= math.exp(-tbb)
            px = ox + te * dx
            py = oy + te * dy
            pz = oz + te * dz
            if code == 1:
                if te >= tp * (1.0 - 1e-9):
                    ddx = px - P[_CAMX]
                    ddy = py - P[_CAMY]
                    ddz = -P[_CAMZ]
                    dist = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                    fp = dist * P[_PIXANG] / max(abs(dz), 0.25)
                    r, g, b = _shade_paper(px, py, fp, 1.0, P, sdf, sorg, scell, bmin, B, box, ag,
                                           aorg, acell, colm, pmap, porg, pcell)
                    return Lr + thr_r * r, Lg + thr_g * g, Lb + thr_b * b
                ox = px
                oy = py
                oz = pz
                inside = False
                continue
            nx_, ny_, nz_ = _normal(sdf, sorg, ic, scell, far, px, py, pz)
            cosi = dx * nx_ + dy * ny_ + dz * nz_
            if cosi < 0.0:
                nx_ = -nx_
                ny_ = -ny_
                nz_ = -nz_
                cosi = -cosi
            k = 1.0 - eta * eta * (1.0 - cosi * cosi)
            if k < 0.0:  # total internal reflection
                dx = dx - 2.0 * cosi * nx_
                dy = dy - 2.0 * cosi * ny_
                dz = dz - 2.0 * cosi * nz_
                ox = px - off * nx_
                oy = py - off * ny_
                oz = pz - off * nz_
                continue
            sk = math.sqrt(k)
            F = _schlick(sk, F0)
            dx = eta * dx - (eta * cosi - sk) * nx_
            dy = eta * dy - (eta * cosi - sk) * ny_
            dz = eta * dz - (eta * cosi - sk) * nz_
            dn = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= dn
            dy *= dn
            dz *= dn
            thr_r *= 1.0 - F
            thr_g *= 1.0 - F
            thr_b *= 1.0 - F
            ox = px + off * nx_
            oy = py + off * ny_
            oz = pz + off * nz_
            inside = False
    r, g, b = _env(dx, dy, dz, P)
    return Lr + thr_r * r, Lg + thr_g * g, Lb + thr_b * b


@njit(parallel=True, fastmath=True, cache=True)
def _render_kernel(out, rows, spp, cam, P, sdf, sorg, scell, bmin, B, box, ag, aorg, acell, colm,
                   pmap, porg, pcell):
    H, W = out.shape[0], out.shape[1]
    inv = 1.0 / (spp * spp)
    for ti in prange(rows.shape[0]):
        py = rows[ti]
        for px in range(W):
            ar = 0.0
            agg = 0.0
            ab = 0.0
            for sy in range(spp):
                for sx in range(spp):
                    u = (px + (sx + 0.5) / spp) / W
                    v = (py + (sy + 0.5) / spp) / H
                    a = (2.0 * u - 1.0) * cam[12]
                    b = (1.0 - 2.0 * v) * cam[13]
                    dx = cam[3] + a * cam[6] + b * cam[9]
                    dy = cam[4] + a * cam[7] + b * cam[10]
                    dz = cam[5] + a * cam[8] + b * cam[11]
                    dn = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
                    r, g, bb = _trace(cam[0], cam[1], cam[2], dx * dn, dy * dn, dz * dn, P, sdf, sorg,
                                      scell, bmin, B, box, ag, aorg, acell, colm, pmap, porg, pcell)
                    ar += r
                    agg += g
                    ab += bb
            out[py, px, 0] = ar * inv
            out[py, px, 1] = agg * inv
            out[py, px, 2] = ab * inv


# ---- grid preprocessing ---------------------------------------------------
@njit(parallel=True, cache=True)
def _brick_min(sdf, B):
    nx, ny, nz = sdf.shape
    bx = max((nx - 2) // B + 1, 1)
    by = max((ny - 2) // B + 1, 1)
    bz = max((nz - 2) // B + 1, 1)
    out = np.empty((bx, by, bz), np.float32)
    for bi in prange(bx):
        for bj in range(by):
            for bk in range(bz):
                m = np.float32(1e30)
                for i in range(bi * B, min(bi * B + B, nx - 1) + 1):
                    for j in range(bj * B, min(bj * B + B, ny - 1) + 1):
                        for k in range(bk * B, min(bk * B + B, nz - 1) + 1):
                            v = sdf[i, j, k]
                            if v < m:
                                m = v
                out[bi, bj, bk] = m
    return out


@njit(parallel=True, cache=True)
def _smooth_axis(src, dst, axis):
    nx, ny, nz = src.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if axis == 0:
                    a = src[max(i - 1, 0), j, k]
                    b = src[min(i + 1, nx - 1), j, k]
                elif axis == 1:
                    a = src[i, max(j - 1, 0), k]
                    b = src[i, min(j + 1, ny - 1), k]
                else:
                    a = src[i, j, max(k - 1, 0)]
                    b = src[i, j, min(k + 1, nz - 1)]
                dst[i, j, k] = 0.25 * a + 0.5 * src[i, j, k] + 0.25 * b


def _smooth_sdf(sdf, passes):
    if passes <= 0:
        return sdf
    tmp = np.empty_like(sdf)
    for _ in range(int(passes)):
        _smooth_axis(sdf, tmp, 0)
        _smooth_axis(tmp, sdf, 1)
        _smooth_axis(sdf, tmp, 2)
        sdf, tmp = tmp, sdf
    return sdf


@njit(parallel=True, cache=True)
def _extrude_contact(sdf, kc):
    """Liquid touching the paper rests on it: below the contact height kc, phi
    is the running minimum from above, so the reconstructed surface meets z = 0
    instead of leaving the half-spacing gap of the bottom particle layer."""
    nx, ny, nz = sdf.shape
    kc = min(kc, nz - 1)
    for i in prange(nx):
        for j in range(ny):
            for k in range(kc - 1, -1, -1):
                if sdf[i, j, k + 1] < sdf[i, j, k]:
                    sdf[i, j, k] = sdf[i, j, k + 1]


@njit(parallel=True, fastmath=True, cache=True)
def _column_map(sdf, sorg, scell, ag, aorg, acell, out):
    """Per (x, y) column above the paper: vertical pigment optical depth (RGB) and
    liquid thickness. Used for sky light reaching wet paper and the wet flag."""
    nx, ny, nz = sdf.shape
    aic = 1.0 / acell
    for i in prange(nx):
        x = sorg[0] + i * scell
        for j in range(ny):
            y = sorg[1] + j * scell
            tr = 0.0
            tg = 0.0
            tb = 0.0
            th = 0.0
            for k in range(nz):
                z = sorg[2] + k * scell
                if z < 0.0:
                    continue
                phi = sdf[i, j, k]
                if phi < 0.5 * scell:
                    w = min(0.5 - phi / scell, 1.0) * scell
                    ar, agg, ab = _abs_at(ag, aorg, aic, x, y, z)
                    tr += ar * w
                    tg += agg * w
                    tb += ab * w
                    th += w
            out[i, j, 0] = tr
            out[i, j, 1] = tg
            out[i, j, 2] = tb
            out[i, j, 3] = th


# ==========================================================================
# scene preparation
# ==========================================================================
def _opts(kw):
    o = dict(DEFAULTS)
    for k, v in kw.items():
        if k not in o:
            raise TypeError("unknown render option %r" % k)
        o[k] = v
    return o


def _absorbance(fr, K):
    rgb = np.asarray(fr.get("pigment_rgb", np.tile([[0.8, 0.1, 0.1]], (K, 1))), np.float64).reshape(-1, 3)
    if rgb.shape[0] < K:
        rgb = np.vstack([rgb, np.tile(rgb[-1:], (K - rgb.shape[0], 1))])
    return -np.log(np.clip(rgb[:K], 1e-3, 1.0))  # (K, 3)


def _liquid_domain(live, d0, o, clip_lo, clip_hi):
    """Node grid covering the live liquid (+kernel margin), clipped to the view."""
    R = o["R"] * d0
    cell = o["sdf_cell"] * d0
    pad = R + 2 * cell
    lo = live.min(0) - pad
    hi = live.max(0) + pad
    lo[:2] = np.maximum(lo[:2], clip_lo)
    hi[:2] = np.minimum(hi[:2], clip_hi)
    lo[2] = -cell
    hi[2] = max(hi[2], 2 * d0)
    if np.any(hi[:2] <= lo[:2]):
        return None
    while True:
        shape = np.maximum(np.ceil((hi - lo) / cell).astype(np.int64) + 1, 2)
        if np.prod(shape) <= o["max_cells"]:
            break
        cell *= 1.15
    return lo, cell, tuple(int(s) for s in shape)


def _prepare(fr, cam, extent, o, timings):
    d0 = _scalar(fr, "d0", 0.005)
    x, c, ab = _live(fr)
    K = c.shape[1]
    A = _absorbance(fr, K)
    sigma = o["absorb_scale"] / d0
    sigma_p = (o["paper_absorb_scale"] if o["paper_absorb_scale"] is not None else o["absorb_scale"]) / d0
    live = x[~ab]
    live_c = c[~ab]
    # keep liquid above the paper plane only (anything below is in the paper), and
    # drop strays far above/away from the bulk (they would blow up the SDF grid)
    keep = (live[:, 2] > -0.5 * d0) & _inliers(live, d0)
    live = live[keep]
    live_c = live_c[keep]

    glo, ghi = _ground_footprint(cam, extent)
    margin = max(4 * d0, 0.15 * float(np.max(ghi - glo)))
    t0 = time.perf_counter()
    dom = _liquid_domain(live, d0, o, glo - margin, ghi + margin) if live.shape[0] else None
    if dom is not None:
        sorg, scell, sshape = dom
        sdf = zhu_bridson_sdf(live, o["radius"] * d0, sorg, scell, sshape, o["R"] * d0)
        sdf = _smooth_sdf(sdf, o["sdf_smooth"])
        timings["sdf"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        kc = int(round((d0 - sorg[2]) / scell))
        _extrude_contact(sdf, kc)
        B = 8
        bmin = _brick_min(sdf, B)
        box = np.array([sorg[0], sorg[1], sorg[2],
                        sorg[0] + (sshape[0] - 1) * scell, sorg[1] + (sshape[1] - 1) * scell,
                        sorg[2] + (sshape[2] - 1) * scell])
        # pigment grid on a coarser lattice over the same box
        pcell = max(o["pigment_cell"] * d0, 1.5 * scell)
        while True:
            pshape = tuple(int(s) for s in np.maximum(np.ceil((box[3:] - box[:3]) / pcell).astype(int) + 1, 2))
            if np.prod(pshape) * 3 <= o["max_cells"]:
                break
            pcell *= 1.15
        pg = pigment_grid(live, live_c, sorg, pcell, pshape, o["pigment_r"] * d0, d0=d0)
        ag = np.empty(pshape + (3,), np.float32)
        for ch in range(3):
            acc = np.zeros(pshape, np.float32)
            for k in range(K):
                if A[k, ch] != 0.0:
                    acc += np.float32(sigma * A[k, ch]) * pg[k]
            ag[..., ch] = acc
        del pg
        aorg = sorg.copy()
        colm = np.zeros((sshape[0], sshape[1], 4), np.float32)
        _column_map(sdf, sorg, scell, ag, aorg, pcell, colm)
        timings["pigment"] = time.perf_counter() - t0
        has = 1.0
    else:
        sorg = np.zeros(3)
        scell = d0
        sdf = np.full((2, 2, 2), 1.0, np.float32)
        B = 8
        bmin = np.full((1, 1, 1), 1.0, np.float32)
        box = np.array([0, 0, 0, d0, d0, d0], np.float64)
        ag = np.zeros((2, 2, 2, 3), np.float32)
        aorg = np.zeros(3)
        pcell = d0
        colm = np.zeros((2, 2, 4), np.float32)
        has = 0.0
        timings["sdf"] = 0.0
        timings["pigment"] = 0.0

    # paper map over the visible part of the sheet
    t0 = time.perf_counter()
    mcell = o["paper_cell"] * d0
    mlo = glo - 2 * mcell
    mhi = ghi + 2 * mcell
    while True:
        mshape = np.maximum(np.ceil((mhi - mlo) / mcell).astype(int) + 1, 2)
        if mshape.max() <= 2400:
            break
        mcell *= 1.2
    px = np.asarray(fr.get("paper_x", np.zeros((0, 3))), np.float64).reshape(-1, 3)
    P_ = px.shape[0]
    pc = np.asarray(fr.get("paper_c", np.zeros((P_, K))), np.float64)
    pc = pc.reshape(P_, -1) if P_ else np.zeros((0, K))
    # contact staining (Sec. 5.2): the solver's persistent per-paper-particle record
    # (paper_stain = max concentration that touched it) when the frame has one,
    # else computed live from the liquid touching the paper this frame
    stain = fr.get("paper_stain") if o["stain_source"] in ("auto", "frame") else None
    if stain is not None and P_ and np.asarray(stain).size:
        st = np.asarray(stain, np.float64).reshape(P_, -1)
        kk = max(pc.shape[1], st.shape[1])
        tot = np.zeros((P_, kk))
        tot[:, :pc.shape[1]] += pc
        tot[:, :st.shape[1]] += o["contact_stain"] * d0 * st
        pmap_k = paper_pigment_map(px, tot, mlo, mcell, mshape, o["paper_r"] * d0, d0=d0)
    elif o["stain_source"] in ("auto", "live"):
        pmap_k = paper_pigment_map(px, pc, mlo, mcell, mshape, o["paper_r"] * d0, d0=d0,
                                   liquid_x=live, liquid_c=live_c, contact_stain=o["contact_stain"])
    else:
        pmap_k = paper_pigment_map(px, pc, mlo, mcell, mshape, o["paper_r"] * d0, d0=d0)
    Kp = pmap_k.shape[0]
    Ap = _absorbance(fr, Kp)
    pmap = np.zeros((int(mshape[0]), int(mshape[1]), 4), np.float32)
    for ch in range(3):
        pmap[..., ch] = np.tensordot(Ap[:, ch] * sigma_p, pmap_k, axes=(0, 0))
    # dampness: absorbed liquid in the sheet, plus paper the liquid has humidified
    absorbed = x[ab]
    if absorbed.shape[0]:
        th = splat_density_2d(absorbed[:, :2], np.full(absorbed.shape[0], d0 ** 3), mlo, mcell, mshape,
                              o["paper_r"] * d0)[0]
        pmap[..., 3] = 1.0 - np.exp(-th / (0.3 * d0))
    touch = fr.get("paper_touch") if o["touch_damp"] > 0 else None
    if touch is not None and P_ and np.asarray(touch).size == P_:
        tw = np.asarray(touch, np.float64).reshape(P_) > 0
        if tw.any():
            area = d0 * d0
            dens = splat_density_2d(px[tw, :2], np.full(int(tw.sum()), area), mlo, mcell, mshape,
                                    o["paper_r"] * d0)[0]
            pmap[..., 3] = np.maximum(pmap[..., 3], o["touch_damp"] * np.clip(dens, 0.0, 1.0))
    timings["paper"] = time.perf_counter() - t0
    return dict(d0=d0, sdf=sdf, sorg=np.asarray(sorg, np.float64), scell=float(scell), bmin=bmin, B=B,
                box=box, ag=ag, aorg=np.asarray(aorg, np.float64), acell=float(pcell), colm=colm,
                pmap=pmap, porg=np.asarray(mlo, np.float64), pcell=float(mcell), has=has,
                far=float(o["R"] * d0))


def _params(o, sc, cam, height):
    pos, fwd, right, up, tx, ty = cam
    L = np.asarray(o["light_dir"], np.float64)
    L /= np.linalg.norm(L)
    t1 = np.cross(L, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(t1, L)
    P = np.zeros(_NPAR)
    P[_ETA] = o["eta"]
    P[_F0] = ((o["eta"] - 1) / (o["eta"] + 1)) ** 2
    P[_IK] = o["key"] * o["exposure"]
    P[_IA] = o["ambient"] * o["exposure"]
    P[_LX:_LZ + 1] = L
    P[_SHINE] = o["shininess"]
    P[_KS] = o["spec"]
    P[_BOXL] = o["softbox"] * o["exposure"]
    P[_ALB] = o["paper_albedo"]
    P[_BUMP] = o["bump"]
    P[_TEXS] = o["tex_scale"]
    P[_GRAN] = o["granulation"]
    P[_D0] = sc["d0"]
    P[_WETD] = o["wet_darken"]
    P[_AOK] = o["ao"]
    P[_PIXANG] = 2.0 * ty / height
    P[_HASLIQ] = sc["has"]
    P[_FAR] = sc["far"]
    P[_CAMX:_CAMZ + 1] = pos
    P[_T1X:_T1Z + 1] = t1
    P[_T2X:_T2Z + 1] = t2
    return P


def _to_srgb8(lin, dither=True, seed=0):
    x = np.maximum(lin, 0.0)
    s = 0.82
    hi = x > s
    x[hi] = s + (1 - s) * (1 - np.exp(-(x[hi] - s) / (1 - s)))
    x = np.clip(x, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    v = srgb * 255.0
    if dither:
        rng = np.random.default_rng(seed)
        v += rng.uniform(-0.5, 0.5, size=v.shape[:2])[..., None]
    return np.clip(v + 0.5, 0, 255).astype(np.uint8)


# ==========================================================================
# public API
# ==========================================================================
def render_frame(frame, camera="top", width=960, height=540, extent=None, info=None, **opts):
    """Render one frame (path to frame_*.npz or a dict with the SPEC.md keys).

    camera : 'top' (looking straight down) or 'oblique' (elevated 3/4 view from -y).
    extent : (x0, x1, y0, y1[, zmax]) region of the paper to frame, metres.
        Default ``auto_extent(frame)``. 'top' grows it to the image aspect ratio;
        'oblique' also fits liquid up to height zmax (default ``content_height``).
    info : optional dict, filled with timings (s) and grid sizes.
    opts : any key of ``DEFAULTS`` (spp, absorb_scale, eta, light_dir, ...).
    Returns an (height, width, 3) uint8 sRGB image.
    """
    o = _opts(opts)
    t_all = time.perf_counter()
    fr = load_frame(frame)
    if extent is None:
        extent = auto_extent(fr)
    extent = tuple(float(v) for v in extent)
    if camera == "oblique" and len(extent) == 4:
        extent = extent + (content_height(fr),)
    cam = _camera(camera, extent, width, height, o)
    timings = {}
    sc = _prepare(fr, cam, extent, o, timings)
    P = _params(o, sc, cam, height)
    pos, fwd, right, up, tx, ty = cam
    camv = np.concatenate([pos, fwd, right, up, [tx, ty]]).astype(np.float64)
    out = np.zeros((height, width, 3), np.float32)
    nthreads = 8
    rows = np.concatenate([np.arange(s, height, nthreads) for s in range(nthreads)]).astype(np.int64)
    t0 = time.perf_counter()
    _render_kernel(out, rows, int(o["spp"]), camv, P, sc["sdf"], sc["sorg"], sc["scell"], sc["bmin"],
                   sc["B"], sc["box"], sc["ag"], sc["aorg"], sc["acell"], sc["colm"], sc["pmap"],
                   sc["porg"], sc["pcell"])
    timings["trace"] = time.perf_counter() - t0
    img = _to_srgb8(out, dither=o["dither"], seed=int(_scalar(fr, "time", 0.0) * 1e4) % (2 ** 31))
    timings["total"] = time.perf_counter() - t_all
    if info is not None:
        info.update(timings)
        info["sdf_shape"] = tuple(sc["sdf"].shape)
        info["sdf_cell"] = sc["scell"]
        info["paper_shape"] = tuple(sc["pmap"].shape[:2])
        info["extent"] = tuple(extent)
    return img


def frame_mesh(frame, radius=None, R=None, cell=None, smooth=None, contact=True):
    """Liquid surface of a frame as a closed triangle mesh (absorbed particles
    excluded), the same surface the ray-marcher traces: Zhu & Bridson SDF,
    ``smooth`` [1,2,1] passes (default DEFAULTS['sdf_smooth']; 0 = raw Eq. 11) and,
    with ``contact``, liquid touching the paper extended down to z = 0.
    Returns (verts, faces, normals) in world metres, normals outward."""
    fr = load_frame(frame)
    d0 = _scalar(fr, "d0", 0.005)
    x, c, ab = _live(fr)
    live = x[~ab]
    live = live[(live[:, 2] > -0.5 * d0) & _inliers(live, d0)]
    radius = DEFAULTS["radius"] * d0 if radius is None else radius
    R = DEFAULTS["R"] * d0 if R is None else R
    cell = DEFAULTS["sdf_cell"] * d0 if cell is None else cell
    smooth = DEFAULTS["sdf_smooth"] if smooth is None else smooth
    if live.shape[0] == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))
    pad = R + 2 * cell
    lo = live.min(0) - pad
    hi = live.max(0) + pad
    if contact:
        lo[2] = max(lo[2], -2 * cell)
    shape = tuple(int(s) for s in np.ceil((hi - lo) / cell).astype(int) + 1)
    while np.prod(shape) > DEFAULTS["max_cells"]:
        cell *= 1.15
        shape = tuple(int(s) for s in np.ceil((hi - lo) / cell).astype(int) + 1)
    sdf = zhu_bridson_sdf(live, radius, lo, cell, shape, R)
    sdf = _smooth_sdf(sdf, smooth)
    if contact:
        _extrude_contact(sdf, int(round((d0 - lo[2]) / cell)))
        kz = int(np.floor((0.0 - lo[2]) / cell))  # close the mesh just below the paper
        sdf[:, :, :max(kz, 1)] = R
    return extract_mesh(sdf, lo, cell)


def _frame_files(frame_dir):
    return sorted(glob.glob(os.path.join(frame_dir, "frame_*.npz")))


def render_sequence(frame_dir, out_mp4, fps=30, camera="top", width=960, height=540, extent="auto",
                    crf=18, preset="medium", obj_dir=None, progress=True, **kw):
    """Render every frame_*.npz in ``frame_dir`` and stream them to ffmpeg
    (libx264, yuv420p, crf 18). One frame is in memory at a time. The camera
    extent is fixed for the whole sequence (``sequence_extent`` by default)."""
    files = _frame_files(frame_dir)
    if not files:
        raise FileNotFoundError("no frame_*.npz in %s" % frame_dir)
    width -= width % 2
    height -= height % 2
    if extent == "auto" or extent is None:
        extent = sequence_extent(files)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out_mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.perf_counter()
    try:
        for n, f in enumerate(files):
            info = {}
            img = render_frame(f, camera=camera, width=width, height=height, extent=extent, info=info, **kw)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            if obj_dir:
                os.makedirs(obj_dir, exist_ok=True)
                v, fc, nr = frame_mesh(f)
                write_obj(os.path.join(obj_dir, os.path.basename(f)[:-4] + ".obj"), v, fc, nr)
            if progress:
                el = time.perf_counter() - t_start
                print("[render] %d/%d %s %.1fs (avg %.1fs/frame)" % (n + 1, len(files), os.path.basename(f),
                      info.get("total", 0.0), el / (n + 1)), flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg exited with %d" % rc)
    return out_mp4


def _save_png(img, path):
    from PIL import Image

    Image.fromarray(img).save(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m paintsim.render",
                                 description="Render paintsim frames (You et al. 2013, Sec. 5).")
    ap.add_argument("path", help="frame directory (-> mp4) or a single frame_*.npz (-> png)")
    ap.add_argument("-o", "--out", default=None, help="output .mp4 / .png path")
    ap.add_argument("--camera", choices=("top", "oblique"), default="top")
    ap.add_argument("--size", default="960x540", help="WIDTHxHEIGHT")
    ap.add_argument("--spp", type=int, default=DEFAULTS["spp"], help="samples per pixel per axis")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--extent", default=None, help="x0,x1,y0,y1[,zmax] in metres (default: auto)")
    ap.add_argument("--obj", default=None, help="also write the liquid mesh(es) as OBJ into this directory")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    extent = tuple(float(v) for v in a.extent.split(",")) if a.extent else None
    if os.path.isdir(a.path):
        out = a.out or os.path.join(a.path, "render_%s.mp4" % a.camera)
        render_sequence(a.path, out, fps=a.fps, camera=a.camera, width=w, height=h,
                        extent=extent or "auto", crf=a.crf, obj_dir=a.obj, spp=a.spp)
    else:
        out = a.out or os.path.splitext(a.path)[0] + "_%s.png" % a.camera
        info = {}
        img = render_frame(a.path, camera=a.camera, width=w, height=h, extent=extent, info=info, spp=a.spp)
        _save_png(img, out)
        if a.obj:
            os.makedirs(a.obj, exist_ok=True)
            v, fc, nr = frame_mesh(a.path)
            write_obj(os.path.join(a.obj, os.path.splitext(os.path.basename(a.path))[0] + ".obj"), v, fc, nr)
        print("[render] %s  %.2fs  (sdf %.2f, pigment %.2f, paper %.2f, trace %.2f)  sdf grid %s"
              % (out, info["total"], info["sdf"], info["pigment"], info["paper"], info["trace"],
                 info["sdf_shape"]))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
