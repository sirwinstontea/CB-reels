"""Liquid surface and pigment density fields for rendering (You et al. 2013, Sec. 5).

Everything here works on regular node grids: sample ``(i, j, k)`` sits at
``origin + (i, j, k) * cell`` and ``shape = (nx, ny, nz)``. World units are SI
metres, z up, paper top at z = 0 (see SPEC.md).

Public API
----------
zhu_bridson_sdf(x, radius, origin, cell, shape, R, far=None) -> (nx, ny, nz) float32
    Eq. 11, phi(x) = |x - x_bar| - r_bar (Zhu & Bridson 2005).
pigment_grid(x, c, origin, cell, shape, r, d0=None) -> (K, nx, ny, nz) float32
    Grid pigment density, concentration gathered within radius r (Sec. 5.2).
paper_pigment_map(paper_x, paper_c, origin_xy, cell, shape_xy, r, ...) -> (K, nx, ny) float32
    Pigment left on the paper plus live contact staining from touching liquid.
extract_mesh(sdf, origin, cell, level=0.0) -> (verts, faces, normals)
write_obj(path, verts, faces, normals=None)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

__all__ = [
    "zhu_bridson_sdf",
    "pigment_grid",
    "paper_pigment_map",
    "extract_mesh",
    "write_obj",
    "kernel_lattice_sum",
]

_MAX_BUCKETS = 1 << 22  # cap on the dense bucket grid (16 MB of int32 offsets)


# --------------------------------------------------------------------------
# neighbour search: uniform bucket grid built with a counting sort
# --------------------------------------------------------------------------
@njit(cache=True)
def _bucket_sort(px, py, pz, lo0, lo1, lo2, bs, d0_, d1_, d2_):
    n = px.shape[0]
    nb = d0_ * d1_ * d2_
    start = np.zeros(nb + 1, np.int32)
    bid = np.empty(n, np.int32)
    inv = 1.0 / bs
    for p in range(n):
        i = int((px[p] - lo0) * inv)
        j = int((py[p] - lo1) * inv)
        k = int((pz[p] - lo2) * inv)
        i = min(max(i, 0), d0_ - 1)
        j = min(max(j, 0), d1_ - 1)
        k = min(max(k, 0), d2_ - 1)
        b = (i * d1_ + j) * d2_ + k
        bid[p] = b
        start[b + 1] += 1
    for b in range(nb):
        start[b + 1] += start[b]
    fill = start[:-1].copy()
    order = np.empty(n, np.int32)
    for p in range(n):
        b = bid[p]
        order[fill[b]] = p
        fill[b] += 1
    return start, order


class _Buckets:
    """Particles sorted into a dense bucket grid of size >= ``reach``.

    Only particles inside ``[lo - reach, hi + reach]`` are kept: nothing else
    can influence a sample in ``[lo, hi]``.
    """

    def __init__(self, x, lo, hi, reach):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        lo = np.asarray(lo, np.float64) - reach
        hi = np.asarray(hi, np.float64) + reach
        keep = np.all((x >= lo) & (x <= hi), axis=1)
        self.index = np.nonzero(keep)[0]
        xs = x[keep]
        bs = float(reach)
        ext = np.maximum(hi - lo, bs)
        dims = np.maximum(np.ceil(ext / bs).astype(np.int64), 1)
        while int(np.prod(dims)) > _MAX_BUCKETS:
            bs *= 1.25
            dims = np.maximum(np.ceil(ext / bs).astype(np.int64), 1)
        self.lo = lo
        self.bs = bs
        self.dims = dims.astype(np.int64)
        if xs.shape[0]:
            start, order = _bucket_sort(xs[:, 0], xs[:, 1], xs[:, 2], lo[0], lo[1], lo[2], bs,
                                        int(dims[0]), int(dims[1]), int(dims[2]))
        else:
            start = np.zeros(int(np.prod(dims)) + 1, np.int32)
            order = np.zeros(0, np.int32)
        self.start = start
        self.order = order
        xs = xs[order]
        self.sx = np.ascontiguousarray(xs[:, 0])
        self.sy = np.ascontiguousarray(xs[:, 1])
        self.sz = np.ascontiguousarray(xs[:, 2])

    def sorted_rows(self, a):
        """Rows of a per-particle array, in bucket order."""
        a = np.asarray(a)
        return np.ascontiguousarray(a[self.index][self.order])


# --------------------------------------------------------------------------
# 5.1  Zhu & Bridson signed distance (Eq. 11)
# --------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _zb_kernel(sx, sy, sz, sr, start, blo, bs, bdim, org, cell, nx, ny, nz, R, far, out):
    R2 = R * R
    invR2 = 1.0 / R2
    invbs = 1.0 / bs
    d0_, d1_, d2_ = bdim[0], bdim[1], bdim[2]
    for i in prange(nx):
        x = org[0] + i * cell
        bi0 = max(int(math.floor((x - R - blo[0]) * invbs)), 0)
        bi1 = min(int(math.floor((x + R - blo[0]) * invbs)), d0_ - 1)
        for j in range(ny):
            y = org[1] + j * cell
            bj0 = max(int(math.floor((y - R - blo[1]) * invbs)), 0)
            bj1 = min(int(math.floor((y + R - blo[1]) * invbs)), d1_ - 1)
            for k in range(nz):
                z = org[2] + k * cell
                bk0 = max(int(math.floor((z - R - blo[2]) * invbs)), 0)
                bk1 = min(int(math.floor((z + R - blo[2]) * invbs)), d2_ - 1)
                sw = 0.0
                swx = 0.0
                swy = 0.0
                swz = 0.0
                swr = 0.0
                if bi0 <= bi1 and bj0 <= bj1 and bk0 <= bk1:
                    for bi in range(bi0, bi1 + 1):
                        for bj in range(bj0, bj1 + 1):
                            base = (bi * d1_ + bj) * d2_
                            q0 = start[base + bk0]
                            q1 = start[base + bk1 + 1]
                            for q in range(q0, q1):
                                dx = sx[q] - x
                                dy = sy[q] - y
                                dz = sz[q] - z
                                d2 = dx * dx + dy * dy + dz * dz
                                if d2 < R2:
                                    s = 1.0 - d2 * invR2
                                    w = s * s * s
                                    sw += w
                                    swx += w * dx
                                    swy += w * dy
                                    swz += w * dz
                                    swr += w * sr[q]
                if sw > 0.0:
                    inv = 1.0 / sw
                    # x - x_bar, accumulated as offsets for precision
                    mx = swx * inv
                    my = swy * inv
                    mz = swz * inv
                    out[i, j, k] = math.sqrt(mx * mx + my * my + mz * mz) - swr * inv
                else:
                    out[i, j, k] = far


def _grid_box(origin, cell, shape):
    origin = np.asarray(origin, np.float64).reshape(3)
    shape = tuple(int(s) for s in shape)
    hi = origin + (np.asarray(shape, np.float64) - 1.0) * float(cell)
    return origin, shape, hi


def zhu_bridson_sdf(x, radius, origin, cell, shape, R, far=None):
    """Signed distance of the liquid, Eq. 11 of the paper (Zhu & Bridson 2005).

    phi(p) = |p - x_bar(p)| - r_bar(p),  x_bar = sum_i w_i x_i,  r_bar = sum_i w_i r_i,
    w_i = k(|p - x_i| / R) / sum_j k(|p - x_j| / R),  k(s) = max(0, (1 - s^2)^3).

    Parameters
    ----------
    x : (N, 3) particle positions (pass only the non-absorbed liquid, Sec. 5.1).
    radius : scalar or (N,) particle radii r_i. 0.5 * d0 puts the zero level of a
        lattice block ~0.05 d0 inside the block face (the particles' volume boundary).
    origin, cell, shape : node grid, sample (i, j, k) at origin + (i, j, k) * cell.
    R : kernel radius, Zhu & Bridson use twice the particle spacing (2 d0 = h).
    far : value for samples with no particle within R (default R: positive, larger
        than any value in the band, and keeps trilinear interpolation well-behaved).

    Returns (nx, ny, nz) float32, negative inside the liquid.
    """
    origin, shape, hi = _grid_box(origin, cell, shape)
    R = float(R)
    far = R if far is None else float(far)
    x = np.asarray(x, np.float64).reshape(-1, 3)
    out = np.empty(shape, np.float32)
    if x.shape[0] == 0:
        out.fill(far)
        return out
    rad = np.broadcast_to(np.asarray(radius, np.float64), (x.shape[0],))
    bk = _Buckets(x, origin, hi, R)
    if bk.sx.shape[0] == 0:
        out.fill(far)
        return out
    sr = bk.sorted_rows(rad).astype(np.float64)
    _zb_kernel(bk.sx, bk.sy, bk.sz, sr, bk.start, bk.lo, bk.bs, bk.dims, origin, float(cell),
               shape[0], shape[1], shape[2], R, far, out)
    return out


# --------------------------------------------------------------------------
# 5.2  pigment density grid
# --------------------------------------------------------------------------
def kernel_lattice_sum(r, d0, dim=3):
    """Sum of k(|p - x_i| / r) = (1 - s^2)^3 over a full cubic (dim=3) or square
    (dim=2) lattice of spacing d0, averaged over sample offsets. This is the kernel
    weight a sample deep inside uniformly filled liquid sees."""
    n = int(math.ceil(r / d0)) + 1
    ax = np.arange(-n, n + 1, dtype=np.float64) * d0
    offs = (np.arange(4) + 0.5) / 4.0 * d0
    tot = 0.0
    cnt = 0
    if dim == 3:
        for ox in offs:
            for oy in offs:
                for oz in offs:
                    gx, gy, gz = np.meshgrid(ax + ox, ax + oy, ax + oz, indexing="ij")
                    s2 = (gx * gx + gy * gy + gz * gz) / (r * r)
                    tot += np.sum(np.clip(1.0 - s2, 0.0, None) ** 3)
                    cnt += 1
    else:
        for ox in offs:
            for oy in offs:
                gx, gy = np.meshgrid(ax + ox, ax + oy, indexing="ij")
                s2 = (gx * gx + gy * gy) / (r * r)
                tot += np.sum(np.clip(1.0 - s2, 0.0, None) ** 3)
                cnt += 1
    return tot / cnt


@njit(parallel=True, fastmath=True, cache=True)
def _pigment_kernel(sx, sy, sz, sc, start, blo, bs, bdim, org, cell, nx, ny, nz, r, wfloor, out):
    K = sc.shape[1]
    r2 = r * r
    invr2 = 1.0 / r2
    invbs = 1.0 / bs
    d0_, d1_, d2_ = bdim[0], bdim[1], bdim[2]
    for i in prange(nx):
        acc = np.zeros(K, np.float64)
        x = org[0] + i * cell
        bi0 = max(int(math.floor((x - r - blo[0]) * invbs)), 0)
        bi1 = min(int(math.floor((x + r - blo[0]) * invbs)), d0_ - 1)
        for j in range(ny):
            y = org[1] + j * cell
            bj0 = max(int(math.floor((y - r - blo[1]) * invbs)), 0)
            bj1 = min(int(math.floor((y + r - blo[1]) * invbs)), d1_ - 1)
            for k in range(nz):
                z = org[2] + k * cell
                bk0 = max(int(math.floor((z - r - blo[2]) * invbs)), 0)
                bk1 = min(int(math.floor((z + r - blo[2]) * invbs)), d2_ - 1)
                sw = 0.0
                for m in range(K):
                    acc[m] = 0.0
                if bi0 <= bi1 and bj0 <= bj1 and bk0 <= bk1:
                    for bi in range(bi0, bi1 + 1):
                        for bj in range(bj0, bj1 + 1):
                            base = (bi * d1_ + bj) * d2_
                            q0 = start[base + bk0]
                            q1 = start[base + bk1 + 1]
                            for q in range(q0, q1):
                                dx = sx[q] - x
                                dy = sy[q] - y
                                dz = sz[q] - z
                                d2 = dx * dx + dy * dy + dz * dz
                                if d2 < r2:
                                    s = 1.0 - d2 * invr2
                                    w = s * s * s
                                    sw += w
                                    for m in range(K):
                                        acc[m] += w * sc[q, m]
                inv = 1.0 / max(sw, wfloor)
                for m in range(K):
                    out[m, i, j, k] = acc[m] * inv


def pigment_grid(x, c, origin, cell, shape, r, d0=None):
    """Grid pigment density: concentration gathered within radius ``r`` (Sec. 5.2).

    Kernel: k(s) = (1 - s^2)^3 for s = |p - x_i| / r < 1 (the same poly6 shape as
    Zhu & Bridson's weight). Normalisation:

        rho_k(p) = sum_i k_i c_ik / max(sum_i k_i, 0.25 * S_full)

    where S_full = ``kernel_lattice_sum(r, d0)`` is the weight sum a sample sees
    inside a fully filled cubic lattice of spacing d0. Inside uniformly filled
    liquid sum_i k_i ~= S_full, so rho = c exactly (a Shepard average of the
    concentrations, no lattice ripple). Towards and past the free surface the
    weight sum drops and rho tapers smoothly to 0, so the field never extrapolates
    pigment into empty space. Units: concentration (pigment volume fraction).

    Parameters: x (N, 3) non-absorbed liquid positions, c (N, K) concentrations,
    node grid origin/cell/shape, gather radius r (default choice in the renderer:
    h = 2 d0), d0 particle spacing (default r / 2).
    Returns (K, nx, ny, nz) float32.
    """
    origin, shape, hi = _grid_box(origin, cell, shape)
    x = np.asarray(x, np.float64).reshape(-1, 3)
    c = np.asarray(c, np.float64)
    if c.ndim == 1:
        c = c[:, None]
    K = c.shape[1]
    out = np.zeros((K,) + shape, np.float32)
    if x.shape[0] == 0:
        return out
    r = float(r)
    d0 = r / 2.0 if d0 is None else float(d0)
    wfloor = 0.25 * kernel_lattice_sum(r, d0, 3)
    bk = _Buckets(x, origin, hi, r)
    if bk.sx.shape[0] == 0:
        return out
    sc = bk.sorted_rows(c).astype(np.float64)
    _pigment_kernel(bk.sx, bk.sy, bk.sz, sc, bk.start, bk.lo, bk.bs, bk.dims, origin, float(cell),
                    shape[0], shape[1], shape[2], r, wfloor, out)
    return out


# --------------------------------------------------------------------------
# 5.2  pigment left on the paper (2D surface density map)
# --------------------------------------------------------------------------
@njit(cache=True)
def _splat2d(px, py, mass, ox, oy, cell, nx, ny, r, out):
    """Mass-conserving splat. Each particle spreads mass[p, :] over the cells within
    radius r with weights (1 - s^2)^3 normalised over all of its (in or out of map)
    cells, so mass is conserved exactly for footprints inside the map."""
    K = mass.shape[1]
    r2 = r * r
    inv_area = 1.0 / (cell * cell)
    for p in range(px.shape[0]):
        tot = 0.0
        for m in range(K):
            tot += abs(mass[p, m])
        if tot == 0.0:
            continue
        x = px[p]
        y = py[p]
        i0 = int(math.ceil((x - r - ox) / cell))
        i1 = int(math.floor((x + r - ox) / cell))
        j0 = int(math.ceil((y - r - oy) / cell))
        j1 = int(math.floor((y + r - oy) / cell))
        wsum = 0.0
        for i in range(i0, i1 + 1):
            dx = ox + i * cell - x
            for j in range(j0, j1 + 1):
                dy = oy + j * cell - y
                d2 = dx * dx + dy * dy
                if d2 < r2:
                    s = 1.0 - d2 / r2
                    wsum += s * s * s
        if wsum <= 0.0:
            # kernel narrower than a cell: bilinear splat
            fx = (x - ox) / cell
            fy = (y - oy) / cell
            ii = int(math.floor(fx))
            jj = int(math.floor(fy))
            tx = fx - ii
            ty = fy - jj
            for a in range(2):
                for b in range(2):
                    wi = tx if a == 1 else 1.0 - tx
                    wj = ty if b == 1 else 1.0 - ty
                    ia = ii + a
                    jb = jj + b
                    if 0 <= ia < nx and 0 <= jb < ny:
                        for m in range(K):
                            out[m, ia, jb] += wi * wj * mass[p, m] * inv_area
            continue
        winv = 1.0 / wsum
        for i in range(max(i0, 0), min(i1, nx - 1) + 1):
            dx = ox + i * cell - x
            for j in range(max(j0, 0), min(j1, ny - 1) + 1):
                dy = oy + j * cell - y
                d2 = dx * dx + dy * dy
                if d2 < r2:
                    s = 1.0 - d2 / r2
                    w = s * s * s * winv * inv_area
                    for m in range(K):
                        out[m, i, j] += w * mass[p, m]


def contact_weight(z, d0, contact_dist=None):
    """Weight in [0, 1] of a liquid particle touching the paper: 1 for centres at or
    below ``contact_dist`` (default d0: resting particles sit 0.5-1 d0 above z = 0),
    fading linearly to 0 half a spacing higher."""
    cd = float(d0) if contact_dist is None else float(contact_dist)
    return np.clip((cd + 0.5 * d0 - np.asarray(z, np.float64)) / (0.5 * d0), 0.0, 1.0)


def paper_pigment_map(paper_x, paper_c, origin_xy, cell, shape_xy, r, *, d0=0.005,
                      paper_area=None, liquid_x=None, liquid_c=None, absorbed=None,
                      contact_dist=None, contact_stain=0.1):
    """Pigment surface density on the paper, (K, nx, ny) float32, in metres.

    Units follow SPEC.md: ``paper_c`` is concentration x volume / paper particle
    area, i.e. the thickness of pure-pigment-concentration liquid that the deposit
    is equivalent to. Each paper particle carries pigment "mass" paper_c * A_p
    (A_p = ``paper_area``, default d0^2), splatted onto the map with the
    normalised kernel (1 - s^2)^3 within radius ``r`` and divided by the cell area,
    so ``map.sum(axis=(1, 2)) * cell^2 == (paper_c * A_p).sum(axis=0)`` for deposits
    inside the map. All paper layers are projected onto the sheet.

    Live contact staining (Sec. 5.2: "We allow the paper to have pigment density
    when the liquid particles are in contact with the paper"): non-absorbed liquid
    particles whose centre is within ~d0 of z = 0 (``contact_weight``) add
    contact_stain * w * c * d0^3 of pigment, i.e. a film ``contact_stain`` of a
    particle volume settles onto the fibres under the wet area. Pass liquid_x /
    liquid_c (/ absorbed) to enable it; contact_stain = 0 disables it.
    """
    nx, ny = int(shape_xy[0]), int(shape_xy[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    cell = float(cell)
    r = float(r)
    d0 = float(d0)
    A_p = d0 * d0 if paper_area is None else float(paper_area)
    pc = np.asarray(paper_c, np.float64)
    if pc.ndim == 1:
        pc = pc[:, None]
    K = pc.shape[1]
    if liquid_c is not None:
        lc = np.asarray(liquid_c, np.float64)
        K = max(K, lc.shape[1] if lc.ndim == 2 else 1)
    out = np.zeros((K, nx, ny), np.float64)
    pxy = np.asarray(paper_x, np.float64).reshape(-1, 3)
    if pxy.shape[0]:
        mass = np.zeros((pxy.shape[0], K))
        mass[:, :pc.shape[1]] = pc * A_p
        _splat2d(pxy[:, 0].copy(), pxy[:, 1].copy(), mass, ox, oy, cell, nx, ny, r, out)
    if liquid_x is not None and liquid_c is not None and contact_stain > 0:
        lx = np.asarray(liquid_x, np.float64).reshape(-1, 3)
        lc = np.asarray(liquid_c, np.float64)
        if lc.ndim == 1:
            lc = lc[:, None]
        sel = np.ones(lx.shape[0], bool) if absorbed is None else ~np.asarray(absorbed, bool)
        w = contact_weight(lx[:, 2], d0, contact_dist) * sel
        touch = w > 0
        if np.any(touch):
            mass = np.zeros((int(touch.sum()), K))
            mass[:, :lc.shape[1]] = lc[touch] * (contact_stain * d0 ** 3) * w[touch, None]
            _splat2d(lx[touch, 0].copy(), lx[touch, 1].copy(), mass, ox, oy, cell, nx, ny, r, out)
    return out.astype(np.float32)


def splat_density_2d(xy, mass, origin_xy, cell, shape_xy, r):
    """Generic mass-conserving 2D splat, (K, nx, ny) float64 of mass per area."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    mass = np.asarray(mass, np.float64)
    if mass.ndim == 1:
        mass = mass[:, None]
    out = np.zeros((mass.shape[1], int(shape_xy[0]), int(shape_xy[1])))
    if xy.shape[0]:
        _splat2d(xy[:, 0].copy(), xy[:, 1].copy(), mass, float(origin_xy[0]), float(origin_xy[1]),
                 float(cell), int(shape_xy[0]), int(shape_xy[1]), float(r), out)
    return out


# --------------------------------------------------------------------------
# meshes (for external rendering, e.g. Blender as a mental ray stand-in)
# --------------------------------------------------------------------------
def extract_mesh(sdf, origin, cell, level=0.0):
    """Marching cubes of the SDF at ``level``. Returns world-space verts (V, 3)
    float32, faces (F, 3) int32 wound counter-clockwise seen from outside, and unit
    outward normals (V, 3) float32 (the SDF gradient). Empty arrays if the field
    has no crossing."""
    from skimage.measure import marching_cubes

    sdf = np.asarray(sdf, np.float32)
    lo, hi = float(sdf.min()), float(sdf.max())
    if not (lo < level < hi) or min(sdf.shape) < 2:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))
    verts, faces, normals, _ = marching_cubes(sdf, level=level, spacing=(cell, cell, cell),
                                              gradient_direction="ascent", allow_degenerate=False)
    faces = faces.astype(np.int32)
    normals = normals.astype(np.float64)
    nl = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nl, 1e-12)
    # outward normal = +grad(phi): probe the field on both sides of the vertices
    idx = np.linspace(0, len(verts) - 1, min(len(verts), 256)).astype(int)
    hi_ix = np.asarray(sdf.shape) - 1

    def probe(p):
        q = np.clip(np.rint(p / cell).astype(int), 0, hi_ix)
        return sdf[q[:, 0], q[:, 1], q[:, 2]].astype(np.float64)

    step = 1.5 * cell * normals[idx]
    if np.sum(probe(verts[idx] + step) - probe(verts[idx] - step)) < 0:
        normals = -normals
    verts = verts.astype(np.float64) + np.asarray(origin, np.float64)[None, :]
    # fix the winding so face normals agree with the outward vertex normals
    if faces.shape[0]:
        a, b, cc = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = np.cross(b - a, cc - a)
        agree = np.sum(fn * (normals[faces[:, 0]] + normals[faces[:, 1]] + normals[faces[:, 2]]))
        if agree < 0:
            faces = faces[:, ::-1].copy()
    return verts.astype(np.float32), faces, normals.astype(np.float32)


def write_obj(path, verts, faces, normals=None):
    """Write a Wavefront OBJ (1-based indices, ``f a//a`` when normals are given)."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64) + 1
    parts = ["# paintsim liquid surface (Zhu & Bridson SDF, marching cubes)\n"]
    if len(verts):
        parts.append("\n".join("v %.6f %.6f %.6f" % tuple(v) for v in verts) + "\n")
    if normals is not None and len(normals):
        parts.append("\n".join("vn %.5f %.5f %.5f" % tuple(n) for n in np.asarray(normals)) + "\n")
        if len(faces):
            parts.append("\n".join("f %d//%d %d//%d %d//%d" % (a, a, b, b, c, c) for a, b, c in faces) + "\n")
    elif len(faces):
        parts.append("\n".join("f %d %d %d" % tuple(f) for f in faces) + "\n")
    with open(path, "w", encoding="ascii") as fh:
        fh.write("".join(parts))
    return path
