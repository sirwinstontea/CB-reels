"""Fast synthetic tests for paintsim rendering (surface.py, render.py).

Run from the project root:  python -m pytest paintsim/tests/test_render.py -q
(The first run JIT-compiles the numba kernels, later runs load them from cache.)
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paintsim.render import render_frame, render_sequence  # noqa: E402
from paintsim.surface import (extract_mesh, paper_pigment_map, pigment_grid, write_obj,  # noqa: E402
                              zhu_bridson_sdf)

D0 = 0.005


def block(n=(8, 8, 6), d0=D0, z0=0.5 * D0):
    """Particles on a cubic lattice, centres at (i, j, k) * d0 + (0, 0, z0)."""
    g = np.stack(np.meshgrid(*[np.arange(m) * d0 for m in n], indexing="ij"), -1).reshape(-1, 3)
    g[:, 2] += z0
    return g


def grid_around(x, cell, pad):
    lo = x.min(0) - pad
    hi = x.max(0) + pad
    shape = tuple(int(s) for s in np.ceil((hi - lo) / cell).astype(int) + 1)
    return lo, shape


def zero_crossing(values, coords):
    k = np.nonzero((values[:-1] < 0) & (values[1:] >= 0))[0]
    assert len(k), "no - to + crossing"
    k = k[-1]
    return coords[k] - values[k] * (coords[k + 1] - coords[k]) / (values[k + 1] - values[k])


# --------------------------------------------------------------------------
def test_sdf_sign_and_face_position():
    x = block()
    cell = D0 / 4
    lo, shape = grid_around(x, cell, 3 * D0)
    sdf = zhu_bridson_sdf(x, 0.5 * D0, lo, cell, shape, 2 * D0)
    assert sdf.shape == shape and sdf.dtype == np.float32
    axes = [lo[a] + np.arange(shape[a]) * cell for a in range(3)]
    ctr = x.mean(0)
    ci = [int(round((ctr[a] - lo[a]) / cell)) for a in range(3)]
    # deep inside: negative (about -r); far outside: positive
    assert sdf[ci[0], ci[1], ci[2]] < -0.4 * D0
    assert sdf[0, 0, 0] > 0 and sdf[-1, -1, -1] > 0
    inside = np.all((np.stack(np.meshgrid(*axes, indexing="ij"), -1) > x.min(0) + 0.5 * D0)
                    & (np.stack(np.meshgrid(*axes, indexing="ij"), -1) < x.max(0) - 0.5 * D0), axis=-1)
    assert np.all(sdf[inside] < 0)
    # zero crossing within 0.5 d0 of every block face (face = outer layer +- d0/2)
    line_x = sdf[:, ci[1], ci[2]]
    line_z = sdf[ci[0], ci[1], :]
    hi_x = zero_crossing(line_x, axes[0])
    lo_x = -zero_crossing(line_x[::-1], -axes[0][::-1])
    top_z = zero_crossing(line_z, axes[2])
    assert abs(hi_x - (x[:, 0].max() + 0.5 * D0)) < 0.5 * D0
    assert abs(lo_x - (x[:, 0].min() - 0.5 * D0)) < 0.5 * D0
    assert abs(top_z - (x[:, 2].max() + 0.5 * D0)) < 0.5 * D0


def test_sdf_excludes_far_points_and_empty():
    x = block((3, 3, 3))
    cell = D0 / 2
    lo, shape = grid_around(x, cell, 6 * D0)
    sdf = zhu_bridson_sdf(x, 0.5 * D0, lo, cell, shape, 2 * D0)
    assert sdf[0, 0, 0] == pytest.approx(2 * D0)  # no particle within R -> far value
    empty = zhu_bridson_sdf(np.zeros((0, 3)), 0.5 * D0, lo, cell, (4, 4, 4), 2 * D0, far=1.0)
    assert np.all(empty == 1.0)


def test_pigment_grid_uniform_block_reads_c():
    x = block((10, 10, 10))
    c = np.ones((len(x), 2))
    c[:, 1] = 0.25
    cell = D0 / 2
    lo, shape = grid_around(x, cell, 3 * D0)
    pg = pigment_grid(x, c, lo, cell, shape, 2 * D0, d0=D0)
    assert pg.shape == (2,) + shape and pg.dtype == np.float32
    ctr = x.mean(0)
    i, j, k = (int(round((ctr[a] - lo[a]) / cell)) for a in range(3))
    core = pg[:, i - 4:i + 5, j - 4:j + 5, k - 4:k + 5]
    assert np.allclose(core[0], 1.0, atol=0.02)
    assert np.allclose(core[1], 0.25, atol=0.01)
    assert pg[0, 0, 0, 0] == 0.0  # far from liquid: no pigment
    assert pg.max() <= 1.0 + 1e-5  # never overshoots the concentration


def test_paper_map_conserves_pigment():
    rng = np.random.default_rng(0)
    xs = np.arange(-20, 21) * D0
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    paper_x = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    r = np.hypot(X, Y).ravel()
    paper_c = np.zeros((len(paper_x), 2))
    paper_c[:, 0] = 0.3 * D0 * np.exp(-0.5 * ((r - 0.05) / 0.01) ** 2)
    paper_c[:, 1] = rng.uniform(0, 0.1 * D0, len(paper_x)) * (r < 0.06)
    cell = D0 / 4
    o = (-0.14, -0.13)
    shape = (int(0.28 / cell) + 1, int(0.27 / cell) + 1)
    m = paper_pigment_map(paper_x, paper_c, o, cell, shape, 1.5 * D0, d0=D0)
    assert m.shape == (2,) + shape and m.dtype == np.float32
    deposited = (paper_c * D0 * D0).sum(0)
    on_map = m.astype(np.float64).sum(axis=(1, 2)) * cell * cell
    assert np.allclose(on_map, deposited, rtol=1e-4)
    # explicit paper area is honoured
    m2 = paper_pigment_map(paper_x, paper_c, o, cell, shape, 1.5 * D0, d0=D0, paper_area=2 * D0 * D0)
    assert np.allclose(m2.astype(np.float64).sum(axis=(1, 2)) * cell * cell, 2 * deposited, rtol=1e-4)


def test_paper_map_contact_staining():
    paper_x = np.zeros((0, 3))
    paper_c = np.zeros((0, 1))
    liquid = np.array([[0.0, 0.0, 0.5 * D0],     # touching the paper
                       [0.02, 0.0, 6 * D0],      # in the air: no stain
                       [-0.02, 0.0, 0.5 * D0]])  # absorbed: no stain
    lc = np.ones((3, 1))
    absorbed = np.array([False, False, True])
    cell = D0 / 4
    m = paper_pigment_map(paper_x, paper_c, (-0.05, -0.05), cell, (401, 401), 1.5 * D0, d0=D0,
                          liquid_x=liquid, liquid_c=lc, absorbed=absorbed, contact_stain=0.1)
    total = m.astype(np.float64).sum() * cell * cell
    assert total == pytest.approx(0.1 * D0 ** 3, rel=1e-4)
    i0 = int(round(0.05 / cell))
    assert m[0, i0, i0] > 0 and m[0, i0 + int(0.02 / cell), i0] == 0


def test_mesh_is_outward_and_obj_roundtrip(tmp_path):
    x = block((6, 6, 6))
    cell = D0 / 3
    lo, shape = grid_around(x, cell, 3 * D0)
    sdf = zhu_bridson_sdf(x, 0.5 * D0, lo, cell, shape, 2 * D0)
    v, f, n = extract_mesh(sdf, lo, cell)
    assert len(v) > 100 and f.shape[1] == 3
    ctr = x.mean(0)
    assert np.mean(np.sum((v - ctr) * n, axis=1) > 0) > 0.99  # normals point away from the centre
    a, b, c = v[f[:, 0]].astype(np.float64), v[f[:, 1]].astype(np.float64), v[f[:, 2]].astype(np.float64)
    vol = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    assert vol > 0  # counter-clockwise from outside -> positive enclosed volume
    assert vol == pytest.approx((6 * D0) ** 3, rel=0.25)
    p = write_obj(str(tmp_path / "m.obj"), v, f, n)
    lines = open(p).read().splitlines()
    assert sum(ln.startswith("v ") for ln in lines) == len(v)
    assert sum(ln.startswith("f ") for ln in lines) == len(f)


# --------------------------------------------------------------------------
def white_paper_frame(with_liquid=False):
    xs = np.arange(-10, 11) * D0
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    paper_x = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1).astype(np.float32)
    if with_liquid:
        x = block((6, 6, 3)).astype(np.float32) - np.array([2.5 * D0, 2.5 * D0, 0], np.float32)
    else:
        x = np.zeros((0, 3), np.float32)
    n = len(x)
    return dict(x=x, v=np.zeros((n, 3), np.float32), kind=np.zeros(n, np.int8),
                c=np.ones((n, 1), np.float32), absorbed=np.zeros(n, bool), paper_x=paper_x,
                paper_c=np.zeros((len(paper_x), 1), np.float32),
                pigment_rgb=np.array([[0.8, 0.05, 0.05]], np.float32),
                d0=np.float32(D0), h=np.float32(2 * D0), time=np.float32(0))


def test_render_white_paper_is_near_white():
    fr = white_paper_frame()
    img = render_frame(fr, camera="top", width=64, height=36, extent=(-0.05, 0.05, -0.03, 0.03), spp=1)
    assert img.shape == (36, 64, 3) and img.dtype == np.uint8
    mean = img.reshape(-1, 3).mean(0)
    assert np.all(mean > 225), mean
    assert np.ptp(mean) < 12, mean  # neutral, not tinted
    img2 = render_frame(fr, camera="oblique", width=48, height=32, extent=(-0.05, 0.05, -0.03, 0.03), spp=1)
    assert img2.shape == (32, 48, 3) and img2.dtype == np.uint8


def test_render_red_blob_is_red_in_the_middle():
    fr = white_paper_frame(with_liquid=True)
    img = render_frame(fr, camera="top", width=64, height=64, extent=(-0.04, 0.04, -0.04, 0.04), spp=1)
    mid = img[28:36, 28:36].reshape(-1, 3).astype(float).mean(0)
    corner = img[:6, :6].reshape(-1, 3).astype(float).mean(0)
    assert mid[0] > mid[1] + 60 and mid[0] > mid[2] + 60, mid  # saturated red under thick paint
    assert np.all(corner > 220), corner


def test_render_ignores_stray_particles():
    fr = white_paper_frame(with_liquid=True)
    stray = np.array([[0.3, -8.0, 59.0], [0.0, 0.0, 40.0]], np.float32)  # ejected by a bad step
    fr["x"] = np.concatenate([np.repeat(fr["x"], 1, 0), stray])
    n = len(fr["x"])
    fr.update(v=np.zeros((n, 3), np.float32), kind=np.zeros(n, np.int8), c=np.ones((n, 1), np.float32),
              absorbed=np.zeros(n, bool))
    info = {}
    img = render_frame(fr, camera="oblique", width=48, height=32, spp=1, info=info)
    assert img.shape == (32, 48, 3)
    assert np.prod(info["sdf_shape"]) < 2_000_000 and info["sdf_cell"] < D0  # grid not blown up
    assert info["extent"][1] - info["extent"][0] < 0.5


def test_render_uses_frame_paper_stain():
    fr = white_paper_frame()
    plain = render_frame(fr, width=32, height=18, extent=(-0.04, 0.04, -0.03, 0.03), spp=1)
    fr["paper_stain"] = np.ones((len(fr["paper_x"]), 1), np.float32)
    stained = render_frame(fr, width=32, height=18, extent=(-0.04, 0.04, -0.03, 0.03), spp=1)
    p = plain.reshape(-1, 3).astype(float).mean(0)
    s = stained.reshape(-1, 3).astype(float).mean(0)
    assert s[1] < p[1] - 20 and s[0] > s[1] + 20, (p, s)  # red pigment stain on the paper
    off = render_frame(fr, width=32, height=18, extent=(-0.04, 0.04, -0.03, 0.03), spp=1, stain_source="none")
    assert np.abs(off.astype(float).mean() - plain.astype(float).mean()) < 1.0


def test_render_sequence_writes_mp4(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    for i in range(2):
        fr = white_paper_frame(with_liquid=True)
        fr["x"] = fr["x"] + np.float32(i * D0)
        np.savez(tmp_path / ("frame_%05d.npz" % i), **fr)
    out = str(tmp_path / "seq.mp4")
    render_sequence(str(tmp_path), out, fps=10, width=64, height=48, spp=1, progress=False)
    assert os.path.getsize(out) > 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames,pix_fmt", "-of", "csv=p=0", out],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        assert "yuv420p" in probe.stdout and "2" in probe.stdout
