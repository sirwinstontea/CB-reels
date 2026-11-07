"""The director: picks which moments of the coloring session become shots, and how each is framed.

A reel is ~6 shots in time order (the page only ever gains color between cuts):
close-ups and medium shots of a vivid region being filled, one wide shot, and a final
hand-free wide shot of the finished page (that one never goes to the video model).
"""
from dataclasses import dataclass, asdict

import numpy as np

from .colors import srgb_to_lab
from .scene import View, OUT_W, OUT_H

# (zoom, angle, seconds, where in the session as a fraction of its length)
# Whole seconds: the video model bills and outputs whole seconds (Kling 4.0 edit: 3-30 s).
TEMPLATE = [
    ("close", "A", 3.0, 0.30),
    ("medium", "B", 3.0, 0.45),
    ("close", "B", 3.0, 0.58),
    ("wide", "A", 3.0, 0.70),
    ("close", "A", 3.0, 0.85),
]
FINAL = ("wide", "A", 3.0)
# crop width as a share of the page's width on the desk; the reference close-ups frame ~100-180 mm of page,
# tighter than that makes every stroke look oversized
ZOOM_W = {"wide": None, "medium": 0.7, "close": 0.45}


@dataclass
class Shot:
    id: str
    zoom: str
    angle: str
    t0: float
    t1: float
    cx: float
    cy: float
    width: float
    hand: bool
    region: int = -1
    marker: int = -1

    def view(self, desks):
        return View(desks[self.angle], self.cx, self.cy, self.width).clamp()

    def to_dict(self):
        return asdict(self)


def _window_ok(tl, t0, t1, rid):
    i0, i1 = np.searchsorted(tl.t, [t0, t1])
    if i1 - i0 < 10:
        return 0.0
    seg = slice(i0, i1)
    return float(np.mean(tl.down[seg] & (tl.region[seg] == rid)))


def plan_shots(tl, page, markers, desks, seed=0):
    rng = np.random.default_rng(seed)
    chroma = {i: float(np.hypot(*srgb_to_lab([m["ink"]])[0, 1:])) for i, m in enumerate(markers)}
    region_marker = {}
    for i in np.flatnonzero(tl.down):
        region_marker.setdefault(int(tl.region[i]), int(tl.marker[i]))
    T = tl.duration
    shots, used, last_t = [], set(), 0.0
    for n, (zoom, angle, dur, frac) in enumerate(TEMPLATE):
        best = None
        for rid, rs in tl.region_start.items():
            re = tl.region_end[rid]
            if rid in used or re - rs < dur + 1.0 or rs < last_t:
                continue
            t0 = rs + min(2.0, (re - rs - dur) / 2)
            fill = _window_ok(tl, t0, t0 + dur, rid)
            if fill < 0.6:
                continue
            area = page.regions[rid].area / page.px_per_mm ** 2
            score = (np.log1p(area) * (0.4 + chroma.get(region_marker.get(rid, -1), 0) / 60) * fill
                     - 30.0 * abs(t0 / T - frac))       # stay near its slot, or later shots run out of page
            if best is None or score > best[0]:
                best = (score, rid, t0)
        if best is None:
            continue
        _, rid, t0 = best
        used.add(rid)
        t1 = t0 + dur
        last_t = t1
        desk = desks[angle]
        seg = (tl.t >= t0) & (tl.t < t1) & tl.down
        cx, cy, width = _frame(desk, page, zoom, tl.x[seg], tl.y[seg])
        shots.append(Shot(f"s{n + 1:02d}", zoom, angle, round(t0, 3), round(t1, 3), cx, cy, width, True,
                          rid, region_marker.get(rid, -1)))
    z, a, d = FINAL
    cx, cy, width = _frame(desks[a], page, z, [page.lines.shape[1] / 2], [page.lines.shape[0] / 2])
    shots.append(Shot("final", z, a, round(T + 1.0, 3), round(T + 1.0 + d, 3), cx, cy, width, False))
    return shots


def _frame(desk, page, zoom, xs, ys):
    """Crop (center, width) in desk px that keeps the nib path (page px xs, ys) in view, with room for the hand."""
    import cv2
    from .scene import page_to_desk, page_quad
    q = page_quad(desk, page.lines.shape)
    if ZOOM_W[zoom] is None:           # the whole page plus some desk, as wide as the picture allows
        Hh, Ww = desk.img.shape[:2]
        widest = float(min(Ww, Hh * OUT_W / OUT_H))
        need = max(np.ptp(q[:, 0]) * 1.12, np.ptp(q[:, 1]) * 1.06 * OUT_W / OUT_H)
        c = q.mean(0)
        return float(c[0]), float(c[1]), float(min(widest, max(need, 0.9 * widest)))
    H = page_to_desk(desk, page.lines.shape)
    pts = np.stack([np.asarray(xs, np.float32), np.asarray(ys, np.float32)], 1)[None]
    d = cv2.perspectiveTransform(pts, H)[0]
    lo, hi = np.percentile(d, 3, axis=0), np.percentile(d, 97, axis=0)
    page_w = float(np.linalg.norm(q[1] - q[0]))
    width = max(ZOOM_W[zoom] * page_w, (hi[0] - lo[0]) * 1.5, (hi[1] - lo[1]) * 1.6 * OUT_W / OUT_H)
    height = width * OUT_H / OUT_W
    u = np.asarray(desk.pen_dir)
    c = (lo + hi) / 2 + u * height * 0.12      # the hand trails toward pen_dir: keep the nib above center
    return float(c[0]), float(c[1]), float(width)
