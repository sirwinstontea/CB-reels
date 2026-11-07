"""Plan every marker movement for a page and write it as one stroke timeline.

The strokes replay gestures recorded on the reference reels (engine/gestlib.py: nib paths from the
optical flow of the marker tip, pen down = the ASMR sound bursts). What the references showed:
- strokes run along the outline / the shape's length (median 19 deg off the nearest outline), worked
  edge-first, locally, each next stroke laid ~1-4 mm beside the last;
- ~3 pen-down bursts a second: 0.1-0.2 s on the paper, quick 0.04-0.12 s lifts; about half the turns
  go straight back the other way (zigzag), a quarter repeat the same direction (lift and return);
- a stroke lands softly and lifts off over its last third (the brush narrows: tapered ends).
Each gesture is resampled, scaled to the room the shape gives it, bent onto a streamline of the
shape's direction field and laid until the whole shape is covered (no paper gaps).

Timeline samples (RATE Hz): time, tip x/y in page px, pen down, marker, tip radius, sub-stroke id (one
pass between turns), region, pressure (0..1 ink amount; lift-offs taper toward 0).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import gestlib
from .colors import srgb_to_lab
from .page import region_mask

RATE = 120                  # samples per second
BRUSH_MM = 1.8              # brush tip radius: one stroke lays ~3-4.5 mm of ink (measured ~3.4 median)
FINE_MM = 0.5               # fine tip
FLOW_GAIN = 1.6             # flow of the barrel under-reads the nib (ink-arrival speed / flow speed, median)
SPEED_MM_S = (35, 260)      # a stroke's mean speed stays in this range
LEN_MM = (8.0, 30.0)        # brush stroke length (refs: ~8-35 mm on fills)
FINE_LEN_MM = (1.5, 7.0)
EDGE_BAND_MM = 4.0          # strokes this close to the outline run parallel to it
DEPTH_W = 0.6               # next stroke: nearest uncovered spot, edges first (cost = distance + DEPTH_W * depth)
COVER = 0.68                # a stroke covers this share of the tip radius each side (overlap -> no gaps)
CELL_MM = 0.35              # planning grid
CONTINUE_P = 0.70           # share of turns made without lifting (zigzag inside one sound burst)
TRAVEL_MM_S = 420           # lifted moves
MARKER_SWAP_S = (1.2, 2.0)  # cap off, swap, cap on
SHADE_MIN_MM2 = 250         # smaller shade/gradient layers are skipped (session length; invisible at reel size)
MAX_LAYERS = 2              # painted layers per region, base included (session length)
BIG_MM2 = 1000              # ...one more on big shapes, where shading shows most


@dataclass
class Timeline:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    down: np.ndarray
    marker: np.ndarray
    radius: np.ndarray
    stroke: np.ndarray
    region: np.ndarray
    region_start: dict      # region id -> time its coloring starts
    region_end: dict
    pressure: np.ndarray = None   # 0..1 ink per sample (None = old timeline = all 1.0)

    def __post_init__(self):
        if self.pressure is None or len(self.pressure) != len(self.t):
            self.pressure = np.ones(len(self.t), np.float32)

    @property
    def duration(self):
        return float(self.t[-1]) if len(self.t) else 0.0

    def save(self, path):
        np.savez_compressed(path, t=self.t, x=self.x, y=self.y, down=self.down, marker=self.marker,
                            radius=self.radius, stroke=self.stroke, region=self.region, pressure=self.pressure,
                            rs=np.array(list(self.region_start.items())), re=np.array(list(self.region_end.items())))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        d = lambda a: {int(k): float(v) for k, v in a} if a.size else {}
        return cls(z["t"], z["x"], z["y"], z["down"], z["marker"], z["radius"], z["stroke"], z["region"],
                   d(z["rs"]), d(z["re"]), z["pressure"] if "pressure" in z.files else None)


def _min_jerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


_LIB = None


def _lib():
    """Gestures as curves over u (time 0..1): s (position along the stroke 0..1), c (sideways, in
    stroke lengths), p (pressure); plus recorded duration, length and path."""
    global _LIB
    if _LIB is None:
        d = gestlib.load()
        G = []
        for g in d["gestures"]:
            t, a, c, p = (np.asarray(g[k], np.float64) for k in ("t", "along", "across", "pressure"))
            span = a.max() - a.min()
            if t[-1] < 0.03 or span < 1.0 or g["path_mm"] < 2.0:
                continue
            G.append(dict(u=t / t[-1], s=(a - a.min()) / span, c=c / span, p=p, dur=float(t[-1]),
                          span=float(span), path=float(g["path_mm"])))
        q = d["pen_up_s"]
        _LIB = dict(G=G, up=(float(q["25"]), float(q["50"]), float(q["75"])),
                    same=float(d["same_direction"]), back=float(d["back_and_forth"]))
    return _LIB


def _pen_up_s(rng):
    """Lift duration from the measured spread (log-normal through the quartiles)."""
    q1, q2, q3 = _lib()["up"]
    sig = max(0.2, np.log(q3 / q1) / 1.35)
    return float(np.clip(q2 * np.exp(rng.normal(0, sig)), 0.025, 0.35))


def _gesture(rng, length_mm, small=False, fast=1.0):
    """A recorded gesture fitted to length_mm: (s, c, p) at RATE. Longer strokes go faster (as in the
    refs); the recorded speed profile is kept."""
    G = _lib()["G"]
    pool = [g for g in G if (g["span"] < 4.0) == small] or G
    g = pool[rng.integers(len(pool))]
    k = length_mm / g["span"]
    v = g["path"] / g["dur"] * FLOW_GAIN * np.sqrt(k) * fast
    v = float(np.clip(v * rng.uniform(0.9, 1.1), SPEED_MM_S[0], SPEED_MM_S[1] * fast))
    n = max(4, int(round(g["path"] * k / v * RATE)))
    u = np.linspace(0, 1, n)
    return np.interp(u, g["u"], g["s"]), np.interp(u, g["u"], g["c"]), np.interp(u, g["u"], g["p"])


class _Writer:
    def __init__(self, rng):
        self.rng = rng
        self.cols = {k: [] for k in ("t", "x", "y", "down", "marker", "radius", "stroke", "region", "pressure")}
        self.time = 0.0
        self.pos = None
        self.marker = -1
        self.region = 0
        self.stroke_id = 0
        self.down = False

    def _emit(self, xs, ys, down, radius, stroke, pres=None):
        n = len(xs)
        ts = self.time + np.arange(1, n + 1) / RATE
        self.cols["t"].append(ts)
        self.cols["x"].append(np.asarray(xs, np.float32))
        self.cols["y"].append(np.asarray(ys, np.float32))
        self.cols["down"].append(np.full(n, bool(down), np.bool_))
        self.cols["marker"].append(np.full(n, self.marker, np.int16))
        self.cols["radius"].append(np.full(n, radius, np.float32))
        self.cols["stroke"].append(np.full(n, stroke, np.int32))
        self.cols["region"].append(np.full(n, self.region, np.int32))
        self.cols["pressure"].append(np.zeros(n, np.float32) if pres is None else np.asarray(pres, np.float32))
        self.time = float(ts[-1])
        self.pos = (float(xs[-1]), float(ys[-1]))
        self.down = bool(down)

    def hold(self, seconds, radius):
        n = max(1, int(seconds * RATE))
        x, y = self.pos
        self._emit(np.full(n, x), np.full(n, y), 0, radius, -1)

    def travel(self, to, ppm, radius, seconds=None):
        """Lifted move; the wrist arcs a little on the way."""
        if self.pos is None:
            self.pos = to
            return
        d = np.hypot(to[0] - self.pos[0], to[1] - self.pos[1])
        sec = max(0.03 + d / ppm / TRAVEL_MM_S, seconds if seconds is not None else _pen_up_s(self.rng))
        n = max(3, int(sec * RATE))
        s = _min_jerk(n)
        xs = self.pos[0] + (to[0] - self.pos[0]) * s
        ys = self.pos[1] + (to[1] - self.pos[1]) * s
        if d > 1:
            bend = np.sin(np.pi * s) * d * self.rng.uniform(-0.12, 0.12)
            xs, ys = xs - bend * (to[1] - self.pos[1]) / d, ys + bend * (to[0] - self.pos[0]) / d
        self._emit(xs, ys, 0, radius, -1)

    def turn(self, to, radius):
        """Turn without lifting: a quick hook onto the next pass, pen down, light."""
        x0, y0 = self.pos
        d = np.hypot(to[0] - x0, to[1] - y0)
        n = max(3, int((0.035 + d / (radius * 60)) * RATE))
        s = _min_jerk(n)
        self._emit(x0 + (to[0] - x0) * s, y0 + (to[1] - y0) * s, 1, radius, self.stroke_id,
                   0.35 + 0.15 * np.sin(np.pi * s))

    def stroke(self, xs, ys, pres, radius):
        self._emit(xs, ys, 1, radius, self.stroke_id, pres)
        self.stroke_id += 1

    def build(self):
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.cols.items()}


def _direction_field(mg, cell_mm, inner_dir, weight=None, directed=None):
    """Unit stroke direction per planning cell, and whether it is axial.
    Axial (base, stripes): parallel to the outline near it, blending into inner_dir deeper inside.
    Directed (shade, gradient): down the weight slope, from heavy ink toward the light end."""
    if directed is not None:
        v = np.broadcast_to(np.asarray(directed, np.float64), mg.shape + (2,)).copy()
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9), False
    if weight is not None:
        w = ndi.gaussian_filter(weight.astype(np.float64), max(1.0, 1.0 / cell_mm))
        gy, gx = np.gradient(w)
        v = np.stack([-gx, -gy], -1)
        nv = np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.where(nv > 1e-6, v / np.maximum(nv, 1e-9), np.asarray(inner_dir, np.float64))
        return v, False
    dist = ndi.distance_transform_edt(np.pad(mg, 1))[1:-1, 1:-1]
    gy, gx = np.gradient(ndi.gaussian_filter(dist, 1.2))
    nrm = np.hypot(gx, gy) + 1e-9
    tx, ty = -gy / nrm, gx / nrm                         # tangent to the iso-distance lines
    sig = max(1.0, 1.5 / cell_mm)
    c2 = ndi.gaussian_filter(tx * tx - ty * ty, sig)     # doubled angle: axial directions average properly
    s2 = ndi.gaussian_filter(2 * tx * ty, sig)
    a = np.arctan2(inner_dir[1], inner_dir[0])
    w = np.clip(dist * cell_mm / EDGE_BAND_MM, 0, 1) ** 1.5
    th = 0.5 * np.arctan2((1 - w) * s2 + w * np.sin(2 * a), (1 - w) * c2 + w * np.cos(2 * a))
    return np.stack([np.cos(th), np.sin(th)], -1), True


def _trace(p0, field, ok, axial, step, max_len):
    """Streamline through p0 (cells): both ways for axial fields, forward for directed ones."""
    H, W = ok.shape

    def walk(d, limit):
        pts, p = [], np.array(p0, np.float64)
        for _ in range(int(limit / step)):
            q = p + d * step
            ix, iy = int(q[0]), int(q[1])
            if not (0 <= ix < W and 0 <= iy < H) or not ok[iy, ix]:
                break
            nd = field[iy, ix]
            if np.dot(nd, d) < 0:
                nd = -nd
            d = 0.6 * d + 0.4 * nd                       # the hand turns smoothly, never on a dime
            d = d / (np.linalg.norm(d) + 1e-9)
            p = q
            pts.append(p.copy())
        return pts

    d0 = field[int(p0[1]), int(p0[0])].astype(np.float64)
    if not axial:
        return np.array([p0] + walk(d0, max_len))
    fw = walk(d0, max_len / 2)
    bw = walk(-d0, max_len - len(fw) * step)
    if len(fw) * step < max_len / 2 - step:              # one side ran out of room: give it to the other
        fw = walk(d0, max_len - len(bw) * step)
    return np.array(bw[::-1] + [np.array(p0, np.float64)] + fw)


def _taps(wr, m, oy, ox, ppm, radius, rng, quick=False):
    """Dots / tiny bits: one short circular tap per blob, lifted in between."""
    lab, n = ndi.label(m)
    if not n:
        return
    idx = range(1, n + 1)
    cents = ndi.center_of_mass(m, lab, idx)
    rads = ndi.maximum(ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1], lab, idx)
    todo = list(range(n))
    while todo:
        px, py = wr.pos if wr.pos is not None else (ox + cents[todo[0]][1], oy + cents[todo[0]][0])
        i = min(todo, key=lambda j: (ox + cents[j][1] - px) ** 2 + (oy + cents[j][0] - py) ** 2)
        todo.remove(i)
        cy, cx = cents[i][0] + oy, cents[i][1] + ox
        r = float(max(1.5, min(radius, rads[i] * 0.95)))
        loop = max(0.0, rads[i] - r) * 0.8
        k = max(4, int(rng.uniform(0.05, 0.09 if quick else 0.16) * RATE))
        a = np.linspace(0, 2 * np.pi * rng.uniform(0.8, 1.6), k) + rng.uniform(0, 2 * np.pi)
        xs, ys = cx + loop * np.cos(a), cy + loop * np.sin(a)
        u = np.linspace(0, 1, k)
        pres = np.clip(np.minimum(u / 0.2, 1) * np.minimum((1 - u) / 0.35, 1), 0.15, 1)
        wr.travel((float(xs[0]), float(ys[0])), ppm, r, 0.04 if quick else None)
        wr.stroke(xs, ys, pres, r)


def _fill(wr, m, oy, ox, ppm, radius, rng, kind="base", weight=None, direction=None, fine=False, exact=False, avoid=None):
    """Cover m (crop at oy, ox) with replayed gestures. base/pattern: until no paper is left; shade and
    gradient: quick tapered flicks from the heavy end until every spot that wants ink (weight > 0.45) had one.
    exact: the mask is not a whole region (stripes, shade zones); avoid (same crop): paper of the region this
    layer must leave clean (light stripes) - the brush keeps a full radius away from it, but still runs
    right up to the outline (the ink stops at the line by itself)."""
    i_start = len(wr.cols["x"])
    h, w = m.shape
    cell = max(1.0, min(ppm * CELL_MM, radius * 0.5))
    gh, gw = max(1, int(round(h / cell))), max(1, int(round(w / cell)))
    sx, sy = w / gw, h / gh
    mg = cv2.resize(m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.5
    if mg.sum() < 2:
        return _taps(wr, m, oy, ox, ppm, radius, rng)
    r_c = radius / cell
    ok = ndi.binary_erosion(mg, iterations=int(r_c * 0.3)) if r_c > 3 else mg
    if avoid is not None and avoid.any():
        ag = cv2.resize(avoid.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.3
        ok = ok & ~ndi.binary_dilation(ag, iterations=int(r_c * 1.05) + 1)
    if ok.sum() < 2:
        ok = mg
    wg = None if weight is None else cv2.resize(np.asarray(weight, np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    flick = kind in ("shade", "gradient")
    # weights are absolute ink amounts (a shade may peak at ~0.5): every spot with a real share gets a flick
    need = mg.copy() if wg is None else mg & (wg > max(0.12, 0.4 * float(wg[mg].max(initial=0))))
    if not need.any():
        need = mg & (wg >= wg[mg].max() * 0.8)
    # spots the brush can't reach from where its centre may go (sharp tips) are left to the gap taps
    reach = ndi.distance_transform_edt(~ok) <= max(1.0, 0.95 * r_c)
    need &= reach
    if not need.any():
        need = ok.copy()
    ys, xs = np.nonzero(mg)
    inner = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(np.float64)))[1][:, 1] if len(xs) > 2 else np.array([1.0, 0.0])
    if kind == "pattern" and direction is not None:
        field, _ = _direction_field(mg, cell / ppm, inner, directed=direction)
        axial = True                                     # stripes: back and forth along them
    elif flick:
        field, axial = _direction_field(mg, cell / ppm, inner, wg if direction is None else None, direction)
    else:
        field, axial = _direction_field(mg, cell / ppm, inner if direction is None else np.asarray(direction, float))
    depth = ndi.distance_transform_edt(np.pad(mg, 1))[1:-1, 1:-1]
    _, (iy_ok, ix_ok) = ndi.distance_transform_edt(~ok, return_indices=True)
    cov = np.zeros((gh, gw), np.uint8)
    thick = max(1, int(round(2 * (1.8 if flick else COVER) * r_c)))   # soft flicks: wider apart
    step = 0.25 * ppm / cell
    lo, hi = FINE_LEN_MM if fine else LEN_MM
    to_page = lambda P: (ox + (P[:, 0] + 0.5) * sx, oy + (P[:, 1] + 0.5) * sy)
    same_p = _lib()["same"] / (_lib()["same"] + _lib()["back"])
    last_dir = None
    for _ in range(int(need.sum() / max(1.0, r_c * r_c) * 4) + 50):
        left = need & (cov == 0)
        if not left.any():
            break
        ys, xs = np.nonzero(left)
        if wr.pos is not None:
            dd = np.hypot(xs - ((wr.pos[0] - ox) / sx - 0.5), ys - ((wr.pos[1] - oy) / sy - 0.5))
        else:
            dd = np.zeros(len(xs))
        pri = depth[ys, xs] * DEPTH_W if wg is None else (1 - wg[ys, xs]) * (EDGE_BAND_MM * ppm / cell)
        j = int(np.argmin(dd + pri))
        sx0, sy0 = int(ix_ok[ys[j], xs[j]]), int(iy_ok[ys[j], xs[j]])
        Lmm = rng.uniform(lo, hi) * (0.8 if flick else 1.0)
        P = _trace((sx0 + 0.5, sy0 + 0.5), field, ok, axial, step, Lmm * ppm / cell)
        if len(P) > 2:                                   # trim run-on over covered paper (keep 1.5 mm)
            ii = np.clip(P.astype(int), 0, [gw - 1, gh - 1])
            near = ndi.binary_dilation(left, iterations=max(1, int(r_c * COVER)))
            useful = near[ii[:, 1], ii[:, 0]]
            if useful.any():
                pad = int(1.5 * ppm / cell / step)
                a = max(0, int(np.argmax(useful)) - pad)
                b = min(len(P), len(useful) - int(np.argmax(useful[::-1])) + pad)
                P = P[a:b]
        if len(P) < 2:
            P = np.array([[sx0 + 0.5, sy0 + 0.5], [sx0 + 0.7, sy0 + 0.7]])
        if axial:                                        # back-and-forth (nearest end first) or same way again
            px, py = to_page(P[[0, -1]])
            if last_dir is not None and rng.random() < same_p:
                if np.dot(P[-1] - P[0], last_dir) < 0:
                    P = P[::-1]
            elif wr.pos is not None and np.hypot(px[1] - wr.pos[0], py[1] - wr.pos[1]) < np.hypot(px[0] - wr.pos[0], py[0] - wr.pos[1]):
                P = P[::-1]
        last_dir = P[-1] - P[0]
        cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))]) * cell / ppm
        L = float(cum[-1])
        if L < 0.3:
            n = max(4, int(0.06 * RATE))
            s_, c_, p_ = np.linspace(0, 1, n), np.zeros(n), np.full(n, 0.8)
        else:
            s_, c_, p_ = _gesture(rng, L, small=L < 4.0, fast=2.0 if flick else 1.0)
        gx, gy = np.interp(s_ * L, cum, P[:, 0]), np.interp(s_ * L, cum, P[:, 1])
        tan = np.gradient(np.stack([gx, gy], 1), axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
        bow = np.clip(c_ * L * ppm / cell, -0.9 * r_c, 0.9 * r_c)    # the recorded sideways bend, bounded
        Q = np.stack([gx - tan[:, 1] * bow, gy + tan[:, 0] * bow], 1)
        X, Y = to_page(Q)
        if wg is not None:                               # less ink where the layer wants less
            qi = np.clip(Q.astype(int), 0, [gw - 1, gh - 1])
            p_ = p_ * np.clip(wg[qi[:, 1], qi[:, 0]] * 1.15, 0, 1)
        else:
            p_ = 0.6 + 0.4 * p_                          # flat fills: soft ends, never starved
        start = (float(X[0]), float(Y[0]))
        if (wr.down and wr.pos is not None and rng.random() < CONTINUE_P
                and np.hypot(start[0] - wr.pos[0], start[1] - wr.pos[1]) < 2.5 * radius):
            wr.turn(start, radius)
        else:
            wr.travel(start, ppm, radius)
        wr.stroke(X, Y, p_, radius)
        cv2.polylines(cov, [np.round(Q).astype(np.int32).reshape(-1, 1, 2)], False, 1, thick)
        cv2.circle(cov, (sx0, sy0), max(1, int(r_c * COVER)), 1, -1)
        cov[ys[j], xs[j]] = 1                            # the target always counts as visited: progress
    if not flick:
        _patch_gaps(wr, i_start, m, oy, ox, ppm, radius, rng, exact, avoid)


def _patch_gaps(wr, i_start, m, oy, ox, ppm, radius, rng, exact, avoid=None):
    """Full-resolution check: any paper of this layer the brush never reached gets a tap."""
    c = wr.cols
    if len(c["x"]) <= i_start:
        return
    X, Y, D = (np.concatenate(c[k][i_start:]) for k in ("x", "y", "down"))
    inked = np.zeros(m.shape, np.uint8)
    rr = max(1, int(round(radius)))                      # the dab's solid core (it feathers out beyond)
    for p in np.stack([X[D] - ox, Y[D] - oy], 1)[::2]:
        cv2.circle(inked, (int(round(p[0])), int(round(p[1]))), rr, 1, -1)
    gap = m & (inked == 0)
    if avoid is not None and avoid.any():                # paper left beside a light stripe is meant to be there
        gap &= ~ndi.binary_dilation(avoid, iterations=max(1, int(radius * 1.2)))
    elif exact:
        gap = ndi.binary_erosion(gap, iterations=max(1, int(radius * 0.3)))
    elif radius > 3:
        gap = ndi.binary_opening(gap)
    lab, n = ndi.label(gap)
    if n:                                                # slivers under ~0.3 mm2 vanish under the line / the feather
        sizes = ndi.sum(gap, lab, range(1, n + 1))
        gap = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.3 * ppm * ppm))
    if gap.any():
        _taps(wr, gap, oy, ox, ppm, min(radius, FINE_MM * ppm * 2), rng, quick=True)   # touch-ups


class _Base:
    """Old plans: one base layer = the whole region."""
    kind, weight, direction, tip = "base", None, None, "brush"

    def __init__(self, marker, y0, x0, mask):
        self.marker, self.y0, self.x0, self.mask = marker, y0, x0, mask


def _pick_layers(L, ppm, area_mm2):
    """Session length: skip shade/gradient on small shapes, at most MAX_LAYERS painted layers per region
    (sparkle/glint taps are quick and always kept)."""
    out, n, cap = [], 0, MAX_LAYERS + (area_mm2 > BIG_MM2)
    for lay in L:
        if lay.kind in ("shade", "gradient") and np.asarray(lay.mask).sum() < SHADE_MIN_MM2 * ppm * ppm:
            continue
        if lay.kind != "highlight":
            if n >= cap:
                continue
            n += 1
        out.append(lay)
    return out


def plan(page, color_plan, markers, seed=1, layers=None):
    """Order regions like a colorist (one marker family at a time, light to dark, nearest next) and color
    each: its layers in order (base, gradient, shade, pattern, highlight), each with its own technique."""
    rng = np.random.default_rng(seed)
    ppm = page.px_per_mm
    lightness = srgb_to_lab([m["ink"] for m in markers])[:, 0]
    by_marker = {}
    for rid, v in color_plan.items():
        if rid in page.regions:
            by_marker.setdefault(int(v["marker"]) if isinstance(v, dict) else int(v), []).append(rid)
    wr = _Writer(rng)
    starts, ends = {}, {}
    for mi in sorted(by_marker, key=lambda mi: -lightness[mi]):
        todo = list(by_marker[mi])
        while todo:
            if wr.pos is None:
                rid = todo[0]
            else:
                px, py = wr.pos
                rid = min(todo, key=lambda r: (page.regions[r].centroid[1] - px) ** 2 + (page.regions[r].centroid[0] - py) ** 2)
            todo.remove(rid)
            if layers and layers.get(rid):
                L = _pick_layers(layers[rid], ppm, page.regions[rid].area / ppm ** 2)
            else:
                m, (oy, ox) = region_mask(page, rid, pad=0)
                L = [_Base(mi, oy, ox, m)]
            starts[rid] = wr.time
            wr.region = rid
            area = page.regions[rid].area
            for lay in L:
                m = np.asarray(lay.mask, bool)
                if not m.any():
                    continue
                if lay.marker != wr.marker:
                    if wr.marker != -1 and wr.pos is not None:
                        wr.hold(rng.uniform(*MARKER_SWAP_S), BRUSH_MM * ppm)
                    wr.marker = lay.marker
                    wr.down = False
                insc = float(ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max())
                exact = m.sum() < 0.97 * area
                if lay.tip == "fine" or lay.kind == "highlight":
                    radius = max(1.5, min(FINE_MM * ppm, insc * 0.9))
                elif exact:                                      # stripes / around stripes: a narrower tip, no spill
                    radius = float(np.clip(insc * 0.9, FINE_MM * ppm, BRUSH_MM * ppm))
                else:
                    radius = BRUSH_MM * ppm if insc > BRUSH_MM * ppm * 1.6 else max(2.0, min(FINE_MM * ppm, insc * 0.9))
                fine = lay.kind == "highlight" or (radius < 0.75 * BRUSH_MM * ppm and not exact)   # short strokes
                if lay.direction is None and lay.kind in ("pattern", "highlight"):
                    _taps(wr, m, lay.y0, lay.x0, ppm, radius, rng)       # dots, sparkles
                else:
                    avoid = None
                    if exact and lay.kind in ("base", "pattern"):
                        h_, w_ = m.shape
                        avoid = (page.labels[lay.y0:lay.y0 + h_, lay.x0:lay.x0 + w_] == rid) & ~m
                    _fill(wr, m, lay.y0, lay.x0, ppm, radius, rng, lay.kind, lay.weight, lay.direction, fine, exact, avoid)
            ends[rid] = wr.time
    c = wr.build()
    return Timeline(c["t"], c["x"], c["y"], c["down"], c["marker"], c["radius"], c["stroke"], c["region"],
                    starts, ends, c["pressure"])
