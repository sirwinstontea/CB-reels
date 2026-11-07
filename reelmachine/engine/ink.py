"""Flat ink fill from the stroke timeline.

Strokes lay a simple color fill (soft tip, clipped to the region). Wet-leveling, tooth
mottling, and alcohol look are NOT simulated — Higgsfield applies marker texture later,
mask-locked to where ink was deposited (`inked`).

Opaque gel covers. Ink only grows. `paper_tile` / TOOTH_PAPER remain for scene.py sheet grain.
"""
import numpy as np
from scipy import ndimage as ndi

TIP_SOFT = 0.45
UNDER_LINE_PX = 2
FILL_ALPHA = 0.98
GEL_OPACITY = 0.93
GEL = np.array([0.985, 0.985, 0.975], np.float32)
PAPER = np.array([250, 248, 243], np.float32) / 255.0
TOOTH_PAPER = 0.010
TILE = 1024


def _unit(a):
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


def paper_tile(seed=11, n=TILE):
    """Seamless paper texture in page px for the desk sheet margin (scene.py)."""
    rng = np.random.default_rng(seed)
    blur = lambda s: _unit(ndi.gaussian_filter(rng.normal(0, 1, (n, n)).astype(np.float32), s, mode="wrap"))
    tooth, felt = blur(0.9), blur(3.5)
    fib = np.zeros((n, n), np.float32)
    for _ in range(n * n // 900):
        y, x = rng.uniform(0, n, 2)
        a, L = rng.uniform(0, np.pi), rng.uniform(6, 19)
        s = np.linspace(-L / 2, L / 2, int(L * 1.5))
        fib[(y + s * np.sin(a)).astype(int) % n, (x + s * np.cos(a)).astype(int) % n] += rng.choice([-1.0, 1.0])
    fib = _unit(ndi.gaussian_filter(fib, 0.7, mode="wrap"))
    return _unit(0.6 * tooth + 0.6 * felt + 0.12 * fib)


def _tiled(tile, shape):
    h, w = shape
    n = tile.shape[0]
    return np.tile(tile, (-(-h // n), -(-w // n)))[:h, :w]


class InkCanvas:
    def __init__(self, page, markers, seed=3):
        h, w = page.shape
        self.page = page
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)
        self.wm = np.full((h, w), -1, np.int16)
        self.gel = None
        self.ink_rgb = np.array([np.asarray(m["ink"], np.float32) / 255.0 for m in markers], np.float32)
        self.opaque = np.array([bool(m.get("opaque")) for m in markers], bool)
        self.medium = [str(m.get("medium") or "alcohol") for m in markers]
        lab = page.labels
        grown = lab.copy()
        for _ in range(UNDER_LINE_PX):
            d = ndi.grey_dilation(grown, size=(3, 3))
            grown = np.where(grown == 0, d, grown)
        self.ink_label = grown
        self.time = 0.0
        self.last_t = 0.0
        self.cov_box = None
        self.cur_stroke = -2
        self.stroke_log = []
        self._i = 0

    def inked(self):
        """Bool page mask where flat ink (or gel) was laid — texture lock."""
        m = self.a > 0.02
        if self.gel is not None:
            m |= self.gel > 0.02
        return m

    def recent_box(self, t, window=2.0):
        boxes = [self.cov_box] if self.cov_box is not None else []
        for te, b in reversed(self.stroke_log):
            if te < t - window:
                break
            boxes.append(b)
        if not boxes:
            return None
        a = np.array(boxes)
        return int(a[:, 0].min()), int(a[:, 1].max()), int(a[:, 2].min()), int(a[:, 3].max())

    def _clear_cov(self):
        if self.cov_box is not None:
            self.stroke_log.append((self.last_t, self.cov_box))
        self.cov_box = None

    def dab(self, t, x, y, r, marker, region, stroke, prev=None, pressure=1.0):
        if stroke != self.cur_stroke:
            self._clear_cov()
            self.cur_stroke = stroke
        pressure = min(1.0, float(pressure))
        if pressure <= 0.01:
            return
        h, w = self.a.shape
        R = int(np.ceil(r + 1.5))
        px, py = prev if prev is not None else (x, y)
        y0, y1 = max(0, int(min(y, py)) - R), min(h, int(max(y, py)) + R + 2)
        x0, x1 = max(0, int(min(x, px)) - R), min(w, int(max(x, px)) + R + 2)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = x - px, y - py
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 < 1e-6 else np.clip(((xx - px) * dx + (yy - py) * dy) / L2, 0, 1)
        d = np.hypot(xx - (px + u * dx), yy - (py + u * dy))
        soft = max(1.0, TIP_SOFT * r)
        k = np.clip((r + 0.5 - d) / soft, 0, 1).astype(np.float32)
        k = k * k * (3 - 2 * k)
        k *= self.ink_label[y0:y1, x0:x1] == region
        if not k.any():
            return
        sl = np.s_[y0:y1, x0:x1]
        self.last_t = t
        b = self.cov_box
        self.cov_box = (y0, y1, x0, x1) if b is None else (min(b[0], y0), max(b[1], y1), min(b[2], x0), max(b[3], x1))
        if self.opaque[marker]:
            if self.gel is None:
                self.gel = np.zeros((h, w), np.float16)
            g = self.gel[sl]
            np.maximum(g, (k * (GEL_OPACITY * pressure)).astype(np.float16), out=g)
            return
        cov = (k * pressure * FILL_ALPHA).astype(np.float32)
        a = self.a[sl]
        grow = cov > a
        if grow.any():
            col = self.ink_rgb[marker]
            wgt = np.clip((cov - a) / np.maximum(cov, 1e-4), 0, 1)
            rgb = self.rgb[sl]
            for c in range(3):
                rgb[..., c] = np.where(grow, rgb[..., c] * (1 - wgt) + col[c] * wgt, rgb[..., c])
            self.wm[sl] = np.where(grow, marker, self.wm[sl])
            np.maximum(a, cov, out=a)

    def run(self, tl, until):
        i0 = getattr(self, "_i", 0)
        i1 = int(np.searchsorted(tl.t, until, side="right"))
        pr = getattr(tl, "pressure", None)
        if pr is not None and len(pr) != len(tl.t):
            pr = None
        for i in range(i0, i1):
            if tl.down[i]:
                same = i > 0 and tl.down[i - 1] and tl.stroke[i - 1] == tl.stroke[i]
                prev = (float(tl.x[i - 1]), float(tl.y[i - 1])) if same else None
                self.dab(float(tl.t[i]), float(tl.x[i]), float(tl.y[i]), float(tl.radius[i]),
                         int(tl.marker[i]), int(tl.region[i]), int(tl.stroke[i]), prev,
                         1.0 if pr is None else float(pr[i]))
        self._i = i1
        self.time = until

    def render(self, t=None, box=None):
        sl = np.s_[:, :] if box is None else np.s_[box[0]:box[1], box[2]:box[3]]
        lines = self.page.lines[sl][..., None]
        rgb = np.broadcast_to(PAPER, (*self.a[sl].shape, 3)).astype(np.float32).copy()
        a = self.a[sl][..., None]
        rgb = rgb * (1 - a) + self.rgb[sl] * a
        rgb *= lines
        # hard burn: never lighten outline pixels
        dark = self.page.lines[sl] < 0.2
        rgb[dark] = 0
        if self.gel is not None:
            g = self.gel[sl].astype(np.float32)
            if g.any():
                rgb = rgb * (1 - g[..., None]) + GEL * g[..., None]
                rgb *= lines
        return np.clip(rgb, 0, 1)
