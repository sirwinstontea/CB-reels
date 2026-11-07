"""Depth for a color plan: the layers a colorist lays in each shape, in drawing order.

Measured on the reference reels (720x1280, page ~2.6 out px/mm):
- every shape gets a darker marker of the same family along its edges, strongest on the side away
  from the light: L drops 6-18 at the outline and fades over ~30% of the shape's inscribed radius
  (2.5-10 mm), with no visible boundary; the shade is also more saturated than the base;
- objects sitting on a background throw a soft darker rim onto it on their shadow side;
- big shapes (sky, wall, water) carry a 2-3 marker gradient across their length;
- patterns: two-tone stripes (period ~15 mm, dark stripe ~30%, ~1.5 mm soft edge), dots;
- a white gel pen adds glints on the lit side and scattered sparkle dots last.

layers_for(page, plan, markers) -> {region id: [Layer, ...]}
plan: {rid: marker index} (old plans: base + automatic edge shade), or a Plan whose .style holds
per-region extras (see colorai), or {rid: {"marker": i, ...style}}.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .colors import srgb_to_lab

LIGHT = np.array([-0.6, -0.8])   # unit vector toward the light: from the upper left
SHADE_MIN_MM = 1.3               # thinner shapes (inscribed radius) get one flat layer
SHADE_MIN_AREA_MM2 = 20.0
RIM = 0.40                       # all-round edge darkening, relative to the shadow side
CAST = 0.65                      # strength of the shadow an object throws onto the shape around it


@dataclass
class Layer:
    marker: int            # index into markers
    kind: str              # "base" | "shade" | "gradient" | "pattern" | "highlight"
    y0: int
    x0: int
    mask: np.ndarray       # bool crop (bbox of this layer): True = this layer's ink goes here
    weight: np.ndarray | None = None   # float32 0..1 desired ink amount, None = full
    direction: tuple | None = None     # (dx, dy) preferred stroke direction, None = free
    tip: str = "brush"     # "brush" | "fine"


class Plan(dict):
    """{rid: base marker} like the old plans (strokes reads this), plus .style {rid: extras}."""

    def __init__(self, base=(), style=None, light=None):
        super().__init__(base)
        self.style = dict(style or {})
        self.light = light


LIGHTS = {"upper-left": (-0.6, -0.8), "upper-right": (0.6, -0.8), "top": (0.0, -1.0),
          "left": (-1.0, -0.15), "right": (1.0, -0.15)}
DIRS = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}


def _idx(markers):
    return {m["id"]: i for i, m in enumerate(markers)}


def shade_of(mi, markers):
    """The darker marker of the same family (markers[i]["shade"]), or None."""
    s = markers[mi].get("shade")
    return _idx(markers).get(s) if s else None


def white_gel(markers):
    return next((i for i, m in enumerate(markers) if m.get("opaque")), None)


def _ss(e0, e1, t):
    t = np.clip((t - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _crop(kind, marker, m, oy, ox, weight=None, direction=None, tip="brush", min_w=0.06):
    """Tight layer from a region-crop mask/weight."""
    if weight is not None:
        m = m & (weight > min_w)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    w = None if weight is None else np.where(m[y0:y1, x0:x1], weight[y0:y1, x0:x1], 0).astype(np.float32)
    return Layer(int(marker), kind, int(oy + y0), int(ox + x0), m[y0:y1, x0:x1].copy(), w,
                 None if direction is None else (float(direction[0]), float(direction[1])), tip)


def _norm_blur(v, m, s):
    """Gaussian blur of v inside m only (no bleeding across the outline)."""
    a = ndi.gaussian_filter(np.where(m, v, 0).astype(np.float32), s)
    b = ndi.gaussian_filter(m.astype(np.float32), s)
    return np.where(m, a / np.maximum(b, 1e-4), 0)


def shade_weight(m, ppm, light=LIGHT):
    """0..1 shade amount: deep on the edges facing away from the light, a soft rim all round, and
    the shadow that objects inside the shape (its holes) throw onto it."""
    d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
    insc = float(d.max())
    w = float(np.clip(0.30 * insc, 1.2 * ppm, 7.0 * ppm))
    g = ndi.gaussian_filter(m.astype(np.float32), 1.5)
    gy, gx = np.gradient(g)
    n = np.hypot(gx, gy) + 1e-6
    facing = (gx * light[0] + gy * light[1]) / n           # >0: this edge faces away from the light
    # (g points inward, so g . light > 0 means the outward normal points away from the light)
    edge = m & ~ndi.binary_erosion(m)
    filled = ndi.binary_fill_holes(m)
    inner = edge & ndi.binary_dilation(filled & ~m, iterations=2)
    s_form = np.where(edge & ~inner, np.clip(facing, 0, 1) ** 0.8, 0)
    s_cast = np.where(inner, CAST * np.clip(-facing, 0, 1), 0)      # the object is toward the light
    src = np.maximum(s_form, s_cast)
    # smooth the strength along the outline so neighbouring sources agree (no Voronoi seams)
    src = np.where(edge, _norm_blur(src, edge, max(2.0, 0.6 * w)), 0)
    strong = edge & (src > 0.05)
    out = RIM * np.exp(-d / (0.55 * w))
    if strong.any():
        dist, (iy, ix) = ndi.distance_transform_edt(~strong, return_indices=True)
        out = np.maximum(out, src[iy, ix] * np.exp(-dist / w))
    out = _norm_blur(out, m, max(1.5, 0.35 * w))
    return np.clip(out, 0, 1).astype(np.float32), insc


def _coords(m):
    ys, xs = np.nonzero(m)
    return ys, xs


def gradient_weights(m, direction, n):
    """n weights for a gradient of n markers toward `direction` (down/up/left/right/in/out)."""
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
    if direction in ("in", "out"):
        d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        t = d / max(1.0, d.max())
        t = t if direction == "in" else 1 - t
    else:
        dx, dy = DIRS.get(direction, (0, 1))
        p = xx * dx + yy * dy
        lo, hi = np.percentile(p[m], [2, 98])
        t = (p - lo) / max(1.0, hi - lo)
    if n == 1:
        return [_ss(0.25, 0.95, t)]
    return [_ss(0.12, 0.60, t), _ss(0.55, 1.0, t)][:n]


def _stripes(m, ppm, direction):
    ys, xs = _coords(m)
    area = len(ys) / ppm ** 2
    period = float(np.clip(0.18 * np.sqrt(area), 3.0, 14.0)) * ppm
    v = {"horizontal": (0, 1), "diagonal": (0.707, 0.707)}.get(direction, (1, 0))  # across the stripes
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
    ph = ((xx * v[0] + yy * v[1]) / period + 0.15) % 1.0
    soft = 0.6 * ppm / period                      # ~0.6 mm soft stripe edge
    wgt = _ss(0, soft, ph) * (1 - _ss(0.36 - soft, 0.36, ph))
    along = (-v[1], v[0])
    return m & (wgt > 0.05), wgt, along


def _dots(m, ppm, spacing_mm=None, r_frac=0.22, rng=None, jitter=0.0):
    ys, xs = _coords(m)
    area = len(ys) / ppm ** 2
    sp = (spacing_mm or float(np.clip(0.12 * np.sqrt(area), 2.5, 7.0))) * ppm
    r = r_frac * sp
    inside = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1] > r + 0.4 * ppm
    out = np.zeros_like(m)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    for row, cy in enumerate(np.arange(sp / 2, m.shape[0], sp * 0.866)):
        for cx in np.arange(sp / 2 + (sp / 2 if row % 2 else 0), m.shape[1], sp):
            if rng is not None and jitter:
                cy2, cx2 = cy + rng.uniform(-jitter, jitter) * sp, cx + rng.uniform(-jitter, jitter) * sp
            else:
                cy2, cx2 = cy, cx
            iy, ix = int(cy2), int(cx2)
            if 0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] and inside[iy, ix]:
                rr = r * (rng.uniform(0.6, 1.2) if rng is not None and jitter else 1.0)
                y0, y1, x0, x1 = max(0, iy - int(rr) - 2), iy + int(rr) + 3, max(0, ix - int(rr) - 2), ix + int(rr) + 3
                out[y0:y1, x0:x1] |= (yy[y0:y1, x0:x1] - cy2) ** 2 + (xx[y0:y1, x0:x1] - cx2) ** 2 <= rr * rr
    return out & m, r


def _glint(m, ppm, light):
    """A short white arc on the lit side plus a dot beside it, like a gel-pen shine."""
    d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
    insc = d.max()
    if insc < 1.8 * ppm:
        return None
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
    cand = d > max(0.42 * insc, 1.0 * ppm)
    proj = np.where(cand, xx * light[0] + yy * light[1], -1e9)
    cy, cx = np.unravel_index(int(np.argmax(proj)), proj.shape)
    tang = np.array([-light[1], light[0]])
    L = float(min(0.55 * insc, 4.5 * ppm))
    s = np.linspace(-0.5, 0.5, 24)[:, None]
    pts = np.array([cx, cy]) + s * L * tang - (s ** 2) * L * 0.5 * np.array(light)   # bows around the form
    width = max(1.2, 0.30 * ppm)
    out = np.zeros_like(m)
    for px, py in pts:
        out |= (xx - px) ** 2 + (yy - py) ** 2 <= width ** 2
    dot = np.array([cx, cy]) + tang * (0.5 * L + 1.0 * ppm) - np.array(light) * 0.4 * ppm
    out |= (xx - dot[0]) ** 2 + (yy - dot[1]) ** 2 <= (1.25 * width) ** 2
    return out & m, tuple(tang)


def layers_for(page, plan, markers, auto_shade=True, seed=5):
    """{region id: [Layer, ...]} in drawing order: base, gradients, shade, pattern, highlights."""
    rng = np.random.default_rng(seed)
    ppm = page.px_per_mm
    idx = _idx(markers)
    style_all = getattr(plan, "style", {}) or {}
    lt = np.array(LIGHTS.get(getattr(plan, "light", None) or "upper-left"), np.float64)
    lt /= np.linalg.norm(lt)
    gel = white_gel(markers)
    lum = srgb_to_lab([m["ink"] for m in markers])[:, 0]
    out = {}
    for rid, v in plan.items():
        if rid not in page.regions:
            continue
        st = dict(v) if isinstance(v, dict) else dict(style_all.get(rid, style_all.get(str(rid), {})))
        base = int(v["marker"]) if isinstance(v, dict) else int(v)
        y0, y1, x0, x1 = page.regions[rid].bbox
        m = page.labels[y0:y1, x0:x1] == rid
        L = [Layer(base, "base", y0, x0, m, None, None, "brush")]
        out[rid] = L
        insc = float(ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max())
        big = insc >= SHADE_MIN_MM * ppm and m.sum() >= SHADE_MIN_AREA_MM2 * ppm ** 2

        def mk(x):
            if x is None or x == "none":
                return None
            return x if isinstance(x, (int, np.integer)) else idx.get(x)

        grads = [g for g in (mk(x) for x in (st.get("gradient") or [])) if g is not None][:2]
        if grads and big:
            gd = st.get("gradient_dir", "down")
            for g, wg in zip(grads, gradient_weights(m, gd, len(grads))):
                # a much darker end marker only glazes over the base, so the shape never turns muddy
                cap = float(np.clip(1.0 - (lum[base] - lum[g] - 15) / 50, 0.35, 0.85))
                lay = _crop("gradient", g, m, y0, x0, (wg * cap).astype(np.float32),
                            DIRS.get(gd) if gd in DIRS else None)
                if lay:
                    L.append(lay)
        sh = st.get("shade", "auto")
        sh = shade_of(base, markers) if sh in (None, "auto") and (auto_shade or rid in style_all) else mk(sh)
        if sh is not None and sh != base and big:
            wgt, _ = shade_weight(m, ppm, lt)
            if grads and big:                      # a gradient already darkens it: don't stack to black
                wgt *= 0.6
            lay = _crop("shade", sh, m, y0, x0, wgt)
            if lay:
                L.append(lay)
        pat = st.get("pattern", "none")
        pm = mk(st.get("pattern_marker")) if st.get("pattern_marker") not in (None, "auto") else None
        pm = pm if pm is not None else (shade_of(base, markers) or base)
        lighter = lum[pm] > lum[base] + 5      # markers can't go lighter: color around it instead
        pmask = pw = None
        if pat == "stripes" and big:
            pmask, wgt, along = _stripes(m, ppm, st.get("pattern_dir", "vertical"))
            pw, tip = (None if lighter else (0.65 * wgt).astype(np.float32)), "brush"
        elif pat == "dots" and big:
            pmask, r = _dots(m, ppm)
            along, tip = None, "fine" if r < 1.2 * ppm else "brush"
        if pmask is not None:
            if lighter:
                pmask = pmask & (wgt > 0.5) if pat == "stripes" else pmask
                # the base goes around the light stripes, its strokes running along them
                L[0] = Layer(base, "base", y0, x0, m & ~pmask, None, along)
                lay = _crop("pattern", pm, pmask, y0, x0, None, along, tip)
                if lay:
                    L.insert(0, lay)
            else:
                lay = _crop("pattern", pm, pmask, y0, x0, pw, along, tip)
                if lay:
                    L.append(lay)
        elif pat == "sparkle" and gel is not None and big:
            dm, _ = _dots(m, ppm, spacing_mm=float(np.clip(insc / ppm * 0.6, 4.0, 9.0)), r_frac=0.07,
                          rng=rng, jitter=0.35)
            lay = _crop("highlight", gel, dm, y0, x0, tip="fine")
            if lay:
                L.append(lay)
        if st.get("highlight") and gel is not None:
            gl = _glint(m, ppm, lt)
            if gl is not None:
                lay = _crop("highlight", gel, gl[0], y0, x0, direction=gl[1], tip="fine")
                if lay:
                    L.append(lay)
    return out
