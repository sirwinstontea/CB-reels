"""Choose a marker for every region.

Two sources of color:
- a colored target image of the same page (an AI recolor or your own colored file):
  each region takes the median color of its middle, then snaps to the nearest marker;
- a fallback palette when no target exists (used for engine tests and as the color gate's fallback).
"""
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .page import region_mask


def load_markers(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["markers"]


def srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def snap(rgb, markers):
    """Nearest marker by Delta E (CIE76, good enough for picking from ~36 markers)."""
    lab = srgb_to_lab(rgb)
    mlab = srgb_to_lab([m["ink"] for m in markers])
    d = np.linalg.norm(mlab - lab, axis=-1)
    d[[bool(m.get("opaque")) for m in markers]] = np.inf     # the gel pen is for highlights only
    i = int(np.argmin(d))
    return i, float(d[i])


def plan_from_target(page, target_path, markers, min_align=0.85):
    """Color gate: the target must still line up with the page's lines before we trust it."""
    tgt = np.asarray(Image.open(target_path).convert("RGB").resize(page.shape[::-1], Image.LANCZOS))
    tgt_lines = tgt.mean(-1) < 110
    page_lines = page.lines < 0.5
    near = ndi.binary_dilation(tgt_lines, iterations=3)
    align = float((page_lines & near).sum() / max(1, page_lines.sum()))
    if align < min_align:
        return None, {"passed": False, "align": round(align, 3), "reason": "target image doesn't line up with the page"}
    plan = {}
    for rid in page.regions:
        m = page.labels == rid
        core = ndi.binary_erosion(m, iterations=4)
        px = tgt[core if core.sum() > 20 else m]
        plan[rid] = snap(np.median(px, axis=0), markers)[0]
    return plan, {"passed": True, "align": round(align, 3), "reason": ""}


FALLBACK_AUTUMN = ["sun-03", "pumpkin-05", "rust-06", "tomato-08", "leaf-20", "moss-21", "olive-23",
                   "sand-24", "tan-25", "caramel-26", "walnut-27", "sky-15", "cream-01", "brick-07", "gold-35", "peach-30"]


def plan_fallback(page, markers, seed=7, names=FALLBACK_AUTUMN):
    """Seasonal palette: neighbouring regions avoid sharing a marker so shapes stay readable."""
    rng = np.random.default_rng(seed)
    idx = {m["id"]: i for i, m in enumerate(markers)}
    pool = [idx[n] for n in names if n in idx]
    plan = {}
    grown = ndi.grey_dilation(page.labels, size=(7, 7))
    for rid in sorted(page.regions, key=lambda r: -page.regions[r].area):
        m, (oy, ox) = region_mask(page, rid, pad=6)
        ring = ndi.binary_dilation(m, iterations=4) & ~m
        neigh = set(np.unique(grown[oy:oy + m.shape[0], ox:ox + m.shape[1]][ring]).tolist()) - {0, rid}
        used = {plan[n] for n in neigh if n in plan}
        choices = [p for p in pool if p not in used] or pool
        plan[rid] = int(rng.choice(choices))
    return plan
