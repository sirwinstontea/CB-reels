"""Read a line-art coloring page and split it into fillable regions.

Output is a Page: the line art (0 = ink line, 1 = paper), a label image where each
fillable shape has its own id, and per-region facts the planner needs.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PAGE_MM = 216.0          # 8.5 in square page; px/mm is derived from the image width
LINE_THRESHOLD = 128     # darker than this is a line
MIN_REGION_MM2 = 0.5     # smaller shapes are texture: merged into a neighbour or left white


@dataclass
class Region:
    id: int
    area: int                    # pixels
    bbox: tuple                  # (y0, y1, x0, x1), exclusive ends
    centroid: tuple              # (y, x)


@dataclass
class Page:
    path: str
    lines: np.ndarray            # float32, 1 = paper, 0 = black line (anti-aliased)
    labels: np.ndarray           # int32, 0 = line / ignored, >0 = region id
    regions: dict = field(default_factory=dict)
    px_per_mm: float = 0.0
    gap_close_px: int = 0
    report: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.lines.shape


def load_page(path, gap_close_px=2):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    px_per_mm = w / PAGE_MM
    line_mask = gray < (LINE_THRESHOLD / 255.0)
    closed = ndi.binary_dilation(line_mask, iterations=gap_close_px) if gap_close_px else line_mask

    labels, n = ndi.label(~closed)
    ids = np.arange(1, n + 1)
    sizes = ndi.sum(np.ones_like(labels, dtype=np.int32), labels, ids).astype(np.int64)

    # The paper around the drawing is whatever touches the border: not colored.
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border > 0]

    min_px = MIN_REGION_MM2 * px_per_mm * px_per_mm
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_px
    keep[border] = False
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    labels = remap[labels]

    # Grow each region back under the gap-closing dilation so ink reaches the real line.
    if gap_close_px:
        grown = ndi.grey_dilation(labels, size=(2 * gap_close_px + 1,) * 2)
        free = (labels == 0) & ~line_mask
        labels = np.where(free & (grown > 0), grown, labels).astype(np.int32)

    page = Page(path=str(path), lines=gray, labels=labels, px_per_mm=px_per_mm, gap_close_px=gap_close_px)
    objs = ndi.find_objects(labels)
    ids = np.arange(1, labels.max() + 1)
    areas = ndi.sum(np.ones_like(labels), labels, ids)
    cents = ndi.center_of_mass(np.ones_like(labels), labels, ids)
    for i, sl, a, c in zip(ids, objs, areas, cents):
        if sl is None:
            continue
        page.regions[int(i)] = Region(int(i), int(a), (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop), c)

    page.report = page_gate(page)
    return page


def page_gate(page):
    """Numbers a person would have eyeballed. A huge region usually means a leak."""
    total = page.shape[0] * page.shape[1]
    areas = sorted((r.area for r in page.regions.values()), reverse=True)
    largest = areas[0] / total if areas else 0.0
    n = len(areas)
    ok = 20 <= n <= 1500 and largest < 0.35
    return {"regions": n, "largest_pct": round(largest * 100, 1), "passed": bool(ok),
            "reason": "" if ok else ("too few/many regions" if not 20 <= n <= 1500 else "one region is suspiciously large (leak?)")}


def load_page_gated(path, max_gap=5):
    """Page gate with automatic retry: close gaps harder until the page splits sensibly."""
    last = None
    for gap in range(2, max_gap + 1):
        page = load_page(path, gap)
        last = page
        if page.report["passed"]:
            return page
    return last


def region_mask(page, rid, pad=0):
    """Boolean mask of one region cropped to its bbox (+pad), plus the crop offset."""
    y0, y1, x0, x1 = page.regions[rid].bbox
    h, w = page.shape
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(h, y1 + pad), min(w, x1 + pad)
    return page.labels[y0:y1, x0:x1] == rid, (y0, x0)


def preview(page, out_path, scale=0.25, seed=1):
    rng = np.random.default_rng(seed)
    pal = rng.integers(70, 255, size=(page.labels.max() + 1, 3), dtype=np.uint8)
    pal[0] = 255
    vis = pal[page.labels].astype(np.float32) * page.lines[..., None]
    img = Image.fromarray(vis.astype(np.uint8))
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img.save(out_path, quality=80)
