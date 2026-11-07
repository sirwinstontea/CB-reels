"""The desk: a photo of a desk with a blank sheet on it, one per camera angle.

The colored page is laid onto the photo's blank sheet as *transmission* (page / paper color),
multiplied into the photo, so the photo's own light, shadows and paper texture stay on the page.
Desk pictures live in assets/desks/<angle>.jpg; corners are found automatically (or read from
<angle>.json when detection would be unreliable). Shots are 9:16 crops of a desk picture.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ink import PAPER, TOOTH_PAPER, paper_tile

OUT_W, OUT_H = 720, 1280


@dataclass
class Desk:
    name: str
    img: np.ndarray          # float RGB 0..1, HxWx3
    corners: np.ndarray      # blank sheet TL, TR, BR, BL in desk px
    pen_dir: tuple           # unit vector (desk px) the marker barrel points along, from the tip
    px_per_mm: float         # desk px per mm of sheet, near the sheet center
    sheet_mm: tuple = (216.0, 279.0)   # real width, height of the blank sheet
    page_fit: float = 0.94   # the printed page covers this share of the sheet's limiting side


def _order(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def find_sheet(img8):
    """Corners of the biggest bright, low-saturation quadrilateral (the blank sheet)."""
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > 170) & (hsv[..., 1] < 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.08 * m.size:
        return None
    hull = cv2.convexHull(c)
    for eps in (0.01, 0.02, 0.03, 0.05):
        q = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(q) == 4:
            return _order(q)
    return None


def sheet_aspect(corners, img_wh):
    """True width/height of the rectangle seen as `corners` (Zhang & He, principal point at the
    image center), or None when the view is too close to straight-on to tell."""
    tl, tr, br, bl = [np.array([*p, 1.0]) for p in corners]
    k2 = np.dot(np.cross(tl, br), bl) / np.dot(np.cross(tr, br), bl)
    k3 = np.dot(np.cross(tl, br), tr) / np.dot(np.cross(bl, br), tr)
    n2, n3 = k2 * tr - tl, k3 * bl - tl
    u0, v0 = img_wh[0] / 2, img_wh[1] / 2
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2)
           + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2)) / (n2[2] * n3[2])
    if not np.isfinite(f2) or f2 <= 0:
        return None
    Ai = np.linalg.inv(np.array([[np.sqrt(f2), 0, u0], [0, np.sqrt(f2), v0], [0, 0, 1]]))
    M = Ai.T @ Ai
    r = float(np.sqrt((n2 @ M @ n2) / (n3 @ M @ n3)))
    return r if 0.4 < r < 2.5 else None


def load_desk(path):
    path = Path(path)
    img8 = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    meta_p = path.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    corners = _order(meta["corners"]) if "corners" in meta else find_sheet(img8)
    if corners is None:
        raise ValueError(f"{path.name}: can't find the blank sheet; add {meta_p.name} with \"corners\"")
    pen_dir = np.asarray(meta.get("pen_dir", (0.55, 0.83)), np.float32)
    pen_dir /= np.linalg.norm(pen_dir)
    if "sheet_mm" in meta:
        sheet_mm = tuple(float(v) for v in meta["sheet_mm"])
    else:      # a letter sheet, turned the way the photo shows it
        r = sheet_aspect(corners, img8.shape[1::-1])
        sheet_mm = (279.0, 216.0) if r and r > 1 else (216.0, 279.0)
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    return Desk(path.stem, img8.astype(np.float32) / 255.0, corners, tuple(pen_dir),
                (top + bottom) / 2 / sheet_mm[0], sheet_mm, float(meta.get("page_fit", 0.94)))


def page_to_desk(desk, page_shape):
    """Page px -> desk px: the page is printed centered on the sheet, aspect kept."""
    h, w = page_shape
    sw, sh = desk.sheet_mm
    s = desk.page_fit * min(sw / w, sh / h)             # mm per page px
    x0, y0 = (sw - w * s) / 2, (sh - h * s) / 2
    to_mm = np.array([[s, 0, x0], [0, s, y0], [0, 0, 1]], np.float64)
    mm = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    return cv2.getPerspectiveTransform(mm, desk.corners).astype(np.float64) @ to_mm


def page_quad(desk, page_shape):
    """The printed page's corners TL, TR, BR, BL in desk px."""
    h, w = page_shape
    src = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    return cv2.perspectiveTransform(src, page_to_desk(desk, page_shape))[0]


@dataclass
class View:
    """One camera framing: a 9:16 window on a desk picture, scaled to the output size."""
    desk: Desk
    cx: float
    cy: float
    width: float             # in desk px

    def S(self):
        s = OUT_W / self.width
        h = self.width * OUT_H / OUT_W
        return np.array([[s, 0, -(self.cx - self.width / 2) * s],
                         [0, s, -(self.cy - h / 2) * s],
                         [0, 0, 1]], np.float64)

    def clamp(self):
        H, W = self.desk.img.shape[:2]
        self.width = min(self.width, W, H * OUT_W / OUT_H)
        h = self.width * OUT_H / OUT_W
        self.cx = float(np.clip(self.cx, self.width / 2, W - self.width / 2))
        self.cy = float(np.clip(self.cy, h / 2, H - h / 2))
        return self


_TILE = None


def _sheet_grain(view, S, H_po, page_shape):
    """Paper tooth on the blank sheet around the printed page, at output px. The ink canvas draws
    the same tile inside the page; the desk photo alone is too soft for it in close shots."""
    global _TILE
    if _TILE is None:
        _TILE = paper_tile()
    s = float(np.linalg.norm(H_po[:2, 0] / H_po[2, 2]))           # ~output px per page px
    tile = cv2.GaussianBlur(_TILE, (0, 0), 0.5 / s) if s < 0.9 else _TILE
    g = cv2.warpPerspective(tile, H_po, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    sheet = np.zeros((OUT_H, OUT_W), np.float32)
    cv2.fillConvexPoly(sheet, cv2.perspectiveTransform(view.desk.corners[None].astype(np.float64), S)[0]
                       .round().astype(np.int32), 1.0, lineType=cv2.LINE_AA)
    h, w = page_shape
    page = np.zeros_like(sheet)
    cv2.fillConvexPoly(page, cv2.perspectiveTransform(np.float64([[[0, 0], [w, 0], [w, h], [0, h]]]), H_po)[0]
                       .round().astype(np.int32), 1.0)
    return 1 + TOOTH_PAPER * g * np.clip(sheet - page, 0, 1)


class Renderer:
    """Renders frames of one view: desk crop x page transmission (ink + lines)."""

    def __init__(self, view, page_shape):
        self.view = view.clamp()
        S = view.S()
        self.H_po = S @ page_to_desk(view.desk, page_shape)          # page px -> output px
        self.H_op = np.linalg.inv(self.H_po)
        self.bg = cv2.warpPerspective(view.desk.img, S, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        self.bg *= _sheet_grain(view, S, self.H_po, page_shape)[..., None]
        # part of the page this view can see, in page px (so only that part is rendered)
        oc = np.float32([[[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]]])
        pc = cv2.perspectiveTransform(oc, self.H_op)[0]
        ph, pw = page_shape
        x0, y0 = np.floor(pc.min(0)).astype(int) - 2
        x1, y1 = np.ceil(pc.max(0)).astype(int) + 2
        self.page_shape = page_shape
        self.f = 1
        box = (max(0, y0), min(ph, y1), max(0, x0), min(pw, x1))
        self.visible = box[0] < box[1] and box[2] < box[3]
        if self.visible:
            mid = ((box[2] + box[3]) / 2, (box[0] + box[1]) / 2)
            self.f = f = max(1, int(1 / (1.5 * self.scale_at(*mid))))   # render far shots at lower res
            box = (box[0] // f * f, min(ph, -(-box[1] // f) * f), box[2] // f * f, min(pw, -(-box[3] // f) * f))
        self.box = box
        # cache pixel (i, j) = page pixel block starting at box origin + f * (i, j)
        f = self.f
        D = np.array([[f, 0, self.box[2] + (f - 1) / 2], [0, f, self.box[0] + (f - 1) / 2], [0, 0, 1]], np.float64)
        self.H_co = self.H_po @ D                                    # cache px -> output px
        self.cache = None

    def to_out(self, x, y):
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H_po)[0, 0]
        return float(p[0]), float(p[1])

    def warp_mask(self, page_mask):
        '''Bool page mask -> bool output-frame mask (same warp as the ink).'''
        if not self.visible:
            return np.zeros((OUT_H, OUT_W), bool)
        by0, by1, bx0, bx1 = self.box
        f = self.f
        m = page_mask[by0:by1, bx0:bx1].astype(np.float32)
        if f > 1:
            h, w = m.shape
            if h % f == 0 and w % f == 0:
                m = m.reshape(h // f, f, w // f, f).max(axis=(1, 3))
            else:
                m = cv2.resize(m, (-(-w // f), -(-h // f)), interpolation=cv2.INTER_AREA)
        out = cv2.warpPerspective(m, self.H_co, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0.35

    def scale_at(self, x, y):
        """Output px per page px around a page point."""
        a = np.array(self.to_out(x, y))
        b = np.array(self.to_out(x + 10, y))
        return float(np.linalg.norm(b - a) / 10)

    def _patch(self, canvas, t, box):
        """Transmission of a page box, block-averaged by f (box is f-aligned to self.box)."""
        trans = canvas.render(t, box) / PAPER
        f = self.f
        if f == 1:
            return trans
        h, w = trans.shape[:2]
        if h % f == 0 and w % f == 0:
            return trans.reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
        return cv2.resize(trans, (-(-w // f), -(-h // f)), interpolation=cv2.INTER_AREA)

    def frame(self, canvas, t=None):
        if not self.visible:
            return self.bg.copy()
        t = canvas.time if t is None else t
        by0, by1, bx0, bx1 = self.box
        f = self.f
        if self.cache is None:
            self.cache = self._patch(canvas, t, self.box).astype(np.float32)
        else:
            rb = canvas.recent_box(t)            # only what was just inked needs a cache refresh
            if rb is not None:
                y0, y1 = max(by0, rb[0]), min(by1, rb[1])
                x0, x1 = max(bx0, rb[2]), min(bx1, rb[3])
                if y0 < y1 and x0 < x1:
                    y0, x0 = by0 + (y0 - by0) // f * f, bx0 + (x0 - bx0) // f * f
                    y1, x1 = min(by1, by0 + -(-(y1 - by0) // f) * f), min(bx1, bx0 + -(-(x1 - bx0) // f) * f)
                    p = self._patch(canvas, t, (y0, y1, x0, x1))
                    cy, cx = (y0 - by0) // f, (x0 - bx0) // f
                    self.cache[cy:cy + p.shape[0], cx:cx + p.shape[1]] = p
        trans = cv2.warpPerspective(self.cache, self.H_co, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
        return self.bg * trans


def make_placeholder_desk(path, angle="A", seed=0):
    """A stand-in desk photo (wood, soft light, blank sheet) until the real ones are generated."""
    rng = np.random.default_rng(seed)
    W, H = 2160, 3840
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    n = cv2.resize(rng.normal(0, 1, (H // 64, 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    rings = np.sin(x / 37.0 + n * 2.2 + np.sin(y / 400.0) * 1.5)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=12)
    base = np.array([0.52, 0.36, 0.22], np.float32)
    wood = base * (1 + 0.10 * rings[..., None] + 0.05 * fine[..., None])
    if angle == "A":
        corners = np.float32([[250, 900], [1890, 860], [1960, 2560], [190, 2610]])
        light = (1500, 700)
    else:
        corners = np.float32([[420, 1050], [1940, 1180], [1760, 2780], [140, 2560]])
        light = (300, 600)
    sheet = np.zeros((H, W), np.float32)
    cv2.fillConvexPoly(sheet, corners.astype(np.int32), 1.0, lineType=cv2.LINE_AA)
    shadow = cv2.GaussianBlur(np.roll(sheet, (14, -10), (0, 1)), (0, 0), 18)
    img = wood * (1 - 0.35 * shadow[..., None])
    paper = PAPER * (1 + 0.01 * fine[..., None] / 3)
    img = img * (1 - sheet[..., None]) + paper * sheet[..., None]
    d = np.hypot(x - light[0], y - light[1]) / np.hypot(W, H)
    img *= (1.08 - 0.30 * d ** 1.5)[..., None]
    img8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    pen = (0.55, 0.83) if angle == "A" else (0.35, 0.94)
    path.with_suffix(".json").write_text(json.dumps({"pen_dir": pen, "placeholder": True}))
    return path
