"""The stand-in: a shaded, hand-shaped right hand holding a fat alcohol marker, drawn exactly where the
ink is going down.

The video model swaps it for a real hand and keeps its structure, so the stand-in is drawn the way the
hand should end up: fingers and the back of the hand only (the wrist runs off toward pen_dir, no sleeve),
index finger on top of the barrel ~25 mm above the nib, thumb on the side, real proportions, soft light
from the same side as the shadow, rounded shading and nails. The old flat beige ovals let the model
re-invent the hand every frame (size and grip drifted, judged 6/10). What must stay right: the nib sits
on the stroke point, the cap and nib match the ink, the barrel points the same way in every shot of one
angle, and a lifted pen floats with its shadow separating.
Sizes are in page mm (projected), drawn in output px.
"""
import cv2
import numpy as np

SKIN = np.array([214, 168, 140], np.float32) / 255
NAIL = np.array([232, 196, 186], np.float32) / 255
BODY = np.array([238, 238, 234], np.float32) / 255
GRIP = np.array([214, 214, 210], np.float32) / 255
FORESHORTEN = 0.6        # the barrel tilts up out of the paper, so it looks this much shorter
SHADOW_DIR = np.array([-0.7, 0.7])
LIGHT = -SHADOW_DIR / np.linalg.norm(SHADOW_DIR)     # toward the light, in output px


def _ell(a, b, ra, rb, rot=0.0, n=28):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    x, y = ra * np.cos(t), rb * np.sin(t)
    return np.stack([a + c * x - s * y, b + s * x + c * y], 1)


def _capsule(a1, b1, a2, b2, r1, r2=None, n=10):
    """Finger: a stadium from (a1, b1) radius r1 to (a2, b2) radius r2, as a polygon in (a, b) mm."""
    r2 = r1 if r2 is None else r2
    d = np.array([a2 - a1, b2 - b1], np.float64)
    L = np.linalg.norm(d) or 1.0
    d /= L
    nrm = np.array([-d[1], d[0]])
    ang = np.arctan2(d[1], d[0])
    t1 = np.linspace(ang + np.pi / 2, ang + 3 * np.pi / 2, n)
    t2 = np.linspace(ang - np.pi / 2, ang + np.pi / 2, n)
    cap1 = np.stack([a1 + r1 * np.cos(t1), b1 + r1 * np.sin(t1)], 1)
    cap2 = np.stack([a2 + r2 * np.cos(t2), b2 + r2 * np.sin(t2)], 1)
    return np.vstack([cap2, cap1])


def _quad(a0, a1, h0, h1=None, b=0.0):
    h1 = h0 if h1 is None else h1
    return np.array([[a0, b - h0], [a1, b - h1], [a1, b + h1], [a0, b + h0]], np.float64)


# (part, polygon in (a along the barrel from the nib, b across) mm, colour, round): back to front.
# A right hand, seen from above; b < 0 is the side the back of the hand is on.
def _parts():
    return [
        ("wrist", _quad(85, 230, 30, 34, b=-20), SKIN * 0.95, 0.9),
        ("pinky", _capsule(70, -48, 48, -40, 7.0, 6.0), SKIN * 0.86, 1.0),
        ("ring", _capsule(78, -38, 44, -28, 8.0, 7.0), SKIN * 0.88, 1.0),
        ("middle", _capsule(80, -24, 30, -8, 8.5, 7.5), SKIN * 0.9, 1.0),
        ("back", _ell(88, -22, 34, 36, 8), SKIN, 1.0),
        ("barrel", _quad(14, 150, 7.0, 8.0), BODY, 0.55),
        ("cap", _quad(108, 152, 8.6), "cap", 0.55),
        ("grip", _quad(8, 16, 5.2, 7.0), GRIP, 0.4),
        ("nib", np.array([[0, 0], [3, -1.6], [8.5, -4.6], [8.5, 4.6], [3, 1.6]], np.float64), "ink", 0.2),
        ("thumb", _capsule(70, 26, 30, 9, 10.5, 8.0), SKIN * 0.98, 1.0),
        ("thumbnail", _ell(30.5, 10.5, 5.6, 4.6), NAIL, 0.5),
        ("index", _capsule(*INDEX[0], *INDEX[1], INDEX[2], INDEX[3]), SKIN * 1.03, 1.0),
        ("indexnail", _ell(INDEX[1][0] + 1.5, INDEX[1][1], 6.2, 4.8), NAIL, 0.5),
    ]


INDEX = ((84.0, -12.0), (26.0, -1.5), 9.0, 7.6)     # knuckle (a, b), tip (a, b), radius there, radius at the tip


def _creases():
    """Index finger joints as short lines across the finger, (a, b) end points: the model sees knuckles."""
    k, t = np.array(INDEX[0]), np.array(INDEX[1])
    d = (k - t) / np.linalg.norm(k - t)
    n = np.array([-d[1], d[0]])
    return [(t + f * (k - t) - 0.7 * INDEX[3] * n, t + f * (k - t) + 0.7 * INDEX[3] * n) for f in (0.3, 0.58)]


CREASES = _creases()


class StandIn:
    def __init__(self, pen_dir):
        self.u0 = np.asarray(pen_dir, np.float64) / np.linalg.norm(pen_dir)

    def _frame(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        u = np.array([c * self.u0[0] - s * self.u0[1], s * self.u0[0] + c * self.u0[1]])
        v = np.array([-u[1], u[0]])
        return u, v

    def _to_px(self, ab, tip, k, theta):
        u, v = self._frame(theta)
        ab = np.asarray(ab, np.float64)
        return tip + (ab[..., :1] * FORESHORTEN * u + ab[..., 1:2] * v) * k

    def draw(self, img, tip, k, ink_rgb, cap_rgb, lift=0.0, theta=0.0):
        """Draw onto img (float RGB, modified in place). tip in output px, k = output px per page mm.
        Returns the silhouette (uint8 mask, full frame)."""
        H, W = img.shape[:2]
        lift_px = lift * 5 * k
        tip_up = np.array(tip, np.float64) + np.array([0, -lift_px])
        parts = [(n, self._to_px(p, tip_up, k, theta), c, r) for n, p, c, r in _parts()]
        off = SHADOW_DIR * (3 + lift * 7) * k
        blur = 2 + 2.5 * k * (1 + lift)
        pts = np.vstack([p for _, p, _, _ in parts])
        pts = np.vstack([pts, pts + off])
        pad = int(3 * blur) + 2
        x0, y0 = np.floor(pts.min(0)).astype(int) - pad
        x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        sil = np.zeros((H, W), np.uint8)
        if x0 >= x1 or y0 >= y1:
            return sil
        o = np.array([x0, y0], np.float64)
        sub = img[y0:y1, x0:x1]
        h, w = sub.shape[:2]
        s = np.zeros((h, w), np.uint8)
        for _, p, _, _ in parts:
            self._fill(s, p - o, 255)
        sil[y0:y1, x0:x1] = s
        # contact shadow: the silhouette, offset and softened
        M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        sh = _soft(cv2.warpAffine(s, M, (w, h)).astype(np.float32) / 255, blur)
        sub *= (1 - 0.38 * sh)[..., None]
        named = {"ink": np.asarray(ink_rgb, np.float32) / 255, "cap": np.asarray(cap_rgb, np.float32) / 255}
        for name, p, col, rnd in parts:
            col = named[col] if isinstance(col, str) else col
            q = p - o
            # work inside this part's own box (the full hand box is up to the whole frame: 10x slower)
            bx0, by0 = np.maximum(0, np.floor(q.min(0)).astype(int) - 2)
            bx1, by1 = np.minimum([w, h], np.ceil(q.max(0)).astype(int) + 2)
            if bx0 >= bx1 or by0 >= by1:
                continue
            q = q - (bx0, by0)
            bh, bw = by1 - by0, bx1 - bx0
            m8 = np.zeros((bh, bw), np.uint8)
            self._fill(m8, q, 255)
            m = m8.astype(np.float32) / 255
            # rounded look: darker toward the edges (a blurred mask), lit from the light side
            size = max(2.0, float(np.sqrt(max(1.0, cv2.contourArea(q.astype(np.float32))))))
            inner = np.clip(_soft(np.pad(m, 8), 0.22 * size)[8:-8, 8:-8] * 1.8, 0, 1)
            cx, cy = (float(v) for v in q.mean(0))
            lx, ly = (float(v) / (0.5 * size) for v in LIGHT)       # python floats keep the maths float32
            g = np.clip((np.arange(bw, dtype=np.float32) - cx) * lx + (np.arange(bh, dtype=np.float32)[:, None] - cy) * ly, -1, 1)
            shade = (1 - 0.24 * rnd) + 0.24 * rnd * inner + 0.08 * rnd * g
            bs = sub[by0:by1, bx0:bx1]
            mm = m[..., None]
            bs *= 1 - mm
            bs += np.minimum(col.astype(np.float32) * shade[..., None], 1) * mm
        lw = max(1, int(round(0.5 * k)))
        for (a0, b0), (a1, b1) in CREASES:          # finger joints: faint lines, so the model sees knuckles
            p0, p1 = self._to_px(np.array([[a0, b0], [a1, b1]]), tip_up, k, theta) - o
            bx0, by0 = np.maximum(0, np.floor(np.minimum(p0, p1)).astype(int) - 3 * lw - 2)
            bx1, by1 = np.minimum([w, h], np.ceil(np.maximum(p0, p1)).astype(int) + 3 * lw + 2)
            if bx0 >= bx1 or by0 >= by1:
                continue
            ln = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            cv2.line(ln, tuple(int(round(v)) for v in p0 - (bx0, by0)), tuple(int(round(v)) for v in p1 - (bx0, by0)),
                     255, lw, cv2.LINE_AA)
            ln = cv2.GaussianBlur(ln.astype(np.float32) / 255, (0, 0), max(0.5, 0.4 * k)) * (s[by0:by1, bx0:bx1] > 0)
            sub[by0:by1, bx0:bx1] *= (1 - 0.18 * ln)[..., None]
        return sil

    @staticmethod
    def _fill(m, g, val):
        s = 4                                   # sub-pixel precision for smooth edges
        cv2.fillPoly(m, [np.round(g * (1 << s)).astype(np.int32)], val, cv2.LINE_AA, s)


def _soft(m, sigma):
    """Gaussian blur, done at 1/4 size when wide (it runs ~15 times per frame)."""
    if sigma < 6:
        return cv2.GaussianBlur(m, (0, 0), max(0.5, sigma))
    h, w = m.shape
    q = cv2.resize(m, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(q, (0, 0), sigma / 4), (w, h), interpolation=cv2.INTER_LINEAR)
