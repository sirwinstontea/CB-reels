"""Hand reference photos, recoloured per shot: the video model copies the marker it sees in the
reference, so the reference's marker cap/rear is tinted to the shot's cap colour.

A photo may have a sidecar <name>.json: {"marker_axis": [[nib x, y], [end x, y]],
"cap_from": 0.55, "half_width": 20, "crop_bottom": 405} (px). Without one, the photo is used as it is.
crop_bottom cuts the photo at the wrist: a sleeve in the reference made the video model paint a big
sweater cuff and forearm over the page; in the real reels only fingers and the back of the hand show.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def tinted(ref_path, cap_rgb, out_dir):
    ref_path = Path(ref_path)
    meta_p = ref_path.with_suffix(".json")
    if not meta_p.exists():
        return ref_path
    meta = json.loads(meta_p.read_text())
    out = Path(out_dir) / f"{ref_path.stem}.c{meta.get('crop_bottom', 0)}.ink{''.join('%02x' % int(v) for v in cap_rgb)}.png"
    if out.exists():
        return out
    img = cv2.imread(str(ref_path))
    h, w = img.shape[:2]
    a, b = np.array(meta["marker_axis"], np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = b - a
    t = ((xx - a[0]) * d[0] + (yy - a[1]) * d[1]) / float(d @ d)
    dist = np.abs((xx - a[0]) * d[1] - (yy - a[1]) * d[0]) / float(np.linalg.norm(d))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255
    cap = np.array(cap_rgb[::-1], np.float32)          # BGR
    ref_l = float(cap.mean()) / 255 or 1.0
    res = img.astype(np.float32)
    hw = meta.get("half_width", 20)
    # rear/cap: grey plastic, shaded and dimmed for light inks
    cap_sel = (t >= meta.get("cap_from", 0.55)) & (t <= 1.03) & (dist <= hw) & (hsv[..., 1] < 40) & (hsv[..., 2] < 235)
    # nib: the dark felt tip; the model copies its colour onto the page, so it gets the ink's own brightness
    nib_sel = (t >= -0.03) & (t <= meta.get("nib_to", 0.045)) & (dist <= hw * 0.6) & (hsv[..., 2] < 150)
    for sel, lo, hi, gain in ((cap_sel, 0.35, None, min(1.0, ref_l * 1.4)), (nib_sel, 0.1, (0.6, 1.3), 1.0)):
        if not sel.any():
            continue
        m = cv2.GaussianBlur(sel.astype(np.float32), (0, 0), 1.2)[..., None]
        shade = lum / max(lo, float(np.median(lum[sel])))
        if hi:
            shade = np.clip(shade, *hi)
        col = np.clip(cap * shade * gain, 0, 255)
        res = res * (1 - m) + col * m
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if meta.get("crop_bottom"):
        res = res[:int(meta["crop_bottom"])]
    cv2.imwrite(str(out), np.clip(res, 0, 255).astype(np.uint8))
    return out
