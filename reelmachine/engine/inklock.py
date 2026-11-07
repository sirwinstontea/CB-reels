"""Align + colour-fit Omni texture results, then keep model pixels only inside the ink mask."""
import os
from pathlib import Path

import cv2
import numpy as np

from .scene import OUT_W, OUT_H
from .video import Writer, read_frames, probe, FPS
from .spot import DE_CHANGED, LIMITS, Q, _align, _apply_color, _color_fit, _count, _lab


def lock_ink(result_path, plate_path, inkmask_path, out_path=None):
    info = probe(result_path)
    n_plate = _count(plate_path)
    masks = np.load(inkmask_path)["masks"] if inkmask_path and Path(inkmask_path).exists() else None
    if masks is None:
        return {"ok": False, "fails": ["missing inkmask"], "result": str(result_path)}
    n = min(n_plate, _count(result_path, FPS), len(masks))
    if n < 3:
        return {"ok": False, "fails": ["clip too short"], "result": str(result_path), "src": info}
    g = max(1, 8 // Q)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), k) for m in masks[:n]]) > 0
    small = lambda im: cv2.resize(im, (OUT_W // Q, OUT_H // Q), interpolation=cv2.INTER_AREA)

    want = sorted({0, n // 2, max(0, n - 1)})
    keep_a, keep_p = {}, {}
    for i, fr in enumerate(read_frames(result_path, OUT_W, OUT_H, fps=FPS)):
        if i in want:
            keep_a[i] = fr
        if i >= max(want):
            break
    for i, fr in enumerate(read_frames(plate_path)):
        if i in want:
            keep_p[i] = fr
        if i >= max(want):
            break
    want = [i for i in want if i in keep_a and i in keep_p]
    f0 = want[0]
    outside0 = cv2.resize((~ink[f0]).astype(np.uint8), (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    M = _align(cv2.resize(keep_a[f0], (OUT_W // 2, OUT_H // 2)),
               cv2.resize(keep_p[f0], (OUT_W // 2, OUT_H // 2)),
               cv2.resize(outside0, (OUT_W // 2, OUT_H // 2), interpolation=cv2.INTER_NEAREST))
    M[:, 2] *= 2
    shift = float(np.hypot(*M[:, 2]) / OUT_W + np.abs(M[:, :2] - np.eye(2)).max())
    warp = (lambda im: cv2.warpAffine(im, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)) if shift > 0.002 else (lambda im: im)
    samp_a, samp_p = [], []
    for i in want:
        o = ~ink[i]
        samp_a.append(small(warp(keep_a[i]))[o])
        samp_p.append(small(keep_p[i])[o])
    gains = _color_fit(np.concatenate(samp_a), np.concatenate(samp_p))

    fails, changed = [], []
    tmp = Path(str(out_path) + ".part.mp4") if out_path else None
    wr = Writer(tmp, OUT_W, OUT_H) if tmp else None
    ga, gp = read_frames(result_path, OUT_W, OUT_H, fps=FPS), read_frames(plate_path)
    for i, a, p in zip(range(n), ga, gp):
        a = _apply_color(warp(a), gains)
        la, lp = _lab(small(a)), _lab(small(p))
        de = np.linalg.norm(la - lp, axis=2)
        outside = ~ink[i]
        changed.append(float((de[outside] > DE_CHANGED).mean()) if outside.any() else 0.0)
        if wr:
            w = cv2.resize(ink[i].astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            w = cv2.GaussianBlur(np.clip(w, 0, 1), (0, 0), 1.2)[..., None]
            fixed = (a.astype(np.float32) * w + p.astype(np.float32) * (1 - w)).astype(np.uint8)
            wr.write(fixed)
    ga.close(); gp.close()
    if wr:
        wr.close()
    changed = np.array(changed)
    bad = changed > LIMITS["outside_changed"]
    rep = {"result": str(result_path), "src": info, "frames": n,
           "align": {"shift": round(shift, 4)}, "color": gains.round(4).tolist(),
           "outside_changed": {"mean": round(float(changed.mean()), 4), "max": round(float(changed.max()), 4),
                               "bad_frames": int(bad.sum())},
           "fails": fails, "ok": True}
    if bad.mean() > LIMITS["bad_frames"]:
        fails.append(f"page changed outside ink in {bad.sum()} of {n} frames")
        rep["ok"] = False
        rep["fails"] = fails
    if tmp and tmp.exists():
        os.replace(tmp, out_path)
        rep["repaired"] = str(out_path)
    return rep
