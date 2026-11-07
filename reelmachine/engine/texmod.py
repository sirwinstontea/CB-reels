"""Apply Omni local texture onto flat plate ink without trusting Omni's page lock.

For each frame: ratio = omni / blur(omni); out = plate outside mask,
plate * ratio inside mask. Keeps our strokes/colors; borrows micro-contrast.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .scene import OUT_W, OUT_H
from .video import Writer, read_frames, probe, FPS
from .spot import Q, _align, _count


def apply(result_path, plate_path, inkmask_path, out_path, blur_sigma=3.0):
    n_plate = _count(plate_path)
    masks = np.load(inkmask_path)["masks"]
    n = min(n_plate, _count(result_path, FPS), len(masks))
    g = max(1, 8 // Q)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), k) for m in masks[:n]]) > 0

    # align on outside of mid frame
    mid = n // 2
    a_mid = p_mid = None
    for i, fr in enumerate(read_frames(result_path, OUT_W, OUT_H, fps=FPS)):
        if i == mid:
            a_mid = fr
            break
    for i, fr in enumerate(read_frames(plate_path)):
        if i == mid:
            p_mid = fr
            break
    outside = cv2.resize((~ink[mid]).astype(np.uint8), (OUT_W // 2, OUT_H // 2), interpolation=cv2.INTER_NEAREST)
    M = _align(cv2.resize(a_mid, (OUT_W // 2, OUT_H // 2)),
               cv2.resize(p_mid, (OUT_W // 2, OUT_H // 2)), outside)
    M[:, 2] *= 2
    shift = float(np.hypot(*M[:, 2]) / OUT_W + np.abs(M[:, :2] - np.eye(2)).max())
    warp = (lambda im: cv2.warpAffine(im, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)) if shift > 0.002 else (lambda im: im)

    tmp = Path(str(out_path) + ".part.mp4")
    wr = Writer(tmp, OUT_W, OUT_H)
    outside_de = []
    ga, gp = read_frames(result_path, OUT_W, OUT_H, fps=FPS), read_frames(plate_path)
    for i, a, p in zip(range(n), ga, gp):
        a = warp(a).astype(np.float32)
        p = p.astype(np.float32)
        base = cv2.GaussianBlur(a, (0, 0), blur_sigma)
        ratio = a / np.maximum(base, 1.0)
        ratio = np.clip(ratio, 0.75, 1.35)
        textured = np.clip(p * ratio, 0, 255)
        w = cv2.resize(ink[i].astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        w = cv2.GaussianBlur(np.clip(w, 0, 1), (0, 0), 1.0)[..., None]
        out = textured * w + p * (1 - w)
        # outside must equal plate
        de = np.mean(np.abs(out - p), axis=2)
        o = ~ink[i]
        o_full = cv2.resize(o.astype(np.uint8), (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST) > 0
        outside_de.append(float(de[o_full].mean()) if o_full.any() else 0.0)
        wr.write(out.astype(np.uint8))
    ga.close(); gp.close()
    wr.close()
    os.replace(tmp, out_path)
    return {
        "ok": float(np.mean(outside_de)) < 1.0,
        "outside_mae": round(float(np.mean(outside_de)), 4),
        "align_shift": round(shift, 4),
        "repaired": str(out_path),
        "src": probe(result_path),
        "frames": n,
    }
