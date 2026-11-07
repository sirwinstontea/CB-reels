"""Spot the difference: checks a video-model result against the plate it was made from.

The model may only change what's near the hand. So, frame by frame:
- align the result to the plate (small shifts/zooms are normal) and match its colors,
- outside the hand zone (stand-in silhouette, grown and smeared over a few frames) the result
  must look like the plate: ink can't vanish, appear early, or change color,
- inside the stand-in silhouette there must be *something* (a hand), not bare page, or the ink
  would appear by itself,
- then the repaired clip takes from the model only the hand itself: the stand-in's place plus
  whatever the model's frame changed strongly next to it (a bigger hand, the barrel, its shadow),
  limited to the zone. Everything else is the exact plate. (It used to take the whole grown zone,
  so the model's re-drawn page around the hand showed: ink that jumped or flickered near the nib.)
Metrics run at 1/4 size; the repaired clip is written at full size. Frames are streamed (2 decodes),
never held: the PC has ~1.5 GB free.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .scene import OUT_W, OUT_H
from .video import Writer, read_frames, probe, FPS

Q = 4                     # metrics run at 1/Q size, the same size the stand-in masks are stored at
ZONE_GROW_PX = 64         # full-size px the model's hand may reach beyond the stand-in
ZONE_SMEAR_FRAMES = 4     # and frames it may lag or lead
DE_CHANGED = 12.0         # Lab distance that counts as "this pixel changed"
LIMITS = {
    "outside_changed": 0.015,     # share of outside pixels changed in one frame before it is bad
    "bad_frames": 0.05,           # share of bad frames before the clip fails
    "bad_run": 6,                 # consecutive bad frames before the clip fails
    "hand_missing": 0.15,         # share of hand frames where the hand area looks like bare page
    "duration": 0.10,             # allowed length difference
    "shift": 0.03,                # allowed alignment shift (share of frame width)
}
MATTE_LO, MATTE_HI = 5.0, 14.0    # Lab distance: below = page (plate wins), above = hand (model wins)


def _lab(rgb8):
    return cv2.cvtColor(rgb8.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def _zones(masks):
    """Hand zone per frame (1/Q size): silhouette grown in space and smeared in time."""
    g = max(1, ZONE_GROW_PX // Q)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    grown = np.array([cv2.dilate(m.astype(np.uint8), k) for m in masks]) > 0
    out = grown.copy()
    for d in range(1, ZONE_SMEAR_FRAMES + 1):
        out[d:] |= grown[:-d]
        out[:-d] |= grown[d:]
    return out


def _align(ai, plate, outside):
    """Affine warp mapping the result onto the plate, fitted on still page pixels."""
    a = cv2.cvtColor(ai, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    p = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(p, a, M, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                    outside.astype(np.uint8), 5)
    except cv2.error:
        M = np.eye(2, 3, dtype=np.float32)
    return M


def _color_fit(ai_px, plate_px):
    """Per-channel gain/offset that maps result colors onto plate colors (robust: two passes)."""
    gains = []
    for c in range(3):
        x, y = ai_px[:, c].astype(np.float64), plate_px[:, c].astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        g = np.linalg.lstsq(A, y, rcond=None)[0]
        r = np.abs(A @ g - y)
        keep = r < np.percentile(r, 80)
        g = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        gains.append(g)
    return np.array(gains, np.float32)       # (3, 2)


def _apply_color(rgb8, gains):
    f = rgb8.astype(np.float32)
    f = f * gains[:, 0] + gains[:, 1]
    return np.clip(f, 0, 255).astype(np.uint8)


def hand_matte(ai, plate, sil, zone):
    """Where the repaired frame takes the model's pixels (float 0..1, full size).
    ai/plate: full-size RGB8 (ai aligned + colour-fitted); sil: stand-in silhouette, zone: allowed area."""
    half = (OUT_W // 2, OUT_H // 2)
    de = np.linalg.norm(_lab(cv2.resize(ai, half, interpolation=cv2.INTER_AREA))
                        - _lab(cv2.resize(plate, half, interpolation=cv2.INTER_AREA)), axis=2)
    z2 = cv2.resize(zone.astype(np.uint8), half, interpolation=cv2.INTER_NEAREST) > 0
    s2 = cv2.resize(sil.astype(np.uint8), half, interpolation=cv2.INTER_NEAREST) > 0
    strong = (de > MATTE_HI) & z2
    # the hand = strong changes connected to the stand-in's place (drops stray re-drawn ink and lines)
    _, lab = cv2.connectedComponents(strong.astype(np.uint8), connectivity=8)
    touch = np.unique(lab[s2 & strong])
    core = np.isin(lab, touch[touch > 0]) | s2
    # fill the hand's inside (a white barrel over white paper barely differs from the page)
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    outside = np.pad(core, 1)                     # a free border: the fill always starts on page
    cv2.floodFill(outside, np.zeros((outside.shape[0] + 2, outside.shape[1] + 2), np.uint8), (0, 0), 1)
    core = (core > 0) | (outside[1:-1, 1:-1] == 0)
    # the soft shadow and edges next to it
    near = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    w = np.maximum(core.astype(np.float32), np.clip((de - MATTE_LO) / (MATTE_HI - MATTE_LO), 0, 1) * near)
    w = cv2.GaussianBlur(w * z2, (0, 0), 1.5)
    return cv2.resize(w, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)


def _count(path, fps=None):
    info = probe(path)
    return int(round(info["duration"] * (fps or info["fps"])))


def check(result_path, plate_path, masks_path, out_path=None, restore=True):
    """Returns a report dict. If out_path is given and the clip passes, writes the repaired clip."""
    info = probe(result_path)
    n_plate = _count(plate_path)
    masks = np.load(masks_path)["masks"] if masks_path and Path(masks_path).exists() else np.zeros((n_plate, OUT_H // Q, OUT_W // Q), bool)
    small = lambda im: cv2.resize(im, (OUT_W // Q, OUT_H // Q), interpolation=cv2.INTER_AREA)
    n = min(n_plate, _count(result_path, FPS), len(masks))
    zones = _zones(masks[:max(n, 1)])
    f0 = int(np.argmin(zones.reshape(len(zones), -1).mean(1)))
    want = {f0, n // 2, max(0, n - 1)}
    # pass 1 (streamed): count the frames, keep only the three used for alignment and colour
    keep_a, keep_p, n_ai, n_plate = {}, {}, 0, 0
    for n_ai, fr in enumerate(read_frames(result_path, OUT_W, OUT_H, fps=FPS), 1):
        if n_ai - 1 in want:
            keep_a[n_ai - 1] = fr
    for n_plate, fr in enumerate(read_frames(plate_path), 1):
        if n_plate - 1 in want:
            keep_p[n_plate - 1] = fr
    rep = {"result": str(result_path), "src": info, "frames_plate": n_plate, "frames_result": n_ai}
    fails = []
    if abs(n_ai - n_plate) > LIMITS["duration"] * n_plate:
        fails.append(f"length {n_ai} frames vs {n_plate}")
    n = min(n_ai, n_plate, len(masks), len(zones))
    want = sorted(i for i in want if i < n and i in keep_a and i in keep_p)
    if n < 10 or not want:
        rep.update(ok=False, fails=fails + ["result too short"])
        return rep
    masks, zones = masks[:n], zones[:n]
    f0 = f0 if f0 in want else want[0]

    # 1. alignment, fitted on a frame where the hand is small
    outside_full = cv2.resize((~zones[f0]).astype(np.uint8), (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    M = _align(cv2.resize(keep_a[f0], (OUT_W // 2, OUT_H // 2)), cv2.resize(keep_p[f0], (OUT_W // 2, OUT_H // 2)),
               cv2.resize(outside_full, (OUT_W // 2, OUT_H // 2), interpolation=cv2.INTER_NEAREST))
    M[:, 2] *= 2
    shift = float(np.hypot(*M[:, 2]) / OUT_W + np.abs(M[:, :2] - np.eye(2)).max())
    rep["align"] = {"matrix": M.round(4).tolist(), "shift": round(shift, 4)}
    if shift > LIMITS["shift"]:
        fails.append(f"result is shifted/zoomed {shift:.3f} of the frame")
    warp = (lambda im: cv2.warpAffine(im, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)) if shift > 0.002 else (lambda im: im)

    # 2. color match on still pixels of three frames
    samp_a, samp_p = [], []
    for i in want:
        o = ~zones[i]
        samp_a.append(small(warp(keep_a[i]))[o])
        samp_p.append(small(keep_p[i])[o])
    gains = _color_fit(np.concatenate(samp_a), np.concatenate(samp_p))
    rep["color"] = gains.round(4).tolist()
    keep_a = keep_p = None

    # 3. per-frame checks (pass 2, streamed), writing the repaired clip as it goes; kept only if it passes
    tmp = Path(str(out_path) + ".part.mp4") if out_path else None
    wr = Writer(tmp, OUT_W, OUT_H) if tmp else None
    changed, missing, hand_frames = [], 0, 0
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ga, gp = read_frames(result_path, OUT_W, OUT_H, fps=FPS), read_frames(plate_path)
    for i, a, p in zip(range(n), ga, gp):
        a = _apply_color(warp(a), gains)
        la, lp = _lab(small(a)), _lab(small(p))
        de = np.linalg.norm(la - lp, axis=2)
        o = ~zones[i]
        changed.append(float((de[o] > DE_CHANGED).mean()) if o.any() else 0.0)
        core = cv2.erode(masks[i].astype(np.uint8), erode) > 0
        if core.sum() > 30:
            hand_frames += 1
            if (de[core] < DE_CHANGED / 2).mean() > 0.7:      # looks like bare page where the hand should be
                missing += 1
        if wr:
            if restore:
                w = hand_matte(a, p, masks[i], zones[i])[..., None]
                a = (a.astype(np.float32) * w + p.astype(np.float32) * (1 - w)).astype(np.uint8)
            wr.write(a)
    ga.close(), gp.close()
    if wr:
        wr.close()
    changed = np.array(changed)
    bad = changed > LIMITS["outside_changed"]
    run = best = 0
    for b in bad:
        run = run + 1 if b else 0
        best = max(best, run)
    rep["outside_changed"] = {"mean": round(float(changed.mean()), 4), "max": round(float(changed.max()), 4),
                              "bad_frames": int(bad.sum()), "worst_run": best}
    rep["hand_missing"] = {"frames": missing, "of": hand_frames}
    if bad.mean() > LIMITS["bad_frames"]:
        fails.append(f"page changed outside the hand in {bad.sum()} of {n} frames")
    if best >= LIMITS["bad_run"]:
        fails.append(f"page changed outside the hand for {best} frames in a row")
    if hand_frames and missing / hand_frames > LIMITS["hand_missing"]:
        fails.append(f"hand missing in {missing} of {hand_frames} frames (ink would appear by itself)")
    rep["fails"] = fails
    rep["ok"] = not fails

    # 4. repaired clip: exact plate except the hand (hand_matte), kept only if the clip passed
    if tmp:
        if rep["ok"] and tmp.exists():
            os.replace(tmp, out_path)
            rep["repaired"] = str(out_path)
        elif tmp.exists():
            tmp.unlink()
    return rep



def lock_ink(result_path, plate_path, inkmask_path, out_path=None):
    """Texture-edit referee: outside the ink mask, restore plate pixels.

    Inside the ink mask the model may add marker texture. Black lines / blank paper /
    desk outside ink must match the flat plate. Writes repaired clip when out_path set.
    """
    info = probe(result_path)
    n_plate = _count(plate_path)
    masks = np.load(inkmask_path)["masks"] if inkmask_path and Path(inkmask_path).exists() else None
    if masks is None:
        return {"ok": False, "fails": ["missing inkmask"], "result": str(result_path)}
    n = min(n_plate, _count(result_path, FPS), len(masks))
    if n < 3:
        return {"ok": False, "fails": ["clip too short"], "result": str(result_path), "src": info}
    # grow ink a little so soft stroke edges count as "inside"
    g = max(1, 6 // Q)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), k) for m in masks[:n]]) > 0
    fails, changed = [], []
    tmp = Path(str(out_path) + ".part.mp4") if out_path else None
    wr = Writer(tmp, OUT_W, OUT_H) if tmp else None
    ga, gp = read_frames(result_path, OUT_W, OUT_H, fps=FPS), read_frames(plate_path)
    for i, a, p in zip(range(n), ga, gp):
        la, lp = _lab(cv2.resize(a, (OUT_W // Q, OUT_H // Q), interpolation=cv2.INTER_AREA)), \
                 _lab(cv2.resize(p, (OUT_W // Q, OUT_H // Q), interpolation=cv2.INTER_AREA))
        de = np.linalg.norm(la - lp, axis=2)
        outside = ~ink[i]
        changed.append(float((de[outside] > DE_CHANGED).mean()) if outside.any() else 0.0)
        if wr:
            w = cv2.resize(ink[i].astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)[..., None]
            w = np.clip(w, 0, 1)
            fixed = (a.astype(np.float32) * w + p.astype(np.float32) * (1 - w)).astype(np.uint8)
            wr.write(fixed)
    ga.close(); gp.close()
    if wr:
        wr.close()
    changed = np.array(changed)
    bad = changed > LIMITS["outside_changed"]
    rep = {"result": str(result_path), "src": info, "frames": n,
           "outside_changed": {"mean": round(float(changed.mean()), 4), "max": round(float(changed.max()), 4),
                               "bad_frames": int(bad.sum())},
           "fails": fails, "ok": True}
    if bad.mean() > LIMITS["bad_frames"]:
        fails.append(f"page changed outside ink in {bad.sum()} of {n} frames")
        rep["ok"] = False
        rep["fails"] = fails
    # always keep the mask composite: plate outside ink, model inside (even if spot fails)
    if tmp and tmp.exists():
        os.replace(tmp, out_path)
        rep["repaired"] = str(out_path)
    return rep

if __name__ == "__main__":
    import sys
    r = check(*sys.argv[1:4], out_path=sys.argv[4] if len(sys.argv) > 4 else None)
    print(json.dumps(r, indent=1))
