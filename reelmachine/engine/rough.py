"""Rough clips: for every shot, the plate (page coloring itself, no hand) and the rough
(same frames with the stand-in hand on the nib). The rough goes to the video model; the plate
is the ground truth that spot-the-difference checks its result against.

Per shot it writes  <id>.rough.mp4, <id>.plate.mp4, <id>.masks.npz (stand-in, 1/4 size),
<id>.inkmask.npz (ink coverage, 1/4 size), <id>.end.png and <id>.json (timing, nib track).
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .scene import Renderer, OUT_W, OUT_H
from .standin import StandIn
from .video import Writer, FPS

LIFT_S = 0.12     # pen rises over this long after it leaves the paper


def _pen_state(tl, t):
    i = int(np.clip(np.searchsorted(tl.t, t), 1, len(tl.t) - 1))
    a = (t - tl.t[i - 1]) / max(1e-6, tl.t[i] - tl.t[i - 1])
    x = tl.x[i - 1] + a * (tl.x[i] - tl.x[i - 1])
    y = tl.y[i - 1] + a * (tl.y[i] - tl.y[i - 1])
    # lift: how far (in time) the nearest pen-down sample is
    lo, hi = np.searchsorted(tl.t, [t - LIFT_S, t + LIFT_S])
    near = tl.down[lo:hi]
    if near.any():
        idx = np.flatnonzero(near) + lo
        gap = float(np.min(np.abs(tl.t[idx] - t)))
        lift = 0.0 if tl.down[i - 1] or tl.down[i] else min(1.0, gap / LIFT_S)
    else:
        lift = 1.0
    return float(x), float(y), lift, int(tl.marker[i])


def render(tl, page, canvas, markers, shots, desks, out_dir, progress=None):
    """Render all shots in time order with one canvas. Returns {shot id: info dict}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    total = sum(int(round((s.t1 - s.t0) * FPS)) for s in shots)
    done = 0
    for shot in sorted(shots, key=lambda s: s.t0):
        view = shot.view(desks)
        r = Renderer(view, page.lines.shape)
        hand = StandIn(view.desk.pen_dir)
        n = int(round((shot.t1 - shot.t0) * FPS))
        masks, inkmasks, track = [], [], []
        wr_plate = Writer(out / f"{shot.id}.plate.mp4", OUT_W, OUT_H)
        wr_rough = Writer(out / f"{shot.id}.rough.mp4", OUT_W, OUT_H) if shot.hand else None
        last = None
        for f in range(n):
            t = shot.t0 + f / FPS
            canvas.run(tl, t)
            plate = r.frame(canvas, t)
            wr_plate.write(plate)
            last = plate
            ink_m = r.warp_mask(canvas.inked())
            inkmasks.append(cv2.resize(ink_m.astype(np.uint8), (OUT_W // 4, OUT_H // 4),
                                       interpolation=cv2.INTER_AREA) > 0)
            if wr_rough:
                x, y, lift, mi = _pen_state(tl, t)
                tip = r.to_out(x, y)
                k = r.scale_at(x, y) * page.px_per_mm
                theta = 0.05 * np.sin(2 * np.pi * 0.31 * t) + 0.03 * np.sin(2 * np.pi * 1.13 * t + 1.0)
                rough = plate.copy()
                sil = hand.draw(rough, tip, k, markers[mi]["ink"], markers[mi]["cap"], lift, theta)
                wr_rough.write(rough)
                masks.append(cv2.resize(sil, (OUT_W // 4, OUT_H // 4), interpolation=cv2.INTER_AREA) > 0)
                track.append([round(t, 4), round(tip[0], 2), round(tip[1], 2), round(lift, 3), mi])
            done += 1
            if progress and f % 15 == 0:
                progress(done / total, f"{shot.id}: frame {f + 1}/{n}")
        wr_plate.close()
        np.savez_compressed(out / f"{shot.id}.inkmask.npz", masks=np.array(inkmasks))
        if wr_rough:
            wr_rough.close()
            np.savez_compressed(out / f"{shot.id}.masks.npz", masks=np.array(masks))
        cv2.imwrite(str(out / f"{shot.id}.end.png"),
                    cv2.cvtColor((np.clip(last, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        meta = shot.to_dict() | {"frames": n, "fps": FPS, "track": track,
                                 "markers": sorted({tr[4] for tr in track})}
        (out / f"{shot.id}.json").write_text(json.dumps(meta))
        info[shot.id] = meta
    return info
