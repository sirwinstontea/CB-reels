"""Spot-the-difference against fake model results made from Gate 2 output.

python tests/test_spot.py [gate2_dir] [shot]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import spot
from engine.video import Writer, read_frames
from engine.scene import OUT_W, OUT_H

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "out" / "gate2"
sid = sys.argv[2] if len(sys.argv) > 2 else "s01"
tmp = d / "spot_test"
tmp.mkdir(exist_ok=True)
plate0 = next(iter(read_frames(d / f"{sid}.plate.mp4")))


def fake(name, make):
    """Streams rough + plate (never holds a clip: low RAM) and writes make(rough, plate) frames."""
    p = tmp / f"{name}.mp4"
    with Writer(p, OUT_W, OUT_H, fps=24) as w:       # models often return 24 fps: resampling is part of the test
        for i, (r, pl) in enumerate(zip(read_frames(d / f"{sid}.rough.mp4"), read_frames(d / f"{sid}.plate.mp4"))):
            if i % 5 == 4:
                continue                              # drop every 5th frame -> 24 fps of the same span
            w.write(make(r, pl))
    return p


M = np.float32([[1.004, 0, 3], [0, 1.004, -2]])
good = lambda r, p: cv2.GaussianBlur(np.clip(cv2.warpAffine(r, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32) * [1.05, 1.0, 0.95] + 6, 0, 255).astype(np.uint8), (3, 3), 0)
cases = {
    "good": (good, True),
    "no_hand": (lambda r, p: p, False),
    "frozen_page": (lambda r, p: np.where(np.abs(r.astype(int) - p.astype(int)).sum(2, keepdims=True) > 30, r, plate0).astype(np.uint8), False),
}
ok = True
for name, (frames, expect) in cases.items():
    (tmp / f"{name}.repaired.mp4").unlink(missing_ok=True)      # repaired clip must exist only for a pass
    rep = spot.check(fake(name, frames), d / f"{sid}.plate.mp4", d / f"{sid}.masks.npz",
                     out_path=tmp / f"{name}.repaired.mp4")
    got = rep["ok"]
    ok &= got == expect
    ok &= (tmp / f"{name}.repaired.mp4").exists() == got if not (tmp / f"{name}.repaired.mp4.part.mp4").exists() else False
    print(f"{name:12s} expect {'pass' if expect else 'fail'} -> {'pass' if got else 'fail'}  "
          f"outside={rep.get('outside_changed')} hand_missing={rep.get('hand_missing')} shift={rep.get('align', {}).get('shift')} {rep['fails']}")
print("ALL OK" if ok else "MISMATCH")
