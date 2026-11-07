"""Gate 2: rough clips. Page -> colors -> strokes -> shot plan -> rough + plate clips for every shot.

python gate2.py <page.png> [out_dir]
Uses assets/desks/A.jpg and B.jpg (placeholder desks are made if missing).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from engine import colors, director, ink, page as pagemod, rough, scene, strokes
from engine.video import read_frames

ROOT = Path(__file__).parent


def desks():
    d = {}
    for a in ("A", "B"):
        p = ROOT / "assets" / "desks" / f"{a}.jpg"
        if not p.exists():
            scene.make_placeholder_desk(p, a, seed=ord(a))
        d[a] = scene.load_desk(p)
    return d


def contact_sheet(out, shots, path):
    tiles = []
    for s in shots:
        name = f"{s.id}.rough.mp4" if s.hand else f"{s.id}.plate.mp4"
        frames = list(read_frames(out / name, 180, 320))
        pick = [frames[0], frames[len(frames) // 2], frames[-1]]
        tiles.append(np.hstack(pick))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])


def main(page_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    pg = pagemod.load_page_gated(page_path)
    markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
    tl = strokes.plan(pg, colors.plan_fallback(pg, markers), markers)
    dk = desks()
    shots = director.plan_shots(tl, pg, markers, dk)
    for s in shots:
        print(s.id, s.zoom, s.angle, f"{s.t0:.1f}-{s.t1:.1f}s", "region", s.region, "hand" if s.hand else "no hand")
    cv = ink.InkCanvas(pg, markers)
    rough.render(tl, pg, cv, markers, shots, dk, out)
    contact_sheet(out, shots, out / "sheet.jpg")
    (out / "shots.json").write_text(json.dumps([s.to_dict() for s in shots], indent=1))
    print(f"done in {time.time() - clock:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ROOT / "out" / "gate2")
