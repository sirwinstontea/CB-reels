"""Framing preview: one frame per planned shot (page lines + ink at shot start + stand-in), no video."""
import sys, json
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(r"C:\Users\truet\Documents\Coloring books\_Working files\ColoringReels\reelmachine")
sys.path.insert(0, str(ROOT))
from engine import colors, director, ink, scene, strokes
from engine import page as pagemod
from engine.rough import _pen_state
from engine.standin import StandIn

SCR = Path(__file__).parent
PAGE = r"C:\Users\truet\Documents\Coloring books\_Working files\NataleCozy\bookgen\books\snug-cozy-christmas\pages_final\page_16.png"
pg = pagemod.load_page_gated(PAGE)
markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
plan = colors.plan_fallback(pg, markers)
tl = strokes.plan(pg, plan, markers)
print("regions", len(pg.regions), "session min", round(tl.duration / 60, 1), "strokes", int(tl.stroke.max()) + 1)
desks = {a: scene.load_desk(ROOT / "assets" / "desks" / f"{a}.jpg") for a in "AB"}
shots = director.plan_shots(tl, pg, markers, desks)
cv = ink.InkCanvas(pg, markers)
tiles = []
for s in sorted(shots, key=lambda s: s.t0):
    print(s.id, s.zoom, s.angle, s.t0, s.t1, round(s.width), "region", s.region)
    cv.run(tl, s.t0)
    v = s.view(desks)
    r = scene.Renderer(v, pg.lines.shape)
    img = r.frame(cv, s.t0)
    if s.hand:
        x, y, lift, mi = _pen_state(tl, s.t0)
        StandIn(v.desk.pen_dir).draw(img, r.to_out(x, y), r.scale_at(x, y) * pg.px_per_mm,
                                     markers[mi]["ink"], markers[mi]["cap"], lift, 0.0)
    tiles.append(cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8), (180, 320), interpolation=cv2.INTER_AREA))
sheet = np.concatenate(tiles, 1)
cv2.imwrite(str(SCR / "framing.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 72])
