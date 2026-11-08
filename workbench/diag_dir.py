import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\truet\Documents\Coloring books\_Working files\ColoringReels\reelmachine")
sys.path.insert(0, str(ROOT))
from engine import colorai, colors, director, strokes
from engine import page as pagemod
J = ROOT / "jobs" / "bubble-bath-test"
pg = pagemod.load_page_gated(J / "page.png")
mk = colors.load_markers(ROOT / "markers" / "basic-36.json")
plan, _ = colorai.plan(pg, mk, cache_path=J / "colorplan.json")
tl = strokes.plan(pg, plan, mk)
T = tl.duration
print("T", round(T), "regions in tl", len(tl.region_start))
rows = []
for rid, rs in tl.region_start.items():
    re = tl.region_end[rid]
    t0 = rs + min(2.0, (re - rs - 3) / 2)
    fill = director._window_ok(tl, t0, t0 + 3, rid) if re - rs >= 4 else -1
    rows.append((round(rs), round(re - rs, 1), round(fill, 2), rid, round(pg.regions[rid].area / pg.px_per_mm ** 2)))
rows.sort()
for r in rows:
    print(r)

from engine import scene
desks = {a: scene.load_desk(ROOT / "assets" / "desks" / f"{a}.jpg") for a in "AB"}
for s in director.plan_shots(tl, pg, mk, desks):
    print(s.id, s.zoom, s.angle, s.t0, s.region, s.marker, mk[s.marker]["id"] if s.marker >= 0 else "")
