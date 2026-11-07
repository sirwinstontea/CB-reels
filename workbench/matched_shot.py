"""One close plate from a matched ref page + local texture; side-by-side with the reference."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reelmachine"))

from engine import colors, director, ink, page as pagemod, rough, scene, shading, strokes, texmod
from engine.scene import OUT_W, OUT_H
from engine.video import Writer, FPS, read_frames

MATCH = ROOT / "workbench" / "matched_ref" / "ref01_halloween-house-girl-ghost_31s"
OUT = ROOT / "workbench" / "matched_ref" / "out_ref01"
REF_VID = ROOT / "reference-videos" / "ref01_halloween-house-girl-ghost_31s.mp4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pg = pagemod.load_page_gated(MATCH / "page.png")
    markers = colors.load_markers(ROOT / "reelmachine" / "markers" / "basic-36.json")
    target = MATCH / "target.png"
    if target.exists():
        plan, gate = colors.plan_from_target(pg, target, markers, min_align=0.0)
        print("color gate", gate, flush=True)
        if plan is None:
            plan = colors.plan_fallback(pg, markers)
    else:
        plan = colors.plan_fallback(pg, markers)
    # plan may be dict rid->idx or richer
    dump = plan if isinstance(plan, dict) else {"plan": str(plan)}
    (OUT / "colorplan.json").write_text(json.dumps({str(k): v for k, v in dump.items()}, default=str))
    layers = shading.layers_for(pg, plan, markers)
    tl = strokes.plan(pg, plan, markers, seed=2, layers=layers)
    print(f"plan {tl.duration:.0f}s strokes ready in {time.time()-t0:.1f}s", flush=True)

    desks = {}
    for a in ("A", "B"):
        p = ROOT / "reelmachine" / "assets" / "desks" / f"{a}.jpg"
        if not p.exists():
            scene.make_placeholder_desk(p, a, seed=ord(a))
        desks[a] = scene.load_desk(p)
    shots = director.plan_shots(tl, pg, markers, desks)
    # keep first close/medium hand shot only, trim to 3s window mid-session
    shots = [s for s in shots if s.hand][:1]
    if not shots:
        raise SystemExit("no hand shot")
    # force a 3s window around 40% of session for a fair ink-progress look
    s0 = shots[0]
    mid = tl.duration * 0.4
    s0.t0, s0.t1 = max(0, mid - 1.5), min(tl.duration, mid + 1.5)
    if s0.t1 - s0.t0 < 2.5:
        s0.t1 = s0.t0 + 3.0
    clips = OUT / "clips"
    clips.mkdir(exist_ok=True)
    cv = ink.InkCanvas(pg, markers)
    rough.render(tl, pg, cv, markers, [s0], desks, clips)
    plate = clips / f"{s0.id}.plate.mp4"
    ink_p = clips / f"{s0.id}.inkmask.npz"
    tex = clips / f"{s0.id}.plate.tex.mp4"
    rep = texmod.apply_refs(plate, ink_p, tex, strength=0.08, edge=0.06)
    print("texmod", rep, flush=True)

    # side-by-side vs ref at similar look
    ref_frame = None
    cap = cv2.VideoCapture(str(REF_VID))
    cap.set(cv2.CAP_PROP_POS_MSEC, 8000)
    ok, bgr = cap.read()
    cap.release()
    if ok:
        ref_frame = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (OUT_W, OUT_H))
    ours = None
    for i, fr in enumerate(read_frames(tex, OUT_W, OUT_H, fps=FPS)):
        if i == int(1.5 * FPS):
            ours = fr
            break
    if ours is not None and ref_frame is not None:
        # letterbox labels
        canvas = np.zeros((OUT_H, OUT_W * 2, 3), np.uint8)
        canvas[:, :OUT_W] = ref_frame
        canvas[:, OUT_W:] = ours
        cv2.putText(canvas, "REF", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(canvas, "OURS", (OUT_W + 40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.imwrite(str(OUT / "side_by_side.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        # also short sbs video: ours only for now + still ref strip
        wr = Writer(OUT / "matched_plate.mp4", OUT_W, OUT_H)
        for fr in read_frames(tex, OUT_W, OUT_H, fps=FPS):
            wr.write(fr)
        wr.close()
    print("OUT", OUT, "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
