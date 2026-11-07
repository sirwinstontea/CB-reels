"""Page-space ink video (no desk) from matched ref page, side-by-side with ref crop."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reelmachine"))

from engine import colors, ink, page as pagemod, shading, strokes, texmod
from engine.video import Writer, FPS

MATCH = ROOT / "workbench" / "matched_ref" / "ref01_halloween-house-girl-ghost_31s"
OUT = ROOT / "workbench" / "matched_ref" / "inkcmp_ref01"
REF = ROOT / "reference-videos" / "ref01_halloween-house-girl-ghost_31s.mp4"
W, H = 540, 960  # output compare half


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pg = pagemod.load_page_gated(MATCH / "page.png")
    markers = colors.load_markers(ROOT / "reelmachine" / "markers" / "basic-36.json")
    plan, gate = colors.plan_from_target(pg, MATCH / "target.png", markers, min_align=0.0)
    print("gate", gate, flush=True)
    if plan is None:
        plan = colors.plan_fallback(pg, markers)
    tl = strokes.plan(pg, plan, markers, seed=3, layers=shading.layers_for(pg, plan, markers))
    print(f"session {tl.duration:.0f}s", flush=True)
    cv = ink.InkCanvas(pg, markers)
    # animate last 4s of a mid-progress moment
    t_end = min(tl.duration, max(8.0, tl.duration * 0.35))
    t_start = max(0.0, t_end - 4.0)
    cv.run(tl, t_start)
    ph, pw = pg.lines.shape
    scale = min(W / pw, H / ph)
    ow, oh = int(pw * scale), int(ph * scale)

    # build inkmask frames while rendering plate in page space
    n = int(round((t_end - t_start) * FPS))
    plates, masks = [], []
    for f in range(n):
        t = t_start + f / FPS
        cv.run(tl, t)
        rgb = (np.clip(cv.render(t), 0, 1) * 255).astype(np.uint8)
        inked = cv.inked().astype(np.uint8)
        rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        inked = cv2.resize(inked, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # letterbox to WxH
        canvas = np.full((H, W, 3), 245, np.uint8)
        y0, x0 = (H - oh) // 2, (W - ow) // 2
        canvas[y0:y0 + oh, x0:x0 + ow] = rgb
        mfull = np.zeros((H, W), np.uint8)
        mfull[y0:y0 + oh, x0:x0 + ow] = inked
        plates.append(canvas)
        masks.append(mfull > 0)

    flat_path = OUT / "page_flat.mp4"
    wr = Writer(flat_path, W, H)
    for fr in plates:
        wr.write(fr)
    wr.close()
    np.savez_compressed(OUT / "inkmask.npz", masks=np.array(masks))
    tex_path = OUT / "page_tex.mp4"
    # write temp plate already done; texmod expects video — reuse flat
    rep = texmod.apply_refs(flat_path, OUT / "inkmask.npz", tex_path, strength=0.07, edge=0.05)
    print("tex", rep, flush=True)

    # ref crop from same extract box if available, else center crop
    cap = cv2.VideoCapture(str(REF))
    cap.set(cv2.CAP_PROP_POS_MSEC, 10000)
    ok, bgr = cap.read()
    cap.release()
    ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else np.full((H, W, 3), 200, np.uint8)
    # use matched crop_early size ratio — load target as colored ref look
    tgt = cv2.imread(str(MATCH / "target.png"))
    if tgt is not None:
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
        ref = cv2.resize(tgt, (W, H), interpolation=cv2.INTER_AREA)

    ours = plates[n // 2]
    # reread textured mid
    from engine.video import read_frames
    for i, fr in enumerate(read_frames(tex_path, W, H, fps=FPS)):
        if i == n // 2:
            ours = fr
            break
    sbs = np.concatenate([cv2.resize(ref, (W, H)), ours], axis=1)
    cv2.putText(sbs, "REF page", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.putText(sbs, "OURS ink", (W + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.imwrite(str(OUT / "side_by_side.jpg"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("done", OUT, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
