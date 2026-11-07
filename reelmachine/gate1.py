"""Gate 1: color one page with the ink engine and write preview files.

python gate1.py <page.png> [out_dir]
Writes: regions.jpg, final.png, timelapse.mp4 (whole session sped up), closeup.mp4 (4 s real time), report.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from engine import colors, ink, page as pagemod, strokes

ROOT = Path(__file__).parent


def ffmpeg_writer(path, w, h, fps):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)


def to8(rgb):
    return (rgb * 255 + 0.5).astype(np.uint8)


def main(page_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    report = {}

    pg = pagemod.load_page_gated(page_path)
    report["page"] = pg.report | {"gap_close_px": pg.gap_close_px}
    pagemod.preview(pg, out / "regions.jpg")
    print("page", report["page"])

    markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
    plan = colors.plan_fallback(pg, markers)
    tl = strokes.plan(pg, plan, markers)
    report["timeline"] = {"duration_s": round(tl.duration, 1), "samples": int(len(tl.t)),
                          "strokes": int(tl.stroke.max() + 1), "regions": len(tl.region_start)}
    tl.save(out / "timeline.npz")
    print("timeline", report["timeline"], f"{time.time() - clock:.1f}s")

    canvas = ink.InkCanvas(pg, markers)
    h, w = pg.shape

    # closeup window: middle of the longest-coloring region, 4 s at real time
    rid = max(tl.region_start, key=lambda r: tl.region_end[r] - tl.region_start[r])
    t_mid = (tl.region_start[rid] + tl.region_end[rid]) / 2
    cw, ch = 720, 1280
    ry, rx = pg.regions[rid].centroid
    y0 = int(np.clip(ry - ch / 2, 0, h - ch)) if h >= ch else 0
    x0 = int(np.clip(rx - cw / 2, 0, w - cw))
    box = (y0, y0 + min(ch, h), x0, x0 + cw)
    close_times = set(np.round(np.arange(t_mid - 2, t_mid + 2, 1 / 30), 4))

    n_lapse = 600
    lapse_times = np.linspace(0, tl.duration + 2.5, n_lapse)
    size = 1080
    lapse = ffmpeg_writer(out / "timelapse.mp4", size, size, 30)
    close = ffmpeg_writer(out / "closeup.mp4", cw, box[1] - box[0], 30)
    events = sorted([(float(t), "lapse") for t in lapse_times] + [(float(t), "close") for t in close_times])
    t_ink = time.time()
    for t, kind in events:
        canvas.run(tl, t)
        if kind == "lapse":
            frame = cv2.resize(canvas.render(t), (size, size), interpolation=cv2.INTER_AREA)
            lapse.stdin.write(to8(frame).tobytes())
        else:
            frame = to8(canvas.render(t, box))
            i = min(int(np.searchsorted(tl.t, t)), len(tl.t) - 1)   # grey dot where the tip is (no hand yet)
            if box[2] <= tl.x[i] < box[3] and box[0] <= tl.y[i] < box[1]:
                cv2.circle(frame, (int(tl.x[i] - box[2]), int(tl.y[i] - box[0])), int(tl.radius[i]) + 3,
                           (90, 90, 90) if tl.down[i] else (170, 170, 170), 2, cv2.LINE_AA)
            close.stdin.write(frame.tobytes())
    for p in (lapse, close):
        p.stdin.close()
        p.wait()
    canvas.run(tl, tl.duration + 10)
    Image.fromarray(to8(canvas.render())).save(out / "final.png")
    report["ink_render_s"] = round(time.time() - t_ink, 1)
    report["total_s"] = round(time.time() - clock, 1)
    json.dump(report, open(out / "report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ROOT / "out" / "gate1")
