"""Fast ink check: color the whole page, save the finished page and two close crops. No video.

python inktest.py <page.png> [out_dir]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from engine import colors, ink, page as pagemod, shading, strokes

ROOT = Path(__file__).parent


def main(page_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    pg = pagemod.load_page_gated(page_path)
    markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
    plan = colors.plan_fallback(pg, markers)
    tl = strokes.plan(pg, plan, markers, layers=shading.layers_for(pg, plan, markers))   # base + auto shade
    cv = ink.InkCanvas(pg, markers)
    # mid-session look around the nib: wet ink, the working front
    tm = tl.duration / 2
    cv.run(tl, tm)
    i = int(np.searchsorted(tl.t, tm))
    y0, x0 = max(0, int(tl.y[i]) - 300), max(0, int(tl.x[i]) - 400)
    mid = cv.render(tm, (y0, y0 + 600, x0, x0 + 800))
    cv2.imwrite(str(out / "mid.jpg"), cv2.cvtColor((mid * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)[::2, ::2],
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv.run(tl, tl.duration + 30)
    rgb = cv.render()
    bgr = cv2.cvtColor((rgb * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "final.png"), bgr)
    cv2.imwrite(str(out / "v1.jpg"), cv2.resize(bgr, (340, 340), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(str(out / "v2.jpg"), cv2.resize(bgr[700:1212, 1000:1512], (300, 300), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "v3.jpg"), bgr[1500:1800, 900:1200], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"done in {time.time() - clock:.1f}s, session {tl.duration:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ROOT / "out" / "inktest")
