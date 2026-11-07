"""Pull clean line-art from a reference early frame (white paper + black lines only)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def extract(video: Path, t: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame at {t}s")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > 150) & (hsv[..., 1] < 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise SystemExit("no page contour")
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    crop = rgb[y0:y1, x0:x1]
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # adaptive: lines = locally dark
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    score = blur - g
    lines = (score > 18) & (g < 140)
    # clean speckles
    lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    page = np.full((*g.shape, 3), 255, np.uint8)
    page[lines > 0] = 0
    # color target from later frame, same crop
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, (t + 10) * 1000)
    ok2, bgr2 = cap.read()
    cap.release()
    if ok2:
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        # replace non-line paper with whitened target colors for planning
        tgt = rgb2.copy()
        cv2.imwrite(str(out_dir / "target.png"), cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "page.png"), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "crop_early.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out_dir, "page", page.shape, "line%", round(100 * lines.mean(), 1))


if __name__ == "__main__":
    vid = ROOT / "reference-videos" / (sys.argv[1] if len(sys.argv) > 1 else "ref01_halloween-house-girl-ghost_31s.mp4")
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    out = ROOT / "workbench" / "matched_ref" / vid.stem
    extract(vid, t, out)
