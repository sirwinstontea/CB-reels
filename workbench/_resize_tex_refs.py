from pathlib import Path
import cv2
import numpy as np

src = Path(__file__).resolve().parents[1] / "reelmachine" / "assets" / "marker_textures" / "alcohol"
out = src / "api"
out.mkdir(exist_ok=True)
W, H = 720, 1280
for p in sorted(src.iterdir()):
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg"} or p.is_dir():
        continue
    im = cv2.imread(str(p))
    if im is None:
        print("skip", p)
        continue
    h, w = im.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), 255, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = r
    dest = out / f"{p.stem}.jpg"
    cv2.imwrite(str(dest), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(p.name, "->", dest.name, f"{w}x{h}")
print("done", out)
