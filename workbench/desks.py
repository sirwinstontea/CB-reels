"""Install the user's desk photos as angle A/B, detect the sheet corners, save previews."""
import json, shutil, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"C:\Users\truet\Documents\Coloring books\_Working files\ColoringReels\reelmachine")
sys.path.insert(0, str(ROOT))
from engine.scene import find_sheet, _order

SCR = Path(r"C:\Users\truet\AppData\Local\Temp\claude\C--Users-truet\fe4b2cf0-68f9-4499-86ec-37b95482a4f1\scratchpad")
D = ROOT / "assets" / "desks"
bk = D / "_placeholder"
bk.mkdir(exist_ok=True)
for f in D.glob("*.*"):
    if f.is_file() and not (bk / f.name).exists():
        shutil.copy(f, bk / f.name)

A = cv2.imread(r"C:\Users\truet\Downloads\hf_20261025_230147_19f239ba-74e9-4c4c-adff-f8488c76263f.png")
B = cv2.imread(r"C:\Users\truet\Pictures\Screenshots\Screenshot 2026-10-26 001150.png")
B = cv2.resize(B, (B.shape[1] * 5 // 2, B.shape[0] * 5 // 2), interpolation=cv2.INTER_LANCZOS4)
blur = cv2.GaussianBlur(B, (0, 0), 2.0)
B = cv2.addWeighted(B, 1.35, blur, -0.35, 0)


def sheet_mask(img8, v=170, s=50):
    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > v) & (hsv[..., 1] < s)).astype(np.uint8)


def quad_from_mask(m):
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    for eps in (0.01, 0.015, 0.02, 0.03, 0.05):
        q = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(q) == 4:
            return _order(q)
    # fall back: min-area rectangle
    return _order(cv2.boxPoints(cv2.minAreaRect(c)))


# A: right page of the open sketchbook. Split the spread at the gutter (darkest column in the page band).
mA = sheet_mask(A, 185, 40)
ys, xs = np.nonzero(mA)
band = A[int(np.percentile(ys, 30)):int(np.percentile(ys, 70)), :, :].mean(axis=(0, 2))
x_lo, x_hi = int(np.percentile(xs, 5)), int(np.percentile(xs, 95))
mid = (x_lo + x_hi) // 2
win = band[mid - 150:mid + 150]
gutter = mid - 150 + int(np.argmin(win))
mR = mA.copy()
mR[:, :gutter + 6] = 0
qA = quad_from_mask(mR)
print("A gutter", gutter, "corners", qA.round(1).tolist())

mB = sheet_mask(B, 205, 17)
qB = quad_from_mask(mB)
print("B corners", qB.round(1).tolist(), "size", B.shape)

for name, img, q in (("A", A, qA), ("B", B, qB)):
    prev = img.copy()
    cv2.polylines(prev, [q.astype(np.int32)], True, (0, 0, 255), 6)
    for i, p in enumerate(q):
        cv2.putText(prev, str(i), tuple(int(v) for v in p), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
    s = 520 / max(prev.shape[:2])
    cv2.imwrite(str(SCR / f"desk{name}_q.jpg"), cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 70])

if "--save" in sys.argv:
    cv2.imwrite(str(D / "A.jpg"), A, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(D / "B.jpg"), B, [cv2.IMWRITE_JPEG_QUALITY, 95])
    json.dump({"corners": qA.tolist()}, open(SCR / "qA.json", "w"))
    json.dump({"corners": qB.tolist()}, open(SCR / "qB.json", "w"))

