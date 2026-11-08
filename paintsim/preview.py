"""Quick diagnostic previews (particle scatter, side + top view, pigment-coloured). Not the paper renderer."""
import sys

import numpy as np
from PIL import Image, ImageDraw


def _colors(f):
    c = f["c"]
    rgb = f["pigment_rgb"]
    if c.shape[0] == 0:
        return np.zeros((0, 3))
    w = np.clip(c.sum(1, keepdims=True), 0, 1)
    mix = (c @ rgb) / np.maximum(c.sum(1, keepdims=True), 1e-9)
    clear = np.array([0.75, 0.85, 0.95])
    return w * mix + (1 - w) * clear


def preview(npz, out, size=(520, 400), extent=None):
    f = dict(np.load(npz))
    if "absorbed" in f and f["absorbed"].any():
        keep = ~f["absorbed"]
        f["x"], f["c"] = f["x"][keep], f["c"][keep]
    x = f["x"]
    col = (_colors(f) * 255).astype(np.uint8)
    W, H = size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    px = f["paper_x"]
    if extent is None:
        allx = np.vstack([x, px]) if x.shape[0] else px
        extent = (allx[:, 0].min(), allx[:, 0].max(), allx[:, 1].min(), allx[:, 1].max(), -0.02,
                  max(0.1, allx[:, 2].max()))
    x0, x1, y0, y1, z0, z1 = extent
    # top half: top view, bottom strip: side view (x-z)
    th = int(H * 0.62)
    sx = (W - 10) / (x1 - x0)
    sy = (th - 10) / (y1 - y0)
    s = min(sx, sy)
    # paper pigment
    pc = f["paper_c"]
    if pc.shape[0] and pc.sum() > 0:
        dens = pc @ f["pigment_rgb"]
        a = np.clip(pc.sum(1) / (np.percentile(pc.sum(1)[pc.sum(1) > 0], 90) + 1e-12), 0, 1)[:, None]
        pcol = ((1 - a) * 1.0 + a * dens / np.maximum(pc.sum(1, keepdims=True), 1e-12)) * 255
        for p, cc in zip(px, pcol.astype(int)):
            u = 5 + (p[0] - x0) * s
            v = th - 5 - (p[1] - y0) * s
            d.rectangle([u - 1, v - 1, u + 1, v + 1], fill=tuple(cc))
    order = np.argsort(x[:, 2]) if x.shape[0] else []
    for i in order:
        u = 5 + (x[i, 0] - x0) * s
        v = th - 5 - (x[i, 1] - y0) * s
        d.point((u, v), fill=tuple(col[i]))
    d.line([0, th, W, th], fill=(0, 0, 0))
    sz = (H - th - 10) / (z1 - z0)
    s2 = min(sx, sz)
    for i in np.argsort(x[:, 1]) if x.shape[0] else []:
        u = 5 + (x[i, 0] - x0) * s2
        v = H - 5 - (x[i, 2] - z0) * s2
        d.point((u, v), fill=tuple(col[i]))
    d.line([5, H - 5 - (0 - z0) * s2, W - 5, H - 5 - (0 - z0) * s2], fill=(120, 90, 60))
    d.text((6, th + 2), "t=%.2fs n=%d" % (float(f["time"]), x.shape[0]), fill=(0, 0, 0))
    img.save(out, quality=70)


def strip(npzs, out, size=(260, 200)):
    ims = []
    for p in npzs:
        tmp = out + ".tmp.jpg"
        preview(p, tmp, size=size)
        ims.append(Image.open(tmp).copy())
    W = size[0] * len(ims)
    canvas = Image.new("RGB", (W, size[1]), (255, 255, 255))
    for k, im in enumerate(ims):
        canvas.paste(im, (k * size[0], 0))
    canvas.save(out, quality=70)


if __name__ == "__main__":
    preview(sys.argv[1], sys.argv[2])
