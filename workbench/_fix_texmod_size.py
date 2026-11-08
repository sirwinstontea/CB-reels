from pathlib import Path
p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "texmod.py"
t = p.read_text(encoding="utf-8")
# make apply_refs use plate size
old = """    n = _count(plate_path)
    masks = np.load(inkmask_path)["masks"]
    n = min(n, len(masks))
    g = max(1, 6 // Q)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), kk) for m in masks[:n]]) > 0
    grain = _tiled(_grain_tile(tex_root), OUT_H, OUT_W)

    tmp = Path(str(out_path) + ".part.mp4")
    wr = Writer(tmp, OUT_W, OUT_H)
"""
new = """    info = probe(plate_path)
    W, H = int(info["w"]), int(info["h"])
    n = _count(plate_path)
    masks = np.load(inkmask_path)["masks"]
    n = min(n, len(masks))
    g = max(1, 6 // Q)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), kk) for m in masks[:n]]) > 0
    grain = _tiled(_grain_tile(tex_root), H, W)

    tmp = Path(str(out_path) + ".part.mp4")
    wr = Writer(tmp, W, H)
"""
if old not in t:
    raise SystemExit("block missing")
t = t.replace(old, new, 1)
t = t.replace(
    """            w = cv2.resize(ink[i].astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
""",
    """            w = cv2.resize(ink[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
""",
)
p.write_text(t, encoding="utf-8")
print("texmod sized")
