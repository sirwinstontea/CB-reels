from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "spot.py"
t = p.read_text(encoding="utf-8")
if "def lock_ink(" in t:
    print("spot.lock_ink already present")
else:
    add = '''

def lock_ink(result_path, plate_path, inkmask_path, out_path=None):
    """Texture-edit referee: outside the ink mask, restore plate pixels.

    Inside the ink mask the model may add marker texture. Black lines / blank paper /
    desk outside ink must match the flat plate. Writes repaired clip when out_path set.
    """
    info = probe(result_path)
    n_plate = _count(plate_path)
    masks = np.load(inkmask_path)["masks"] if inkmask_path and Path(inkmask_path).exists() else None
    if masks is None:
        return {"ok": False, "fails": ["missing inkmask"], "result": str(result_path)}
    n = min(n_plate, _count(result_path, FPS), len(masks))
    if n < 3:
        return {"ok": False, "fails": ["clip too short"], "result": str(result_path), "src": info}
    # grow ink a little so soft stroke edges count as "inside"
    g = max(1, 6 // Q)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), k) for m in masks[:n]]) > 0
    fails, changed = [], []
    tmp = Path(str(out_path) + ".part.mp4") if out_path else None
    wr = Writer(tmp, OUT_W, OUT_H) if tmp else None
    ga, gp = read_frames(result_path, OUT_W, OUT_H, fps=FPS), read_frames(plate_path)
    for i, a, p in zip(range(n), ga, gp):
        la, lp = _lab(cv2.resize(a, (OUT_W // Q, OUT_H // Q), interpolation=cv2.INTER_AREA)), \\
                 _lab(cv2.resize(p, (OUT_W // Q, OUT_H // Q), interpolation=cv2.INTER_AREA))
        de = np.linalg.norm(la - lp, axis=2)
        outside = ~ink[i]
        changed.append(float((de[outside] > DE_CHANGED).mean()) if outside.any() else 0.0)
        if wr:
            w = cv2.resize(ink[i].astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)[..., None]
            w = np.clip(w, 0, 1)
            fixed = (a.astype(np.float32) * w + p.astype(np.float32) * (1 - w)).astype(np.uint8)
            wr.write(fixed)
    ga.close(); gp.close()
    if wr:
        wr.close()
    changed = np.array(changed)
    bad = changed > LIMITS["outside_changed"]
    rep = {"result": str(result_path), "src": info, "frames": n,
           "outside_changed": {"mean": round(float(changed.mean()), 4), "max": round(float(changed.max()), 4),
                               "bad_frames": int(bad.sum())},
           "fails": fails, "ok": True}
    if bad.mean() > LIMITS["bad_frames"]:
        fails.append(f"page changed outside ink in {bad.sum()} of {n} frames")
        rep["ok"] = False
        rep["fails"] = fails
    if tmp:
        if rep["ok"] and tmp.exists():
            os.replace(tmp, out_path)
            rep["repaired"] = str(out_path)
        elif tmp.exists():
            tmp.unlink()
    return rep
'''
    # fix accidental escapes from writing
    add = add.replace(" \\\\\n", "\n")
    # insert before __main__
    anchor = '\nif __name__ == "__main__":\n'
    if anchor not in t:
        raise SystemExit("spot __main__ missing")
    t = t.replace(anchor, add + anchor, 1)
    p.write_text(t, encoding="utf-8")
    print("spot.py patched")
