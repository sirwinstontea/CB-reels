from pathlib import Path

helper = r'''
def _rebuild_rough_from_plate(plate_path, meta_path, markers, out_path):
    """Composite stand-in onto textured plate frames using the saved nib track."""
    from .standin import StandIn
    from .video import read_frames
    from .scene import OUT_W, OUT_H
    meta = json.loads(Path(meta_path).read_text())
    track = meta.get("track") or []
    if not track:
        shutil.copy2(plate_path, out_path)
        return
    hand = StandIn((0.15, -1.0))
    wr = Writer(out_path, OUT_W, OUT_H)
    for i, fr in enumerate(read_frames(plate_path, OUT_W, OUT_H, fps=FPS)):
        rough = fr.astype(np.float32) / 255.0
        if i < len(track):
            _, tipx, tipy, lift, mi = track[i]
            mi = int(mi)
            inkc = markers[mi]["ink"] if 0 <= mi < len(markers) else [80, 80, 80]
            cap = markers[mi]["cap"] if 0 <= mi < len(markers) else [80, 80, 80]
            hand.draw(rough, (float(tipx), float(tipy)), 1.0, inkc, cap, float(lift), 0.0)
        wr.write(rough)
    wr.close()


'''

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "pipeline.py"
t = p.read_text(encoding="utf-8")
if "_rebuild_rough_from_plate" not in t:
    t = t.replace("\ndef load_settings():\n", "\n" + helper + "def load_settings():\n", 1)
    p.write_text(t, encoding="utf-8")
    print("pipe2 helper ok")
else:
    print("pipe2 skip")
