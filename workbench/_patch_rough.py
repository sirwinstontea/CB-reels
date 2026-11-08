from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "rough.py"
t = p.read_text(encoding="utf-8")
t2 = t.replace(
    "Per shot it writes  <id>.rough.mp4, <id>.plate.mp4, <id>.masks.npz (where the stand-in was,\n"
    "per frame, at 1/4 size), <id>.end.png (last plate frame) and <id>.json (timing, nib track).",
    "Per shot it writes  <id>.rough.mp4, <id>.plate.mp4, <id>.masks.npz (stand-in, 1/4 size),\n"
    "<id>.inkmask.npz (ink coverage, 1/4 size), <id>.end.png and <id>.json (timing, nib track).",
)
t = t2
old = (
    "        masks, track = [], []\n"
    "        wr_plate = Writer(out / f\"{shot.id}.plate.mp4\", OUT_W, OUT_H)\n"
    "        wr_rough = Writer(out / f\"{shot.id}.rough.mp4\", OUT_W, OUT_H) if shot.hand else None\n"
    "        last = None\n"
    "        for f in range(n):\n"
    "            t = shot.t0 + f / FPS\n"
    "            canvas.run(tl, t)\n"
    "            plate = r.frame(canvas, t)\n"
    "            wr_plate.write(plate)\n"
    "            last = plate\n"
    "            if wr_rough:\n"
)
new = (
    "        masks, inkmasks, track = [], [], []\n"
    "        wr_plate = Writer(out / f\"{shot.id}.plate.mp4\", OUT_W, OUT_H)\n"
    "        wr_rough = Writer(out / f\"{shot.id}.rough.mp4\", OUT_W, OUT_H) if shot.hand else None\n"
    "        last = None\n"
    "        for f in range(n):\n"
    "            t = shot.t0 + f / FPS\n"
    "            canvas.run(tl, t)\n"
    "            plate = r.frame(canvas, t)\n"
    "            wr_plate.write(plate)\n"
    "            last = plate\n"
    "            ink_m = r.warp_mask(canvas.inked())\n"
    "            inkmasks.append(cv2.resize(ink_m.astype(np.uint8), (OUT_W // 4, OUT_H // 4),\n"
    "                                       interpolation=cv2.INTER_AREA) > 0)\n"
    "            if wr_rough:\n"
)
if old not in t:
    raise SystemExit("rough loop anchor missing")
t = t.replace(old, new, 1)
old2 = (
    "        if wr_rough:\n"
    "            wr_rough.close()\n"
    "            np.savez_compressed(out / f\"{shot.id}.masks.npz\", masks=np.array(masks))\n"
)
new2 = (
    "        np.savez_compressed(out / f\"{shot.id}.inkmask.npz\", masks=np.array(inkmasks))\n"
    "        if wr_rough:\n"
    "            wr_rough.close()\n"
    "            np.savez_compressed(out / f\"{shot.id}.masks.npz\", masks=np.array(masks))\n"
)
if old2 not in t:
    raise SystemExit("rough save anchor missing")
t = t.replace(old2, new2, 1)
p.write_text(t, encoding="utf-8")
print("rough.py patched")
