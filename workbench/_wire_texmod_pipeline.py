from pathlib import Path

# pipeline: use texmod.apply after Omni texture
p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "pipeline.py"
t = p.read_text(encoding="utf-8")
if "texmod" not in t.split("from . import", 1)[1][:300]:
    t = t.replace(
        "from . import colorai, colors, handref, director, editor, final, ink, judge, rough, scene, shading, sound, spot, strokes, texture",
        "from . import colorai, colors, handref, director, editor, final, ink, judge, rough, scene, shading, sound, spot, strokes, texture, texmod",
        1,
    )
old = '                rep = spot.lock_ink(res, plate_p, ink_p, out_path=clips / f"{x.id}.plate.tex.mp4")\n'
new = '                rep = texmod.apply(res, plate_p, ink_p, clips / f"{x.id}.plate.tex.mp4")\n'
if old not in t:
    raise SystemExit("lock_ink call missing")
t = t.replace(old, new, 1)
# ok check: texmod uses ok from outside_mae
t = t.replace(
    '                job.log({"clip": cid, "step": "texture_spot", "ok": rep.get("ok"), "fails": rep.get("fails")})\n'
    '                if rep.get("ok") and rep.get("repaired"):\n',
    '                job.log({"clip": cid, "step": "texture_mod", "ok": rep.get("ok"), "outside_mae": rep.get("outside_mae")})\n'
    '                if rep.get("repaired"):\n',
    1,
)
p.write_text(t, encoding="utf-8")

# runner
r = Path(__file__).resolve().parent / "run_higgs_s01.py"
rt = r.read_text(encoding="utf-8")
rt = rt.replace("from engine import colors, pipeline, spot, texture",
                "from engine import colors, pipeline, texmod, texture")
rt = rt.replace(
    "    trep = spot.lock_ink(tres, plate, ink_p, out_path=clips / f\"{SID}.plate.tex.mp4\")\n"
    "    (clips / f\"{cid}.spot.json\").write_text(json.dumps(trep, indent=1))\n"
    "    print(\"  lock_ink\", {k: trep.get(k) for k in (\"ok\", \"fails\")}, flush=True)\n"
    "    if not trep.get(\"ok\") or not trep.get(\"repaired\"):\n"
    "        raise SystemExit(\"texture spot failed\")\n",
    "    trep = texmod.apply(tres, plate, ink_p, clips / f\"{SID}.plate.tex.mp4\")\n"
    "    (clips / f\"{cid}.texmod.json\").write_text(json.dumps(trep, indent=1))\n"
    "    print(\"  texmod\", {k: trep.get(k) for k in (\"ok\", \"outside_mae\")}, flush=True)\n"
    "    if not trep.get(\"repaired\"):\n"
    "        raise SystemExit(\"texture modulate failed\")\n",
)
r.write_text(rt, encoding="utf-8")
print("pipeline+runner use texmod")
