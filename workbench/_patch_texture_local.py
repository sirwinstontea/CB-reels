from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "pipeline.py"
t = p.read_text(encoding="utf-8")
# replace the paid Omni texture loop with local ref texmod
start = t.find("        # 5b. mask-locked ink texture")
if start < 0:
    raise SystemExit("5b missing")
# find next major section: hand_shots loop already after; end at "        dry_tex" or "        use_judge"
# Actually after our block comes "        use_judge" - find "        if s.get(\"apply_ink_texture\")"
end = t.find("        use_judge = s[\"use_judge\"]", start)
if end < 0:
    end = t.find("        refs = sorted", start)
if end < 0:
    raise SystemExit("end missing")

new = '''        # 5b. ref-driven ink texture (local; Omni residual leaked ghosts — do not use for texture).
        if s.get("apply_ink_texture"):
            stage = "texture"
            upd(stage, "running", "ref grain on ink masks")
            for i, x in enumerate(shots):
                plate_p = clips / f"{x.id}.plate.mp4"
                ink_p = clips / f"{x.id}.inkmask.npz"
                if not plate_p.exists() or not ink_p.exists():
                    continue
                tex_out = clips / f"{x.id}.plate.tex.mp4"
                rep = texmod.apply_refs(plate_p, ink_p, tex_out,
                                        tex_root=Path(s.get("marker_textures") or (ROOT / "assets" / "marker_textures")) / "alcohol")
                job.log({"clip": x.id, "step": "texture_mod", **{k: rep.get(k) for k in ("ok", "outside_mae", "mode")}})
                if rep.get("repaired"):
                    flat = clips / f"{x.id}.plate.flat.mp4"
                    if not flat.exists():
                        plate_p.replace(flat)
                    else:
                        plate_p.unlink(missing_ok=True)
                    Path(rep["repaired"]).replace(plate_p)
                    if x.hand:
                        _rebuild_rough_from_plate(plate_p, clips / f"{x.id}.json", markers,
                                                  clips / f"{x.id}.rough.mp4")
            upd(stage, "passed", "local ref texture", 1)

'''
t = t[:start] + new + t[end:]
p.write_text(t, encoding="utf-8")
print("pipeline texture = local refs")
