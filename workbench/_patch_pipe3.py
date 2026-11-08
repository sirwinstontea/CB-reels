from pathlib import Path

block = '''
        # 5b. mask-locked ink texture on plates (then rebuild rough for hand shots).
        if s.get("apply_ink_texture") and not dry and courier is not None:
            stage = "texture"
            upd(stage, "running", "Higgs texture on ink masks")
            tex_root = Path(s.get("marker_textures") or (ROOT / "assets" / "marker_textures"))
            for i, x in enumerate(shots):
                plate_p = clips / f"{x.id}.plate.mp4"
                ink_p = clips / f"{x.id}.inkmask.npz"
                if not plate_p.exists() or not ink_p.exists():
                    continue
                meta = json.loads((clips / f"{x.id}.json").read_text())
                idxs = list(meta.get("markers") or [])
                if not idxs:
                    idxs = sorted({int(tl.marker[j]) for j in range(len(tl.t))
                                   if tl.down[j] and x.t0 <= float(tl.t[j]) < x.t1})
                meds = texture.shot_mediums(markers, idxs)
                want_blend = len(set(idxs)) > 1
                refs = texture.refs_for(meds, tex_root, want_blend=want_blend)
                if not refs:
                    job.log({"clip": x.id, "step": "texture_skip", "msg": "no texture refs"})
                    continue
                cid = f"{x.id}.texture"
                st = _clip_state(job, cid)
                have = st.get("path") and Path(st["path"]).exists()
                upd(stage, "running", f"texture {i + 1}/{len(shots)}", i / max(1, len(shots)))
                try:
                    res = Path(st["path"]) if have else courier.run(
                        job.name, cid, s.get("texture_model") or "kling-omni-edit",
                        f"{x.id}.plate.mp4", refs, texture.prompt(meds, want_blend=want_blend),
                        [], s["resolution"], st)
                except (BudgetExceeded, CourierError) as e:
                    job.log({"clip": cid, "step": "texture_error", "msg": str(e)})
                    continue
                rep = spot.lock_ink(res, plate_p, ink_p, out_path=clips / f"{x.id}.plate.tex.mp4")
                (clips / f"{cid}.spot.json").write_text(json.dumps(rep, indent=1))
                job.log({"clip": cid, "step": "texture_spot", "ok": rep.get("ok"), "fails": rep.get("fails")})
                if rep.get("ok") and rep.get("repaired"):
                    flat = clips / f"{x.id}.plate.flat.mp4"
                    if not flat.exists():
                        plate_p.replace(flat)
                    else:
                        plate_p.unlink(missing_ok=True)
                    Path(rep["repaired"]).replace(plate_p)
                    if x.hand:
                        _rebuild_rough_from_plate(plate_p, clips / f"{x.id}.json", markers,
                                                  clips / f"{x.id}.rough.mp4")
            upd(stage, "passed", progress=1)
        elif s.get("apply_ink_texture"):
            upd("texture", "skipped", "draft / no Higgsfield key")
'''

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "pipeline.py"
t = p.read_text(encoding="utf-8")
if 'stage = "texture"' in t:
    print("pipe3 skip")
else:
    anchor = "        courier = None if dry else Courier(ledger, job.log, key=key)\n"
    if anchor not in t:
        raise SystemExit("courier anchor missing")
    t = t.replace(anchor, anchor + block, 1)
    p.write_text(t, encoding="utf-8")
    print("pipe3 ok")
