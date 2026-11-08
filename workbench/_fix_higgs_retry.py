from pathlib import Path

p = Path(__file__).resolve().parent / "run_higgs_s01.py"
t = p.read_text(encoding="utf-8")
old = """    cid = f"{SID}.texture"
    st = pipeline._clip_state(job, cid)
    t0 = time.time()
    if st.get("path") and Path(st["path"]).exists():
        tres = Path(st["path"])
        print("  resume", tres)
    else:
        tres = cour.run(job.name, cid, model, str(plate), refs,
                        texture.prompt(meds, want_blend=want_blend), [], s["resolution"], st)
"""
new = """    cid = f"{SID}.texture"
    st = pipeline._clip_state(job, cid)
    # failed Omni jobs keep old image_urls in the log; never reuse those
    if st.get("status") in ("failed", "nsfw", "canceled"):
        for k in ("images", "request_id", "status_url", "path", "idempotency_key"):
            st.pop(k, None)
        cid = f"{SID}.texture.r2"
        st = pipeline._clip_state(job, cid)
    t0 = time.time()
    if st.get("path") and Path(st["path"]).exists():
        tres = Path(st["path"])
        print("  resume", tres)
    else:
        print("  refs", [r.name for r in refs], flush=True)
        tres = cour.run(job.name, cid, model, str(plate), refs,
                        texture.prompt(meds, want_blend=want_blend), [], s["resolution"], st)
"""
if old not in t:
    raise SystemExit("texture block missing")
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("retry fixed")
