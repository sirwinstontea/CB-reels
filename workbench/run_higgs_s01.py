"""Run mask-locked ink texture + hand Omni on one shot of an existing job, then open the result."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\truet\Claude Projects\Coloring Reel Machine\reelmachine")
sys.path.insert(0, str(ROOT))

from engine import colors, pipeline, spot, texture
from engine import courier as C
from engine.job import Job
from engine.paths import HOME as RM

JOB = sys.argv[1] if len(sys.argv) > 1 else "flat-ink-preview"
SID = sys.argv[2] if len(sys.argv) > 2 else "s01"


def main():
    s = pipeline.load_settings()
    job = Job(s["jobs_dir"], JOB)
    clips = job.dir / "clips"
    markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
    plate = clips / f"{SID}.plate.mp4"
    ink_p = clips / f"{SID}.inkmask.npz"
    rough = clips / f"{SID}.rough.mp4"
    meta = json.loads((clips / f"{SID}.json").read_text())
    idxs = list(meta.get("markers") or [])
    meds = texture.shot_mediums(markers, idxs)
    want_blend = len(set(idxs)) > 1
    tex_root = Path(s.get("marker_textures") or (RM / "assets" / "marker_textures"))
    refs = texture.refs_for(meds, tex_root, want_blend=want_blend)
    ledger = C.Ledger(Path(s["jobs_dir"]) / "ledger.json", s["budget_reel_usd"], s["budget_day_usd"])
    cour = C.Courier(ledger, job.log)
    model = s.get("texture_model") or "kling-omni-edit"

    print("1/3 texture edit on plate…", flush=True)
    cid = f"{SID}.texture"
    st = pipeline._clip_state(job, cid)
    t0 = time.time()
    if st.get("path") and Path(st["path"]).exists():
        tres = Path(st["path"])
        print("  resume", tres)
    else:
        tres = cour.run(job.name, cid, model, str(plate), refs,
                        texture.prompt(meds, want_blend=want_blend), [], s["resolution"], st)
    print("  texture", tres, "in", round(time.time() - t0), "s", flush=True)
    trep = spot.lock_ink(tres, plate, ink_p, out_path=clips / f"{SID}.plate.tex.mp4")
    (clips / f"{cid}.spot.json").write_text(json.dumps(trep, indent=1))
    print("  lock_ink", {k: trep.get(k) for k in ("ok", "fails")}, flush=True)
    if not trep.get("ok") or not trep.get("repaired"):
        raise SystemExit("texture spot failed")
    flat = clips / f"{SID}.plate.flat.mp4"
    if not flat.exists():
        plate.replace(flat)
    else:
        plate.unlink(missing_ok=True)
    Path(trep["repaired"]).replace(plate)
    pipeline._rebuild_rough_from_plate(plate, clips / f"{SID}.json", markers, rough)

    print("2/3 hand Omni on rough…", flush=True)
    cap = pipeline._marker_name(markers[idxs[0]]) if idxs else "matching"
    hrefs = sorted(Path(s["hand_refs"]).glob("*.[jp][pn]g"))
    from engine import handref
    shot_refs = [handref.tinted(r, markers[idxs[0]]["cap"], clips / "refs") if idxs else r for r in hrefs[:4]]
    hmodel = s["models"][0]
    hcid = f"{SID}.{hmodel}"
    hst = pipeline._clip_state(job, hcid)
    t0 = time.time()
    if hst.get("path") and Path(hst["path"]).exists():
        hres = Path(hst["path"])
    else:
        hres = cour.run(job.name, hcid, hmodel, str(rough), shot_refs,
                        pipeline.HAND_PROMPT.format(cap=cap), s["elements"], s["resolution"], hst)
    print("  hand", hres, "in", round(time.time() - t0), "s; spent", ledger.spent(job.name), flush=True)
    hrep = spot.check(hres, plate, clips / f"{SID}.masks.npz", out_path=clips / f"{hcid}.fixed.mp4")
    (clips / f"{hcid}.spot.json").write_text(json.dumps(hrep, indent=1))
    print("  hand spot", {k: hrep.get(k) for k in ("ok", "fails", "outside_changed")}, flush=True)
    out = Path(hrep["repaired"]) if hrep.get("ok") and hrep.get("repaired") else hres
    print("3/3 result:", out, flush=True)
    return out


if __name__ == "__main__":
    p = main()
    print("OPEN", p)
