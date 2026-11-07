"""Continue after texture Omni: force mask composite, rebuild rough, run hand, open result."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\truet\Claude Projects\Coloring Reel Machine\reelmachine")
sys.path.insert(0, str(ROOT))

from engine import colors, handref, pipeline, spot
from engine import courier as C
from engine.job import Job

JOB, SID = "flat-ink-preview", "s01"


def main():
    s = pipeline.load_settings()
    job = Job(s["jobs_dir"], JOB)
    clips = job.dir / "clips"
    markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
    plate = clips / f"{SID}.plate.mp4"
    ink_p = clips / f"{SID}.inkmask.npz"
    rough = clips / f"{SID}.rough.mp4"
    tres = clips / f"{SID}.texture.r2.out.mp4"
    meta = json.loads((clips / f"{SID}.json").read_text())
    idxs = list(meta.get("markers") or [])

    print("lock_ink composite…", flush=True)
    trep = spot.lock_ink(tres, plate, ink_p, out_path=clips / f"{SID}.plate.tex.mp4")
    print("  ", {k: trep.get(k) for k in ("ok", "fails", "repaired")}, flush=True)
    if not trep.get("repaired"):
        raise SystemExit("no repaired texture")
    flat = clips / f"{SID}.plate.flat.mp4"
    if not flat.exists():
        plate.replace(flat)
    else:
        plate.unlink(missing_ok=True)
    Path(trep["repaired"]).replace(plate)
    pipeline._rebuild_rough_from_plate(plate, clips / f"{SID}.json", markers, rough)

    print("hand Omni…", flush=True)
    cap = pipeline._marker_name(markers[idxs[0]]) if idxs else "matching"
    hrefs = sorted(Path(s["hand_refs"]).glob("*.[jp][pn]g"))
    shot_refs = [handref.tinted(r, markers[idxs[0]]["cap"], clips / "refs") if idxs else r for r in hrefs[:4]]
    ledger = C.Ledger(Path(s["jobs_dir"]) / "ledger.json", s["budget_reel_usd"], s["budget_day_usd"])
    cour = C.Courier(ledger, job.log)
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
    print("  spot", {k: hrep.get(k) for k in ("ok", "fails")}, flush=True)
    out = Path(hrep["repaired"]) if hrep.get("repaired") else hres
    # also copy texture raw + fixed for side-by-side
    preview = job.dir / "higgs_s01_preview.mp4"
    import shutil
    shutil.copy2(out, preview)
    print("OPEN", preview)
    return preview


if __name__ == "__main__":
    print(main())
