"""Pilot run of the real pipeline, in two phases, on one job folder (so nothing is paid twice):

  python pilot.py draft  NAME   -> page..rough + a DRAFT reel with the stand-in (no Higgsfield call)
  python pilot.py shot   NAME s01 -> send ONE shot to the video model, spot-check, judge; print reports
  python pilot.py full   NAME   -> the normal unattended run (reuses everything already done)
"""
import json, sys, time
from pathlib import Path
ROOT = Path(r"C:\Users\truet\Claude Projects\Coloring Reel Machine\reelmachine")
sys.path.insert(0, str(ROOT))
from engine import pipeline, spot, judge, colors
from engine import courier as C
from engine.job import Job

PAGE = r"C:\Users\truet\Claude Projects\Coloring Reel Machine\workbench\test_page_bubble_bath.png"
USAGE = "usage: python pilot.py draft|shot|full NAME [shot_id] [model]"
if len(sys.argv) < 3:
    raise SystemExit(USAGE)
mode, name = sys.argv[1], sys.argv[2]
s = pipeline.load_settings()


def show(st):
    cur = [f"{k}:{v['status']}" + (f"({v['msg'][:60]})" if v["msg"] else "") for k, v in st["stages"].items()
           if v["status"] not in ("waiting",)]
    print(time.strftime("%H:%M:%S"), " | ".join(cur[-2:]), flush=True)


last = [0.0]


def on_update(st):
    if time.time() - last[0] > 20:
        last[0] = time.time()
        show(st)


if mode == "draft":
    def nokey():
        raise C.CourierError("pilot: draft phase")
    pipeline.get_key = nokey
    pipeline.notify = lambda *a, **k: None
    s = {**s, "apply_ink_texture": False}  # early-chat draft: sim plate/rough only, no texture pass
    out = pipeline.run(PAGE, s, name=name, on_update=on_update)
    print("DRAFT:", out)
elif mode == "shot":
    if len(sys.argv) < 4:
        raise SystemExit(USAGE + "  (shot mode needs shot_id, e.g. s01)")
    sid = sys.argv[3]
    model = sys.argv[4] if len(sys.argv) > 4 else s["models"][0]
    job = Job(s["jobs_dir"], name)
    clips = job.dir / "clips"
    meta = json.loads((clips / f"{sid}.json").read_text())
    markers = colors.load_markers(ROOT / "markers" / "basic-36.json")
    cap = pipeline._marker_name(markers[meta["markers"][0]]) if meta["markers"] else "matching"
    refs = sorted(Path(s["hand_refs"]).glob("*.[jp][pn]g"))
    ledger = C.Ledger(Path(s["jobs_dir"]) / "ledger.json", s["budget_reel_usd"], s["budget_day_usd"])
    cour = C.Courier(ledger, job.log)
    cid = f"{sid}.{model}"
    st = pipeline._clip_state(job, cid)
    t0 = time.time()
    res = cour.run(job.name, cid, model, clips / f"{sid}.rough.mp4", refs[:4], pipeline.HAND_PROMPT.format(cap=cap),
                   s["elements"], s["resolution"], st)
    print("result", res, "in", round(time.time() - t0), "s; spent", ledger.spent(job.name))
    rep = spot.check(res, clips / f"{sid}.plate.mp4", clips / f"{sid}.masks.npz", out_path=clips / f"{cid}.fixed.mp4")
    (clips / f"{cid}.spot.json").write_text(json.dumps(rep, indent=1))
    job.log({"clip": cid, "step": "spot", "ok": rep["ok"], "fails": rep["fails"]})
    print(json.dumps({k: rep.get(k) for k in ("ok", "fails", "src", "frames_result", "align", "outside_changed", "hand_missing")}, indent=1))
    if rep["ok"] and judge.available():
        v = judge.judge(rep["repaired"], cap)
        job.log({"clip": cid, "step": "judge", **v})
        print("judge", json.dumps(v))
elif mode == "full":
    out = pipeline.run(PAGE, s, name=name, on_update=on_update)
    print("REEL:", out)

