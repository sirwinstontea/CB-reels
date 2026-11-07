"""The whole machine, start to finish, with no human in the loop.

run(page_path, settings) -> path of the finished reel (or raises JobFailed after notifying you).
Every stage updates the job's state.json (the window draws it) and log.jsonl (resume + audit).

Retry ladder per hand shot: model 1 -> checks -> model 2 -> checks -> fallback still shot.
Checks = spot the difference (must pass) + the judge (if a key is set).
A job fails for good (and notifies) only when the page can't be read, too many shots fell
back, or the final file is broken.
"""
import ctypes
import inspect
import json
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from . import colorai, colors, handref, director, editor, final, ink, judge, rough, scene, shading, sound, spot, strokes, texture, texmod
from . import page as pagemod
from .courier import Courier, CourierError, BudgetExceeded, Ledger, get_key, price, cost_report
from .job import Job
from .notify import notify
from .paths import DATA, HOME as ROOT
from .video import Writer, FPS
DEFAULTS = {
    "jobs_dir": str(ROOT / "jobs"),
    # attempts per shot, in order (a model may repeat). Kling 4.0 edit redraws the whole page, so it isn't used.
    "models": ["kling-omni-edit", "kling-omni-edit"],
    "resolution": "720p",
    # Only these framings go to the video model: in a wide shot Omni redraws the small page (spot failed 2/2 on
    # the test reel, $0.64 wasted), so a wide hand shot becomes a free hand-free hold of the page.
    "model_zooms": ["close", "medium"],
    "max_paid_shots": 4,            # 4 x 3 s Omni std = $1.29; more hand shots become holds
    "budget_reel_usd": 1.5,
    "budget_day_usd": 15.0,
    "use_judge": True,
    "elements": [],                 # Higgsfield Element ids for the hand (made once in the web app)
    "sound_dir": str(ROOT / "assets" / "sounds"),
    "desk_dir": str(ROOT / "assets" / "desks"),
    "hand_refs": str(ROOT / "assets" / "hand"),
    "marker_textures": str(ROOT / "assets" / "marker_textures"),
    "apply_ink_texture": True,
    "texture_model": "kling-omni-edit",
    "max_fallback_share": 0.5,
}
# Why it reads this way: the old prompt asked for the reference's sweater cuff,
# so Omni drew a big sleeve and forearm over the page; the nib came out dark green in every shot; the hand
# changed size and grip between frames; skin was smooth and waxy.
HAND_PROMPT = """Replace the flat beige hand shape and the plain marker with a real person's right hand like the
hand in the reference image, holding a real alcohol marker: a fat white barrel with a {cap} cap posted on its
back end and a soft brush nib in the same {cap} colour as the fresh ink. Only the fingers and the back of the
hand are in the picture, coming in from its edge; no sleeve, no cuff. A relaxed grip a few centimetres above
the nib, with the same hand size, pose and grip in every frame. Real skin: fine pores, knuckle creases,
slightly uneven tone, short natural nails, lit by the same soft light as the desk, with the same focus, grain
and colour as the paper, and a soft natural shadow on the paper. The nib stays exactly where the plain
marker's nib is in every frame and moves exactly the same way, gliding smoothly on the paper.
Keep everything else exactly as it is: the page, every black line, every colour of ink, the desk,
the markers in the background, the framing, the camera and the light. Do not add a second hand.
Do not add, remove or recolour anything else."""


class JobFailed(Exception):
    pass



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


def load_settings():
    p = ROOT / "settings.json"
    s = dict(DEFAULTS)
    if p.exists():
        s.update(json.loads(p.read_text(encoding="utf-8")))
    return s


def _keep_awake(on):
    try:   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: the PC may not sleep mid-job (screen may)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def _still(png, seconds, out):
    img = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    with Writer(out, img.shape[1], img.shape[0]) as w:
        for _ in range(int(round(seconds * FPS))):
            w.write(img)
    return out


def _clip_state(job, clip_id):
    st = {}
    for e in job.events():
        if e.get("clip") == clip_id:
            st.update({k: v for k, v in e.items() if k not in ("t", "clip", "step")})
            st["last_step"] = e.get("step")
    return st


def _marker_name(m):
    return m["id"].rsplit("-", 1)[0].replace("_", " ")


def _judged(job, cid, step, fn):
    """(verdict, new spend): the logged verdict for this clip and step if there is one, else a new one."""
    v = next(({k: e[k] for k in e if k not in ("t", "clip", "step", "usd")} for e in job.events()
              if e.get("clip") == cid and e.get("step") == step), None)
    if v is not None:
        return v, 0.0
    v = fn()
    job.log({"clip": cid, "step": step, **v})
    return v, v.get("usd", 0.0)


def _retry_blocked(last, ledger, job, s, model, x, later):
    """Why a paid retry should be skipped (None = go): don't pay twice for a failure that repeats, and
    never let a retry eat the money the remaining shots' first attempts need."""
    if last and last.get("systematic"):
        return "the model changed the page in almost every frame: the same framing fails the same way again"
    need = price(model, x.t1 - x.t0, s["resolution"]) + sum(
        price(s["models"][0], y.t1 - y.t0, s["resolution"]) for y in later)
    if ledger.spent(job)[0] + need > ledger.per_reel + 1e-6:
        return f"a retry would leave too little budget for the {len(later)} shot(s) still to go"
    return None


def run(page_path, settings=None, name=None, on_update=None):
    s = settings or load_settings()
    page_path = Path(page_path)
    name = name or f"{page_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    job = Job(s["jobs_dir"], name)
    d = job.dir
    if not (d / "page.png").exists():
        shutil.copy(page_path, d / "page.png")
    _keep_awake(True)
    stage = "page"

    def upd(st, status=None, msg=None, progress=None):
        job.set(st, status, msg, progress)
        if on_update:
            on_update(job.state)

    try:
        # 1. page
        upd(stage, "running", "finding the shapes to color")
        try:
            pg = pagemod.load_page_gated(d / "page.png")
        except Exception as e:
            raise JobFailed(f"Can't read this page: {e}")
        upd(stage, "passed", f"{len(pg.regions)} shapes", 1)

        # 2. colors
        stage = "colors"
        upd(stage, "running", "choosing markers")
        markers = colors.load_markers(DATA / "markers" / "basic-36.json")
        target = next((p for p in (page_path.with_suffix(".color.png"), d / "target.png") if p.exists()), None)
        plan = None
        if target is not None:
            try:
                plan, gate = colors.plan_from_target(pg, target, markers)
                if plan is None:
                    raise ValueError(gate["reason"])
                upd(stage, msg=f"from {target.name}")
            except Exception as e:
                job.log({"stage": stage, "note": f"target rejected: {e}"})
        color_usd = 0.0
        if plan is None and judge.available():
            try:
                upd(stage, msg="asking Claude to plan the colors")
                plan, color_usd = colorai.plan(pg, markers, cache_path=d / "colorplan.json")
                upd(stage, msg=f"planned by Claude ({len(plan)} shapes)")
            except Exception as e:
                job.log({"stage": stage, "note": f"Claude color plan failed: {e}"})
                plan = None
        if plan is None:
            plan = colors.plan_fallback(pg, markers)
            upd(stage, msg="autumn palette (no color target)")
        upd(stage, "passed", progress=1)

        # 3. strokes
        stage = "strokes"
        upd(stage, "running", "planning every stroke")
        layers = shading.layers_for(pg, plan, markers)     # base, shade, gradients, patterns, glints
        if "layers" in inspect.signature(strokes.plan).parameters:
            tl = strokes.plan(pg, plan, markers, layers=layers)
        else:                                                # strokes without layer support yet
            tl = strokes.plan(pg, plan, markers)
        upd(stage, "passed", f"{tl.duration / 60:.0f} min session, {int(tl.stroke.max()) + 1} strokes", 1)

        # 4-5. ink + rough clips (one chronological pass)
        stage = "rough"
        desks = {}
        for a in ("A", "B"):
            p = Path(s["desk_dir"]) / f"{a}.jpg"
            if not p.exists():
                scene.make_placeholder_desk(p, a, seed=ord(a))
            desks[a] = scene.load_desk(p)
        shots = director.plan_shots(tl, pg, markers, desks)
        (d / "shots.json").write_text(json.dumps([x.to_dict() for x in shots], indent=1))
        clips = d / "clips"
        if all((clips / f"{x.id}.end.png").exists() for x in shots):
            upd("ink", "passed", "already rendered", 1)
            upd(stage, "passed", "already rendered", 1)
        else:
            upd("ink", "running", "laying ink")
            upd(stage, "running", f"{len(shots)} shots")
            cv = ink.InkCanvas(pg, markers)
            rough.render(tl, pg, cv, markers, shots, desks, clips,
                         progress=lambda f, m: upd(stage, progress=f, msg=m))
            upd("ink", "passed", progress=1)
            upd(stage, "passed", f"{len(shots)} shots", 1)

        # 6-8. courier -> spot -> judge, per hand shot, with the retry ladder.
        # Paid shots: close/medium only, at most max_paid_shots; other hand shots become free holds (a still
        # of the page as it is at the shot's end: no ink appears without a hand).
        hand_shots = [x for x in shots if x.hand and x.zoom in s["model_zooms"]][:s["max_paid_shots"]]
        holds = [x for x in shots if x.hand and x.id not in {y.id for y in hand_shots}]
        dry = False
        try:
            key = get_key()
        except CourierError:
            key = None
        if key is None:
            dry = True
            upd("courier", "skipped", "no Higgsfield key: making a DRAFT with the stand-in hand")
            notify("Coloring Reel Machine", "No Higgsfield key saved, so this reel is a draft with the stand-in hand.")
        ledger = Ledger(Path(s["jobs_dir"]) / "ledger.json", s["budget_reel_usd"], s["budget_day_usd"])
        courier = None if dry else Courier(ledger, job.log, key=key)

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
                rep = texmod.apply(res, plate_p, ink_p, clips / f"{x.id}.plate.tex.mp4")
                (clips / f"{cid}.spot.json").write_text(json.dumps(rep, indent=1))
                job.log({"clip": cid, "step": "texture_mod", "ok": rep.get("ok"), "outside_mae": rep.get("outside_mae")})
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
            upd(stage, "passed", progress=1)
        elif s.get("apply_ink_texture"):
            upd("texture", "skipped", "draft / no Higgsfield key")
        use_judge = s["use_judge"] and judge.available() and not dry
        if not use_judge:
            upd("judge", "skipped", "no Anthropic key" if s["use_judge"] and not dry else "off")
        refs = sorted(Path(s["hand_refs"]).glob("*.[jp][pn]g")) if Path(s["hand_refs"]).exists() else []
        final_clips, budget_hit, judge_usd = {}, False, color_usd
        for i, x in enumerate(hand_shots):
            stem = clips / x.id
            meta = json.loads((clips / f"{x.id}.json").read_text())
            cap = _marker_name(markers[meta["markers"][0]]) if meta["markers"] else "matching"
            # the model copies the reference's marker, so its cap is tinted to this shot's marker
            shot_refs = [handref.tinted(r, markers[meta["markers"][0]]["cap"], clips / "refs") if meta["markers"] else r
                         for r in refs[:4]]
            if dry:
                final_clips[x.id] = clips / f"{x.id}.rough.mp4"
                continue
            good, last_fail = None, None
            models = [m for m in s["models"][:3] if refs or m != "genjutsu"] or ["kling-omni-edit"]
            for attempt, model in enumerate(models):      # genjutsu needs 1+ hand photos in assets/hand
                n_before = models[:attempt].count(model)
                cid = f"{x.id}.{model}" + (f".r{n_before + 1}" if n_before else "")
                st = _clip_state(job, cid)
                have = st.get("path") and Path(st["path"]).exists()     # already paid for and downloaded
                if budget_hit and not have:
                    continue
                if attempt and not have:
                    why = _retry_blocked(last_fail, ledger, job.name, s, model, x, hand_shots[i + 1:])
                    if why:
                        job.log({"clip": cid, "step": "no_retry", "msg": why})
                        break
                upd("courier", "running" if attempt == 0 else "retrying",
                    f"shot {i + 1}/{len(hand_shots)} -> {model}", i / len(hand_shots))
                try:
                    if have:
                        res = Path(st["path"])
                    else:
                        res = courier.run(job.name, cid, model, f"{stem}.rough.mp4", shot_refs,
                                          HAND_PROMPT.format(cap=cap), s["elements"], s["resolution"], st)
                except BudgetExceeded as e:
                    if not budget_hit:
                        notify("Budget reached", str(e))
                    budget_hit = True
                    job.log({"clip": cid, "step": "budget", "msg": str(e)})
                    continue
                except CourierError as e:
                    job.log({"clip": cid, "step": "error", "msg": str(e)})
                    continue
                upd("spot", "running", f"checking shot {x.id} ({model})")
                rep = spot.check(res, f"{stem}.plate.mp4", f"{stem}.masks.npz", out_path=clips / f"{cid}.fixed.mp4")
                (clips / f"{cid}.spot.json").write_text(json.dumps(rep, indent=1))
                job.log({"clip": cid, "step": "spot", "ok": rep["ok"], "fails": rep["fails"]})
                if not rep["ok"]:
                    upd("spot", "retrying", f"{x.id}: {rep['fails'][0]}")
                    oc = rep.get("outside_changed") or {}
                    last_fail = {"kind": "spot", "systematic": oc.get("bad_frames", 0) >= 0.8 * max(1, rep.get("frames_plate", 1))}
                    continue
                if use_judge:
                    upd("judge", "running", f"looking at shot {x.id}")
                    try:
                        # a clip is judged once: on resume the first verdict stands (the judge isn't deterministic)
                        v, u = _judged(job, cid, "judge", lambda: judge.judge(rep["repaired"], cap))
                        judge_usd += u
                        if not v["ok"]:
                            # verdicts are noisy (Opus gave the same s02 clip 6 and 5-severe), and a second
                            # opinion on other frames costs ~$0.006 vs $0.32 for a new clip: retry only if it agrees
                            v, u = _judged(job, cid, "judge2", lambda: judge.judge(rep["repaired"], cap, offset=0.5))
                            judge_usd += u
                        if not v["ok"]:
                            upd("judge", "retrying", f"{x.id}: {'; '.join(v['problems'][:2])}")
                            last_fail = {"kind": "judge", "systematic": False}
                            continue
                    except RuntimeError as e:
                        job.log({"clip": cid, "step": "judge_error", "msg": str(e)})   # judge down: don't block
                good = Path(rep["repaired"])
                break
            if good is None:
                job.log({"clip": x.id, "step": "fallback"})
                good = _still(clips / f"{x.id}.end.png", x.t1 - x.t0, clips / f"{x.id}.still.mp4")
            final_clips[x.id] = good
            job.state["usd"] = round(ledger.spent(job.name)[0] + judge_usd, 4)   # ledger = source of truth
            job.add_cost(0)
        fell_back = sum(1 for p in final_clips.values() if p.name.endswith(".still.mp4"))
        if not dry:
            rep_c = cost_report(job, courier, color_usd, final_clips)
            upd("courier", "passed", f"{len(hand_shots) - fell_back}/{len(hand_shots)} shots with a real hand, "
                f"${rep_c['higgsfield_usd']:.2f} Higgsfield", 1)
            upd("spot", "passed" if fell_back < len(hand_shots) else "failed", progress=1)
            if use_judge:
                upd("judge", "passed", progress=1)
        else:
            upd("spot", "skipped", "draft")
        if hand_shots and fell_back / len(hand_shots) > s["max_fallback_share"]:
            raise JobFailed(f"{fell_back} of {len(hand_shots)} shots failed every check"
                            + (" (budget reached)" if budget_hit else ""))
        for x in shots:
            if not x.hand:
                final_clips[x.id] = clips / f"{x.id}.plate.mp4"
        for x in holds:          # ".still.mp4": the sound stage gives it no stroke sounds
            final_clips[x.id] = _still(clips / f"{x.id}.end.png", x.t1 - x.t0, clips / f"{x.id}.hold.still.mp4")

        # 9. sound
        stage = "sound"
        upd(stage, "running", "placing a sound on every stroke")
        bank = sound.Bank(s["sound_dir"])
        wavs = {}
        for x in shots:
            hand = x.hand and not final_clips[x.id].name.endswith(".still.mp4")
            xs = x if hand else director.Shot(**{**x.to_dict(), "hand": False})
            wavs[x.id] = sound.render(tl, xs, bank, clips / f"{x.id}.wav", pg.px_per_mm)
        upd(stage, "passed", "sound bank" if bank.ok else "procedural marker sound", 1)

        # 10. cut
        stage = "cut"
        upd(stage, "running", "editing")
        order = sorted(shots, key=lambda x: x.t0)
        out = d / ("reel.DRAFT.mp4" if dry else "reel.mp4")
        editor.cut([final_clips[x.id] for x in order], [wavs[x.id] for x in order], out,
                   progress=lambda f, m: upd(stage, progress=f, msg=m))
        upd(stage, "passed", out.name, 1)

        # 11. final check
        stage = "final"
        upd(stage, "running", "checking the file")
        rep = final.check(out, clips / f"{order[-1].id}.end.png")
        (d / "final.json").write_text(json.dumps(rep, indent=1))
        if not rep["ok"]:
            raise JobFailed("Final check: " + "; ".join(rep["fails"]))
        upd(stage, "passed", f"{rep['duration']:.0f}s, ${job.state['usd']:.2f}", 1)
        job.finish(True)
        if on_update:
            on_update(job.state)
        notify("Reel ready" + (" (draft)" if dry else ""), f"{name}: {rep['duration']:.0f}s, cost ${job.state['usd']:.2f}")
        return out
    except JobFailed as e:
        job.set(stage, "failed", str(e))
        job.finish(False, str(e))
        if on_update:
            on_update(job.state)
        notify(f"Stopped at: {job.state['stages'][stage]['label']}", str(e))
        raise
    except Exception as e:
        job.log({"stage": stage, "crash": traceback.format_exc()})
        job.set(stage, "failed", f"crash: {e}")
        job.finish(False, f"crash: {e}")
        if on_update:
            on_update(job.state)
        notify(f"Crashed at: {job.state['stages'][stage]['label']}", str(e))
        raise
    finally:
        _keep_awake(False)
