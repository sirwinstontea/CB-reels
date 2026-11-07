"""The courier: sends rough clips to Higgsfield and fetches the results, with no human in the loop.

Flow per clip: estimate cost -> budget check -> presigned upload -> submit (idempotent) -> poll -> download.
Every step is written to the job log, so a restarted job never pays for the same clip twice.
The API key lives in Windows Credential Manager (service "reelmachine-higgsfield"), never in a file.
"""
import json
import math
import time
import uuid
from pathlib import Path

import requests

API = "https://api.higgsfield.ai"
KEYRING_SERVICE = "reelmachine-higgsfield"
TERMINAL = {"completed", "failed", "nsfw", "canceled"}

MODELS = {
    # name: (endpoint path, function building the request body)
    # Kling 4.0 edit: duration is a whole number of seconds (3-30) and should equal the clip's length.
    "kling-4-edit": ("kling-video/v4.0/video-edit",
                     lambda video, images, prompt, elements, res, dur: {
                         "video_url": video, "prompt": prompt, "image_urls": images[:4],
                         "elements": [str(e) for e in elements[:4]], "resolution": "1080p" if res == "1080p" else "720p",
                         "aspect_ratio": "9:16", "duration": int(min(30, max(3, round(dur))))}),
    # Genjutsu object swap (motion transfer rebuilds the whole scene, so it is not used)
    "genjutsu": ("higgsfield/genjutsu/object-swap/v1.0",
                 lambda video, images, prompt, elements, res, dur: {
                     "video_url": video, "image_urls": images[:8], "prompt": prompt, "resolution": res}),
    # Kling Omni (O1) edit keeps the page; "std" = 720p ("pro" is the API's default). Source must be 3-10 s.
    "kling-omni-edit": ("kling-video/omni/video-edit", lambda *a: _omni(*a)),
    "kling-omni-edit-pro": ("kling-video/omni/video-edit", lambda *a: _omni(*a, mode="pro")),
    # Kling O3 (3.0 Omni) edit: same request shape, newer model, same list price; source 3-15.5 s. Untested.
    "kling-o3-edit": ("kling-video/o3/video-edit", lambda *a: _omni(*a)),
}


def _omni(video, images, prompt, elements, res, dur, mode=None):
    return {"mode": mode or ("std" if res == "720p" else "pro"), "prompt": prompt, "video_urls": [video],
            "image_urls": images[:4], "elements": [str(e) for e in elements[:4]]}


# What a clip costs. Measured 2026-10-28 with the free GET /models/usage (Higgsfield's own per-model cost):
# Omni std edit $0.3213 per 3.0 s clip = list $0.126/s less this account's 15% discount (also shown by
# /estimate on other models); Kling 4.0 edit $0.4794 = $0.094/s x 0.85 x (3 s out + 3 s in).
# Billing is on whole seconds of the 3.000 s source (the 3.04 s / 24 fps output didn't round up to 4),
# so rough clips must be exact whole seconds. Unverified prices are guessed high (pro, O3).
DISCOUNT = 0.85
MIN_S = {"kling-omni-edit": 3, "kling-omni-edit-pro": 3, "kling-o3-edit": 3, "genjutsu": 4, "kling-4-edit": 3}


def billed_s(dur):
    return max(1, math.ceil(dur - 0.02))      # 3.000 s (or a 3.01 s container rounding) bills 3 s


PRICES = {
    "kling-4-edit": lambda dur, res: (0.141 if res == "1080p" else 0.094) * DISCOUNT * 2 * max(3, billed_s(dur)),
    "genjutsu": lambda dur, res: {"480p": 0.318, "720p": 0.681, "1080p": 1.632}.get(res, 0.681) * DISCOUNT * billed_s(dur),
    "kling-omni-edit": lambda dur, res: (0.126 if res == "720p" else 0.168) * DISCOUNT * billed_s(dur),
    "kling-omni-edit-pro": lambda dur, res: 0.168 * DISCOUNT * billed_s(dur),
    "kling-o3-edit": lambda dur, res: (0.126 if res == "720p" else 0.168) * DISCOUNT * billed_s(dur),
}


def price(model, dur, res="720p"):
    return round(PRICES[model](dur, res), 4)


def cost_report(job, courier=None, color_usd=0.0, final_clips=None):
    """Writes <job>/cost.json: every paid clip (model, $, what became of it), Claude spend, totals, and
    Higgsfield's own cost figure for the job's hours (free; account-wide, so another job in the same hour
    shows up too). Works from the job log alone, so it can be run again on any finished job."""
    from .video import probe
    used = {Path(p).name for p in (final_clips or {}).values()}
    clips, claude, t_paid = {}, 0.0, []
    for e in job.events():
        c, st = e.get("clip"), e.get("step")
        if st in ("judge", "judge2"):
            claude += e.get("usd") or 0.0
        if not c or "." not in c:
            continue
        r = clips.setdefault(c, {"model": None, "usd": 0.0, "paid": False})
        if st == "submit":
            r["model"] = e.get("model")
        elif st == "queued":
            r.update(usd=e.get("usd", 0.0), paid=True)
            t_paid.append(e["t"])
        elif st == "finished" and e.get("status") in ("failed", "nsfw", "canceled"):
            r.update(usd=0.0, paid=False, outcome=e["status"] + " (not charged)")
        elif st == "spot":
            r["spot"] = "ok" if e.get("ok") else (e.get("fails") or ["failed"])[0]
        elif st in ("judge", "judge2"):
            r[st] = e.get("score")
        elif st in ("budget", "no_retry"):
            r["outcome"] = "skipped: " + e.get("msg", st)
    for c, r in clips.items():
        r["used"] = f"{c}.fixed.mp4" in used
        if r["paid"]:
            rough = job.dir / "clips" / f"{c.split('.')[0]}.rough.mp4"
            r["billed_s"] = billed_s(probe(rough)["duration"]) if rough.exists() else None
            if r["billed_s"] and r["model"] in PRICES:      # today's real price (old logs hold inflated estimates)
                r["usd_logged"], r["usd"] = r["usd"], price(r["model"], r["billed_s"])
    paid = [r for r in clips.values() if r["paid"]]
    hf = sum(r["usd"] for r in paid)
    good_s = sum(r.get("billed_s") or 0 for r in paid if r["used"])
    rep = {"higgsfield_usd": round(hf, 4), "higgsfield_wasted_usd": round(sum(r["usd"] for r in paid if not r["used"]), 4),
           "paid_clips": len(paid), "used_clips": sum(r["used"] for r in paid), "hand_seconds": good_s,
           "usd_per_hand_second": round(hf / good_s, 4) if good_s else None,
           "claude_usd": {"color_plan": round(color_usd, 4), "judge": round(claude, 4)},
           "total_usd": round(hf + color_usd + claude, 4), "clips": clips}
    if courier is not None and t_paid:
        try:
            iso = lambda t: time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime(t))
            rows = courier.usage(iso(min(t_paid)), iso(time.time() + 3600))
            ours = {ep for ep, _ in MODELS.values()}
            rep["higgsfield_reported"] = {"usd": round(sum(float(x["cost"] or 0) for x in rows if x["model"] in ours), 4),
                                          "rows": rows, "note": "Higgsfield /models/usage for these hours, all jobs"}
        except CourierError as e:
            rep["higgsfield_reported"] = {"error": str(e)[:200]}
    (job.dir / "cost.json").write_text(json.dumps(rep, indent=1))
    return rep


class CourierError(Exception):
    """Something the courier can't fix by waiting; the job decides whether to retry or fall back."""


class BudgetExceeded(CourierError):
    pass


def get_key():
    import keyring
    key = keyring.get_password(KEYRING_SERVICE, "api")
    if not key:
        raise CourierError("No Higgsfield key saved. Run: python setkey.py")
    return key


def save_key(key):
    import keyring
    keyring.set_password(KEYRING_SERVICE, "api", key.strip())


class Ledger:
    """Spend per job and per day, kept in a small JSON file next to the jobs."""

    def __init__(self, path, per_reel_usd, per_day_usd):
        self.path = Path(path)
        self.per_reel = per_reel_usd
        self.per_day = per_day_usd
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=1))

    def spent(self, job):
        day = time.strftime("%Y-%m-%d")
        return self.data.get("jobs", {}).get(job, 0.0), self.data.get("days", {}).get(day, 0.0)

    def check(self, job, usd):
        j, d = self.spent(job)
        if j + usd > self.per_reel:
            raise BudgetExceeded(f"reel budget ${self.per_reel:.2f} would be exceeded (spent ${j:.2f}, next ${usd:.2f})")
        if d + usd > self.per_day:
            raise BudgetExceeded(f"daily budget ${self.per_day:.2f} would be exceeded (spent ${d:.2f}, next ${usd:.2f})")

    def add(self, job, usd):
        day = time.strftime("%Y-%m-%d")
        self.data.setdefault("jobs", {})[job] = self.data.get("jobs", {}).get(job, 0.0) + usd
        self.data.setdefault("days", {})[day] = self.data.get("days", {}).get(day, 0.0) + usd
        self._save()

    def refund(self, job, usd):
        self.add(job, -usd)


class Courier:
    def __init__(self, ledger, log, key=None, poll_s=15, timeout_s=3600):
        self.key = key or get_key()
        self.ledger = ledger
        self.log = log            # callable(event: dict)
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Key {self.key}"

    def _req(self, method, url, tries=5, **kw):
        for i in range(tries):
            try:
                r = self.s.request(method, url, timeout=60, **kw)
            except requests.RequestException as e:
                if i == tries - 1:
                    raise CourierError(f"network: {e}")
                time.sleep(min(60, 5 * 2 ** i))
                continue
            if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(min(60, 5 * 2 ** i))
                continue
            if r.status_code >= 400:
                raise CourierError(f"{method} {url} -> {r.status_code}: {r.text[:300]}")
            return r.json() if r.content else {}
        raise CourierError("unreachable")

    def upload(self, path, content_type):
        info = self._req("POST", f"{API}/files/generate-upload-url", json={"content_type": content_type})
        with open(path, "rb") as f:   # the presigned URL must not see our credentials: plain requests.put
            r = requests.put(info["upload_url"], data=f, headers=info.get("upload_headers", {}), timeout=600)
        if r.status_code >= 400:
            raise CourierError(f"upload failed {r.status_code}: {r.text[:200]}")
        return info["public_url"]

    def estimate(self, model, endpoint, body, dur, resolution):
        """Our own price (the API's estimate is prose and sometimes a 500); the API's text is logged."""
        usd = price(model, dur, resolution)
        try:
            res = self._req("POST", f"{API}/estimate/{endpoint}", json=body, tries=2)
            if isinstance(res.get("usd"), (int, float)):
                usd = max(usd, float(res["usd"]))
            return usd, res.get("pricing_description", "")
        except CourierError as e:
            return usd, f"estimate unavailable: {str(e)[:120]}"

    def run(self, job, clip_id, model, video_path, image_paths, prompt, elements=(), resolution="720p",
            state=None):
        """Send one clip and return the local path of the result. `state` is this clip's saved log entry."""
        from .video import probe
        endpoint, build = MODELS[model]
        state = state or {}
        if state.get("request_id") and state.get("status") not in TERMINAL - {"completed"}:
            status_url = state["status_url"]          # resume: already paid for, wait / fetch again
            usd = state.get("usd", 0.0)
        else:
            dur = probe(video_path)["duration"]
            if dur < MIN_S.get(model, 0) - 0.02:          # the API would reject it: don't upload for nothing
                raise CourierError(f"{model} needs a source of at least {MIN_S[model]} s, got {dur:.2f} s")
            images = state.get("images") or [self.upload(p, "image/png" if str(p).endswith(".png") else "image/jpeg") for p in image_paths]
            video = self.upload(video_path, "video/mp4")
            body = build(video, images, prompt, list(elements), resolution, dur)
            usd, note = self.estimate(model, endpoint, body, dur, resolution)
            self.log({"clip": clip_id, "step": "estimate", "usd": usd, "note": note})
            self.ledger.check(job, usd)
            # reuse the key only if the last submit may have gone through without us seeing a request id
            idem = state.get("idempotency_key") if not state.get("request_id") else None
            idem = idem or str(uuid.uuid4())
            self.log({"clip": clip_id, "step": "submit", "model": model, "usd": usd, "idempotency_key": idem, "images": images})
            res = self._req("POST", f"{API}/{endpoint}", json=body, headers={"Idempotency-Key": idem})
            status_url = res["status_url"]
            self.ledger.add(job, usd)
            self.log({"clip": clip_id, "step": "queued", "request_id": res["request_id"], "status_url": status_url, "usd": usd})
        t0 = time.time()
        while True:
            st = self._req("GET", status_url)
            status = st.get("status")
            if status in TERMINAL:
                break
            if time.time() - t0 > self.timeout_s:
                raise CourierError(f"clip {clip_id} still {status} after {self.timeout_s}s")
            time.sleep(self.poll_s)
        self.log({"clip": clip_id, "step": "finished", "status": status})
        if status != "completed":
            if status in ("failed", "nsfw", "canceled"):
                self.ledger.refund(job, usd)          # Higgsfield doesn't charge these
            raise CourierError(f"clip {clip_id} ended {status}: {st.get('error', '')}")
        url = (st.get("video") or {}).get("url") or (st.get("videos") or [{}])[0].get("url")
        if not url:
            raise CourierError(f"clip {clip_id} completed without a video url: {json.dumps(st)[:200]}")
        out = Path(video_path).with_name(f"{clip_id}.out.mp4")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        self.log({"clip": clip_id, "step": "downloaded", "path": str(out)})
        return out

    def usage(self, start, end, timeframe="hour"):
        """Higgsfield's own record of completed generations and their cost (free GET; may lag; account-wide).
        start/end: ISO 8601 UTC, at most 31 days apart. Returns [{bucket, model, quantity, unit_price, cost}]."""
        rows, page = [], 1
        while True:
            r = self._req("GET", f"{API}/models/usage", tries=2, params={
                "start": start, "end": end, "timeframe": timeframe, "page": page, "limit": 100})
            for b in r.get("time_series", []):
                rows += [{"bucket": b["bucket"], **{k: x.get(k) for k in ("model", "quantity", "unit_price", "cost")}}
                         for x in b.get("results", [])]
            if page * 100 >= int(r.get("total") or 0):
                return rows
            page += 1
