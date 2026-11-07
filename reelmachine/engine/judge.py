"""The judge (optional): Claude looks at a few frames of each repaired clip and says whether the
hand looks real. Catches what pixel checks can't: extra fingers, a melted marker, a nib floating
above the ink, a cap that doesn't match. Costs a fraction of a cent per clip on Sonnet.

Key: Windows Credential Manager "reelmachine-anthropic", else the usual ANTHROPIC_API_KEY / ant login.
No key -> the stage is skipped, the reel still gets made.
"""
import base64
import json

import cv2
import numpy as np

from .video import read_frames, probe

MODEL = "claude-opus-5-5"            # the colour planner (colorai) uses this one
# Sonnet 5.5 gave the same verdicts as Opus on the test reel's clips at ~half the price (cost/judge_compare.py);
# Haiku 5.5 was too lenient on the flat stand-in, so it isn't used.
JUDGE_MODEL = "claude-sonnet-5-5"
KEYRING_SERVICE = "reelmachine-anthropic"
PASS_SCORE = 6
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # $/MTok in, out

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1-10, how real the hand and marker look"},
        "severe": {"type": "boolean", "description": "true if any frame has a defect a viewer would notice at once"},
        "problems": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "severe", "problems"],
    "additionalProperties": False,
}

PROMPT = """These are frames, in order, from one shot of a short video: a person coloring a printed
coloring page with an alcohol marker, filmed from above. The page and the ink are correct by
construction; judge only the hand and the marker.

Check: the hand has a normal number of fingers and normal anatomy in every frame; the marker
is one rigid object held in a natural grip; the nib touches the paper where the newest ink is;
the marker's cap/body colour stays the same across frames ({ink_note}); nothing smears, melts,
flickers or duplicates; skin looks like real skin, not plastic or a flat cut-out.

Score 1-10 for how real it looks to a casual viewer scrolling Instagram. Set severe=true if any
single frame has a defect such a viewer would notice at once. List each problem briefly."""


def _client():
    import anthropic
    try:
        import keyring
        key = keyring.get_password(KEYRING_SERVICE, "api")
    except Exception:
        key = None
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def available():
    import os
    try:
        import keyring
        if keyring.get_password(KEYRING_SERVICE, "api"):
            return True
    except Exception:
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def usd(model, usage):
    pin, pout = PRICES.get(model, PRICES[MODEL])
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6


def _frames_b64(path, n=4, size=(360, 640), offset=0.0):
    """n frames spread over the clip (offset 0..1 shifts the sample by a fraction of the spacing), streamed."""
    info = probe(path)
    total = max(1, int(round(info["duration"] * info["fps"])))
    step = (total - 1) / max(1, n - 1 + (offset > 0))
    want = set(np.clip(np.round(np.arange(n) * step + offset * step), 0, total - 1).astype(int).tolist())
    out = []
    last = None
    for i, fr in enumerate(read_frames(path, *size)):
        last = fr
        if i in want:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            out.append(base64.standard_b64encode(buf.tobytes()).decode())
    while len(out) < min(n, len(want)) and last is not None:     # clip shorter than probed: repeat the last frame
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(last, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        out.append(base64.standard_b64encode(buf.tobytes()).decode())
    return out


def judge(clip_path, ink_name="", model=None, offset=0.0):
    """Returns {"ok", "score", "severe", "problems", "usd", "model"} for one repaired clip."""
    import anthropic
    model = model or JUDGE_MODEL
    client = _client()
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
               for b in _frames_b64(clip_path, offset=offset)]
    note = f"it should look like a {ink_name} marker" if ink_name else "one marker throughout"
    content.append({"type": "text", "text": PROMPT.format(ink_note=note)})
    kw = dict(model=model, max_tokens=2000, messages=[{"role": "user", "content": content}],
              output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    if "haiku" not in model:      # Haiku has no server-side refusal fallback
        kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    try:
        r = client.beta.messages.create(**kw)
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"judge: network error {e}")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"judge: API error {e.status_code}: {e.message}")
    if r.stop_reason == "refusal":
        return {"ok": True, "score": None, "severe": False, "problems": ["judge declined; not counted"], "usd": 0.0,
                "model": model}
    text = next((b.text for b in r.content if b.type == "text"), "{}")
    v = json.loads(text)
    v["ok"] = v["score"] >= PASS_SCORE and not v["severe"]
    v["usd"] = round(usd(model, r.usage), 5)
    v["model"] = model
    return v
