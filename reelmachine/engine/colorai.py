"""Colors from Claude (when a page has no colored target): Claude sees the page twice, clean and
with every fillable shape numbered, and picks a marker (or "none" = leave the paper white) for
each number, like a colorist planning a page. Costs a few cents; the answer is cached per job.
"""
import base64
import json

import cv2
import numpy as np
from scipy import ndimage as ndi

from .judge import MODEL, _client

PROMPT = """You are planning how to color this printed coloring page with alcohol markers, for a calm,
satisfying coloring video. Image 1 is the page. Image 2 is the same page with every fillable shape
numbered in red (the number sits inside its shape).

Pick one marker for each numbered shape from this set (id: ink colour):
{markers}

Color it the way a skilled colorist would: a cozy, harmonious palette that suits the scene, every
object in a believable colour, the same kind of object in the same colour (or a deliberate
alternating pattern, e.g. tiles), enough contrast between neighbouring shapes that the drawing
stays readable, and light colours for things that are white in real life (use cream-01, fog-31 or
sky-15 rather than leaving them blank, so they still get colored on camera). Use "none" only for
tiny details and for the letters of the title. Return every number listed: {ids}.

Then plan the depth, like the best alcohol-marker colorists on Instagram (cozy, rich, never muddy):
- light: where the light comes from for the whole page.
- Every shape automatically gets a darker marker of its own family along its edges on the shadow
  side. In "accents", override "shade" only where a different darker marker looks better (e.g. a
  warm brown under a yellow duck, violet in a blue shadow), or "none" for things that must stay flat.
- Big shapes (sky, wall, water, floor, a bath, a big garment) get a soft gradient: 1-2 extra
  markers ("gradient") blended toward a side ("gradient_dir": down/up/left/right, "in" = toward the
  middle, "out" = toward the edges), e.g. a sky deepening upward, water darker at the bottom, warm
  glow fading out around a lamp. Use harmonious neighbouring hues, light to mid values.
- Patterns where the object would have them: "stripes" (wallpaper, a towel, a sweater, a scarf,
  pattern_dir vertical/horizontal/diagonal), "dots" (fabric, a mushroom cap), "sparkle" (white
  gel-pen dots on a night sky, snow, bubbles, water). pattern_marker = which marker draws it
  ("auto" = the shade of the base).
- highlight: true adds a white gel-pen glint on the lit side of round shiny things (eyes, noses,
  bubbles, baubles, balls, a glossy tub, fruit). Use it on maybe 5-15 shapes, not everything.
List only the shapes that get something beyond the automatic edge shade (usually 10-30)."""


def _labeled(page, size=1400):
    h, w = page.shape
    s = size / max(h, w)
    base = (np.clip(page.lines, 0, 1) * 255).astype(np.uint8)
    base = cv2.resize(base, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    clean = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    lab = clean.copy()
    for rid, r in page.regions.items():
        y0, y1, x0, x1 = r.bbox
        m = page.labels[y0:y1, x0:x1] == rid
        d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)
        p = (int((x0 + cx) * s), int((y0 + cy) * s))
        fs = 0.32 if d.max() * s < 9 else 0.45
        txt = str(rid)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        org = (p[0] - tw // 2, p[1] + th // 2)
        cv2.putText(lab, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(lab, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 230), 1, cv2.LINE_AA)
    return clean, lab


def _b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.standard_b64encode(buf.tobytes()).decode()


def _style(accents, idx, ids):
    """Claude's accents -> {rid: style} with marker indices (unknown names dropped)."""
    style = {}
    for a in accents:
        rid = int(a.get("region", -1))
        if rid not in ids:
            continue
        s = {}
        if a.get("shade") not in (None, "auto"):
            s["shade"] = "none" if a["shade"] == "none" else idx.get(a["shade"], "auto")
        g = [idx[x] for x in a.get("gradient") or [] if x in idx][:2]
        if g:
            s["gradient"], s["gradient_dir"] = g, a.get("gradient_dir", "down")
        if a.get("pattern", "none") != "none":
            s["pattern"], s["pattern_dir"] = a["pattern"], a.get("pattern_dir", "vertical")
            if a.get("pattern_marker") in idx:
                s["pattern_marker"] = idx[a["pattern_marker"]]
        if a.get("highlight"):
            s["highlight"] = True
        if s:
            style[rid] = s
    return style


def plan(page, markers, cache_path=None):
    """shading.Plan: {region id: base marker index} (+ .style per-region depth, .light).
    Old cached plans (no style) load as a plain base plan. Raises RuntimeError on failure."""
    import anthropic
    from .shading import Plan
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text())
        return (Plan({int(k): v for k, v in data["plan"].items()},
                     {int(k): v for k, v in data.get("style", {}).items()}, data.get("light")),
                data.get("usd", 0.0))
    ids = sorted(page.regions)
    mids = [m["id"] for m in markers if not m.get("opaque")]
    allm = mids + ["none", "auto"]
    accent = {
        "type": "object",
        "properties": {
            "region": {"type": "integer"},
            "shade": {"type": "string", "enum": allm},
            "gradient": {"type": "array", "items": {"type": "string", "enum": mids}},
            "gradient_dir": {"type": "string", "enum": ["down", "up", "left", "right", "in", "out"]},
            "pattern": {"type": "string", "enum": ["none", "stripes", "dots", "sparkle"]},
            "pattern_dir": {"type": "string", "enum": ["vertical", "horizontal", "diagonal"]},
            "pattern_marker": {"type": "string", "enum": allm},
            "highlight": {"type": "boolean"}},
        "required": ["region", "shade", "gradient", "gradient_dir", "pattern", "pattern_dir", "pattern_marker",
                     "highlight"],
        "additionalProperties": False}
    schema = {
        "type": "object",
        "properties": {
            "assignments": {"type": "array", "items": {
                "type": "object",
                "properties": {"region": {"type": "integer"}, "marker": {"type": "string", "enum": mids + ["none"]}},
                "required": ["region", "marker"], "additionalProperties": False}},
            "light": {"type": "string", "enum": ["upper-left", "upper-right", "top", "left", "right"]},
            "accents": {"type": "array", "items": accent}},
        "required": ["assignments", "light", "accents"], "additionalProperties": False,
    }
    clean, lab = _labeled(page)
    mk = "\n".join(f"{m['id']}: #{''.join('%02x' % v for v in m['ink'])}" for m in markers if not m.get("opaque"))
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(clean)}},
               {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(lab)}},
               {"type": "text", "text": PROMPT.format(markers=mk, ids=", ".join(map(str, ids)))}]
    try:
        r = _client().beta.messages.create(
            model=MODEL, max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}])
    except anthropic.APIError as e:
        raise RuntimeError(f"color planner: {e}")
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"color planner stopped: {r.stop_reason}")
    out = json.loads(next(b.text for b in r.content if b.type == "text"))
    idx = {m["id"]: i for i, m in enumerate(markers) if not m.get("opaque")}
    result = {int(a["region"]): idx[a["marker"]] for a in out["assignments"]
              if a["marker"] in idx and int(a["region"]) in page.regions}
    if len(result) < 0.6 * len(ids):
        raise RuntimeError(f"color planner covered only {len(result)} of {len(ids)} shapes")
    style = _style(out.get("accents", []), idx, set(result))
    light = out.get("light", "upper-left")
    usd = round((r.usage.input_tokens * 4 + r.usage.output_tokens * 20) / 1e6, 4)
    if cache_path:
        cache_path.write_text(json.dumps({"plan": result, "usd": usd, "light": light, "style": style,
                                          "none": [a["region"] for a in out["assignments"] if a["marker"] == "none"]}))
    return Plan(result, style, light), usd
