"""Low-effort on-screen-text explainer: how the coloring simulator works.
Renders PNG slides then ffmpeg → sim_how_it_works.mp4
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_MP4 = ROOT / "sim_how_it_works.mp4"
FRAMES = ROOT / "_frames"
W, H = 1920, 1080
BG = (18, 18, 22)
FG = (236, 236, 240)
MUTED = (160, 165, 175)
ACCENT = (120, 190, 255)
BOX = (32, 34, 42)

# (seconds, title, body_lines, optional_image_name under ROOT)
SLIDES: list[tuple[float, str, list[str], str | None]] = [
    (
        8,
        "How the coloring simulator works",
        [
            "This pipeline fakes a real marker coloring session.",
            "Our code owns the ink on the page (so it stays consistent).",
            "AI is used later only to put a real-looking hand on top.",
            "Goal of this video: what happens, and the main technical pieces.",
        ],
        None,
    ),
    (
        10,
        "1. Inputs",
        [
            "Line-art page → segmented into regions (closed shapes to fill).",
            "Color plan: which marker for each region (and richer plans for layers).",
            "Marker set: ink RGB, tip size, optional white gel pen.",
            "Shading layers (shading.py), in order per region:",
            "  base · shade · gradient · pattern · highlight",
            "Each layer can carry a mask, ink weight map, stroke direction, tip type.",
        ],
        None,
    ),
    (
        14,
        "2. Stroke planner — strokes.py + gestlib.py",
        [
            "Builds a Timeline sampled at 120 Hz (120 tip samples per second).",
            "Each sample stores:",
            "  time · tip x/y (page pixels) · pen down · marker id · tip radius",
            "  stroke id · region id · pressure 0..1 (ink amount; lifts taper to 0)",
            "Paths are remixed from real coloring-video gestures (gestlib),",
            "not random scribbles — along/across/pressure curves fitted to length.",
            "Layer kind picks the technique: full fill, edge flicks, stripes, taps…",
        ],
        None,
    ),
    (
        8,
        "Stroke planner — mental model",
        [
            "Think: a script the fake marker must follow.",
            "Pen down → glide → lift → move to the next uncovered spot.",
            "Pacing is tuned to refs (~few strokes per second, short lifts).",
            "Same seed → same timeline (deterministic).",
        ],
        None,
    ),
    (
        16,
        "3. Ink canvas — ink.py / InkCanvas",
        [
            "Digital paper that takes alcohol-marker ink from the timeline.",
            "Per pixel it tracks: how full it is, which marker, wet age, capacity.",
            "Each dab() at the tip deposits ink scaled by pressure.",
            "Same-marker wet ink levels (~0.25 mm) so strokes melt together.",
            "A new wet color can lift a bit of still-wet different ink (soft blends).",
            "Dried ink can re-wet and darken slightly when hit again.",
            "Paper tooth: ±~6% density (darker in grain valleys).",
            "White gel is opaque (covers, does not mix). Ink stays inside region masks.",
        ],
        None,
    ),
    (
        9,
        "Ink — why this matters",
        [
            "Alcohol markers should look smooth (butter), not scratchy hatch lines.",
            "Because ink is simulated, every shot sees the same page state.",
            "Nothing painted can vanish between jump cuts.",
            "This plate is the ground truth for the rest of the pipeline.",
        ],
        None,
    ),
    (
        10,
        "4. Desk composite — scene.py",
        [
            "The colored page is warped onto a real desk photo (homography).",
            "Transmission multiply lets desk light/paper texture show through.",
            "Clip resolution for this stage is about 720×1280 (phone-ish).",
            "You get a still-looking coloring page sitting on a desk — no hand yet.",
        ],
        None,
    ),
    (
        9,
        "5. Shots — director.py",
        [
            "The full session can be many minutes of coloring.",
            "The director cuts it into short timed windows (close / medium / wide).",
            "Each shot has a crop and duration for the final vertical reel.",
            "Later cost policy: often only close/medium go to paid AI.",
        ],
        None,
    ),
    (
        8,
        "6. Plate vs rough — rough.py + standin.py",
        [
            "For each shot we write two videos:",
            "PLATE = ink on desk only (official coloring — ground truth).",
            "ROUGH = same frames + a cheap stand-in hand glued to the nib track.",
            "Also: hand masks, end frame, JSON nib track for QA later.",
        ],
        None,
    ),
    (
        7,
        "Plate (ground truth — no hand)",
        [
            "Example still from an existing job (s01.plate).",
            "This is what must never change when AI runs.",
        ],
        "_still_plate.jpg",
    ),
    (
        7,
        "Rough (plate + stand-in hand)",
        [
            "Same moment with the placeholder hand on the tip path.",
            "This clip is what we send to the video model.",
        ],
        "_still_rough.jpg",
    ),
    (
        12,
        "7. AI hand — courier.py → Kling Omni",
        [
            "Upload the rough MP4 (and reference images) to Higgsfield.",
            "Model: Kling Omni video-edit (keeps the page better than older edits).",
            "Ask: replace the stand-in with a photoreal hand holding the marker;",
            "      do not redraw / recolor the page.",
            "Poll → download the edited clip.",
            "This step costs money; draft reels skip it and keep the stand-in.",
        ],
        None,
    ),
    (
        11,
        "8. Referee — spot.py",
        [
            "Compare AI frames to the plate outside the hand area.",
            "Hand matte = where the stand-in (or detected hand) was.",
            "If the page drifted, paste plate pixels back (or fail the shot).",
            "Invariant: every colored pixel on the page is owned by the sim.",
            "AI may only change the hand (and maybe tiny edge under the matte).",
        ],
        None,
    ),
    (
        9,
        "9. Finish — sound.py + editor.py",
        [
            "Per-stroke soft marker sounds, timed to the timeline (no music).",
            "Jump-cut the accepted shots, grade, light grain/shake.",
            "Encode the final vertical reel (e.g. 1080×1920).",
            "Optional Claude judge scores hand/look; cost.json logs spend.",
        ],
        None,
    ),
    (
        10,
        "End-to-end flow",
        [
            "page + colors + layers",
            "  → stroke timeline (120 Hz)",
            "  → ink sim (InkCanvas dabs)",
            "  → desk scene",
            "  → plate + rough per shot",
            "  → (optional) Kling Omni hand",
            "  → spot lock to plate",
            "  → sound + cut → reel",
        ],
        None,
    ),
    (
        8,
        "Takeaway",
        [
            "Simulator = scripted tip path + physics-ish alcohol ink on paper.",
            "Plate = locked coloring. Rough = plate + fake hand.",
            "AI = real hand only. Spot = referee that protects the page.",
            "That split is why the video can stay visually consistent.",
        ],
        None,
    ),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibril.ttf",
    ]
    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ] + candidates
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def render_slide(title: str, body: list[str], image_name: str | None) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    title_f = _font(48, bold=True)
    body_f = _font(32)
    small_f = _font(22)

    dr.rectangle((0, 0, W, 8), fill=ACCENT)
    dr.text((80, 48), title, font=title_f, fill=FG)

    y = 130
    text_right = W - 80
    img_path = ROOT / image_name if image_name else None
    if img_path and img_path.exists():
        still = Image.open(img_path).convert("RGB")
        # fit into right third
        box_w, box_h = 520, 900
        still.thumbnail((box_w, box_h))
        x0 = W - 80 - still.width
        y0 = (H - still.height) // 2
        dr.rectangle((x0 - 8, y0 - 8, x0 + still.width + 8, y0 + still.height + 8), fill=BOX)
        im.paste(still, (x0, y0))
        text_right = x0 - 40

    max_w = text_right - 80
    for line in body:
        if not line.strip():
            y += 18
            continue
        wrapped = _wrap(dr, line, body_f, max_w)
        for wl in wrapped:
            if y > H - 80:
                break
            color = ACCENT if wl.strip().startswith("→") or "PLATE" in wl or "ROUGH" in wl else FG
            if line.strip().startswith("  "):
                dr.text((110, y), wl, font=body_f, fill=MUTED)
            else:
                dr.text((80, y), wl, font=body_f, fill=color)
            y += 44
        y += 8

    dr.text((80, H - 48), "Coloring Reel Machine · simulator explainer", font=small_f, fill=MUTED)
    return im


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()

    concat_lines: list[str] = []
    for i, (sec, title, body, img) in enumerate(SLIDES):
        path = FRAMES / f"slide_{i:02d}.png"
        render_slide(title, body, img).save(path)
        # ffmpeg concat demuxer needs forward slashes
        concat_lines.append(f"file '{path.as_posix()}'")
        concat_lines.append(f"duration {sec}")
    # last file repeated for concat demuxer
    last = FRAMES / f"slide_{len(SLIDES) - 1:02d}.png"
    concat_lines.append(f"file '{last.as_posix()}'")

    list_path = FRAMES / "concat.txt"
    list_path.write_text("\n".join(concat_lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vf", "fps=30,format=yuv420p",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(OUT_MP4),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print("wrote", OUT_MP4, "size_mb", round(OUT_MP4.stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
