"""Final check: is the reel a valid, postable file? Format, length, sound present, nothing frozen
or black, and the last frame shows the finished page."""
import json
import subprocess

import cv2
import numpy as np

from .video import read_frames


def check(reel_path, finished_png=None, min_s=8, max_s=60):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(reel_path)],
                                  capture_output=True, text=True, check=True).stdout)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    dur = float(j["format"]["duration"])
    fails, rep = [], {"duration": round(dur, 2)}
    if not v or (v["width"], v["height"]) != (1080, 1920):
        fails.append("video is not 1080x1920")
    if not a:
        fails.append("no audio track")
    if not min_s <= dur <= max_s:
        fails.append(f"length {dur:.1f}s outside {min_s}-{max_s}s")
    # audio level
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(reel_path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    if len(x):
        rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        rep["audio_rms_db"] = round(float(rms), 1)
        if rms < -45:
            fails.append(f"audio almost silent ({rms:.0f} dB)")
        if np.abs(x).max() >= 0.999:
            fails.append("audio clips")
    # frames: black, frozen
    prev, frozen_run, worst, black, last = None, 0, 0, 0, None
    for f in read_frames(reel_path, 270, 480):
        g = f.mean()
        if g < 12:
            black += 1
        if prev is not None:
            d = np.abs(f.astype(np.int16) - prev).mean()
            frozen_run = frozen_run + 1 if d < 0.15 else 0
            worst = max(worst, frozen_run)
        prev = f.astype(np.int16)
        last = f
    rep.update(black_frames=black, longest_frozen=worst)
    if black:
        fails.append(f"{black} black frames")
    if worst > 45:
        fails.append(f"picture frozen for {worst} frames")
    if finished_png is not None and last is not None:
        ref = cv2.cvtColor(cv2.imread(str(finished_png)), cv2.COLOR_BGR2RGB)
        ref = cv2.resize(ref, (270, 480), interpolation=cv2.INTER_AREA)
        # the editor zooms/shakes slightly, so compare blurred color layout, not pixels
        b = lambda im: cv2.GaussianBlur(im.astype(np.float32), (0, 0), 6)
        err = float(np.abs(b(last) - b(ref)).mean())
        rep["end_vs_finished"] = round(err, 2)
        if err > 18:
            fails.append("last frame doesn't look like the finished page")
    rep["fails"] = fails
    rep["ok"] = not fails
    return rep
