"""The editor: shots -> one 1080x1920 reel with jump cuts, a light grade, grain, handheld shake,
a slow push-in on the last shot, and each shot's own stroke sounds. No music.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

from .sound import SR
from .video import FPS, read_frames

W, H = 1080, 1920
SHAKE_PX = 2.5            # handheld drift (output px)
SHAKE_DEG = 0.12
GRAIN = 3.0               # grain strength in 8-bit levels


def _lut():
    x = np.arange(256, dtype=np.float32) / 255
    s = x - 0.15 * np.sin(2 * np.pi * x) / (2 * np.pi)              # soft S-curve, ends fixed
    r = np.clip(s * 1.015 + 0.004, 0, 1)
    b = np.clip(s * 0.975, 0, 1)
    return [(np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8) for c in (r, s, b)]


def _padded(frames, n):
    """First n frames, holding the last one if the clip is short; never holds the clip in memory."""
    fr = None
    i = 0
    for fr in frames:
        if i == n:
            return
        yield fr
        i += 1
    while fr is not None and i < n:
        yield fr
        i += 1


def _smooth_noise(n, rng, period_frames=45):
    k = max(2, n // period_frames + 3)
    pts = rng.normal(0, 1, k)
    return np.interp(np.arange(n), np.linspace(0, n, k), pts)


def cut(clips, wavs, out_path, push_in_last=True, seed=0, progress=None):
    """clips: list of mp4 paths (any size, will be fit to 9:16); wavs: matching mono wav paths."""
    rng = np.random.default_rng(seed)
    lut = _lut()
    grains = [np.clip(rng.normal(0, GRAIN, (H // 2, W // 2)), -10, 10).astype(np.float32) for _ in range(6)]
    # audio: concatenate with tiny crossfades at the cuts
    audio = []
    for w in wavs:
        sr, a = wavfile.read(str(w))
        assert sr == SR
        audio.append(a.astype(np.float32))
    tmp_wav = Path(out_path).with_suffix(".tmp.wav")
    wavfile.write(str(tmp_wav), SR, _join(audio))
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-i", str(tmp_wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest",
         "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    total = sum(int(round(len(a) / SR * FPS)) for a in audio)
    done = 0
    for ci, (clip, a) in enumerate(zip(clips, audio)):
        n_want = int(round(len(a) / SR * FPS))
        n = n_want
        dx, dy, dr = (_smooth_noise(n, rng) for _ in range(3))
        last = push_in_last and ci == len(clips) - 1
        for i, fr in enumerate(_padded(read_frames(clip, fps=FPS), n)):   # streamed: exact length = its audio
            zoom = 1.03 + (0.05 * i / max(1, n - 1) if last else 0)
            M = cv2.getRotationMatrix2D((fr.shape[1] / 2, fr.shape[0] / 2), dr[i] * SHAKE_DEG, 1.0)
            s = W / fr.shape[1] * zoom
            M = np.vstack([M, [0, 0, 1]])
            S = np.array([[s, 0, W / 2 - s * fr.shape[1] / 2 + dx[i] * SHAKE_PX],
                          [0, s, H / 2 - s * fr.shape[0] / 2 + dy[i] * SHAKE_PX], [0, 0, 1]])
            A = (S @ M)[:2]
            out = cv2.warpAffine(fr, A, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            out = np.dstack([cv2.LUT(out[..., c], lut[c]) for c in range(3)])
            g = cv2.resize(grains[(i + ci) % len(grains)], (W, H), interpolation=cv2.INTER_LINEAR)
            out = np.clip(out.astype(np.float32) + g[..., None], 0, 255).astype(np.uint8)
            p.stdin.write(out.tobytes())
            done += 1
            if progress and done % 30 == 0:
                progress(done / total, f"cutting: frame {done}/{total}")
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the reel")
    tmp_wav.unlink(missing_ok=True)
    return out_path


def _join(parts, fade_ms=4):
    """Butt-join the shots' audio with a few-ms fade at each cut: no clicks, no A/V drift."""
    k = int(SR * fade_ms / 1000)
    r = np.linspace(0, 1, k, dtype=np.float32)
    out = []
    for i, a in enumerate(parts):
        a = a.copy()
        if len(a) > 2 * k:
            if i > 0:
                a[:k] *= r
            if i < len(parts) - 1:
                a[-k:] *= r[::-1]
        out.append(a)
    return np.concatenate(out)
