"""Sound: marker-on-paper ASMR placed on every stroke, no music.

Alcohol markers glide: a soft, smooth "shhh" with no scratch, grit or crackle. The tone curve
below was measured on the marker-only stretches of the reference reels: flat from ~0.6 to 3 kHz,
rolling off gently above 4 kHz (-9 dB at 10 kHz, -20 dB at 16 kHz).

Source, in order of preference:
1. a sound bank: any marker ASMR recordings dropped in assets/sounds/ (Pixabay, Artlist, your own),
   sliced automatically into single strokes and taps, then softened the same way;
2. a procedural felt-tip glide, shaped by the pen's real speed (no files needed).
Each shot gets <id>.wav, sample-aligned with its frames.
"""
import subprocess
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

SR = 48000
EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}
ROOM_DB = -52.0           # constant room tone so cuts never drop to digital silence
STROKE_DB = -21.0         # loudness of the strokes in a shot (RMS while the nib is down)
ATTACK_S, RELEASE_S = 0.022, 0.050
# (Hz, dB) target tone of a gliding nib, rel. 1 kHz (measured on the references, rumble left out)
TONE = [(0, -40), (120, -14), (250, -2), (400, 3), (600, 2), (1000, 0), (3000, 0), (4000, -2),
        (5500, -4), (8000, -6), (10000, -9), (12500, -14), (16000, -22), (20000, -40), (SR / 2, -60)]


@lru_cache(maxsize=4)
def _tone_fir(dark=0.0):
    """Linear-phase FIR with the TONE curve; `dark` (dB/octave above 3 kHz) for slow, light strokes."""
    f = np.array([p[0] for p in TONE], float)
    g = np.array([p[1] for p in TONE], float)
    g = g - dark * np.log2(np.maximum(f, 3000) / 3000)
    return signal.firwin2(1023, f / (SR / 2), 10 ** (g / 20))


def _shaped_noise(n, rng, dark=0.0):
    x = signal.fftconvolve(rng.normal(0, 1, n + 1022), _tone_fir(dark), mode="valid")[:n]
    return (x / (x.std() + 1e-9)).astype(np.float32)


def _mean(x, n):
    """Centered moving mean over n samples, same length as x (also when x is shorter than n)."""
    return ndimage.uniform_filter1d(np.asarray(x, np.float64), size=max(1, n), mode="nearest")


def _smooth(x, tau_s):
    """One-pole low-pass both ways (no lag): turns a jumpy curve into a gentle one."""
    a = np.exp(-1.0 / max(1.0, tau_s * SR))
    sos = np.array([[1 - a, 0, 0, 1, -a, 0]])
    return signal.sosfiltfilt(sos, x)


def _soften(seg):
    """Bank slice -> soft glide: gentle top roll-off, de-essed, clicks and grain smoothed away."""
    seg = signal.sosfiltfilt(signal.butter(2, [150, 7000], "bp", fs=SR, output="sos"), seg)
    hi = signal.sosfiltfilt(signal.butter(2, 2500, "hp", fs=SR, output="sos"), seg)
    fast = np.sqrt(_mean(hi * hi, 96)) + 1e-9     # 2 ms
    slow = np.sqrt(_mean(hi * hi, 2400)) + 1e-9  # 50 ms
    over = np.maximum(fast / (1.6 * slow), 1.0)                                 # spikes above the glide
    seg = seg - hi + hi / _smooth(over, 0.002)
    return (seg / (np.abs(seg).max() + 1e-9)).astype(np.float32)


def _silk(x, amount=0.35):
    """Iron out the millisecond flutter of raw noise above 1.5 kHz (that flutter reads as grit/scratch).
    amount 0 = untouched, 1 = perfectly even hiss."""
    hi = signal.sosfiltfilt(signal.butter(2, 1500, "hp", fs=SR, output="sos"), x)
    fast = np.sqrt(_mean(hi * hi, 96)) + 1e-9     # 2 ms
    slow = np.sqrt(_mean(hi * hi, 1920)) + 1e-9  # 40 ms
    return x - hi + hi / (_smooth(fast, 0.0007) / slow) ** amount


def _decode(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _envelope(x, win_s=0.01):
    n = max(1, int(win_s * SR))
    e = np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same"))
    return e


class Bank:
    """Strokes and taps cut from recordings. Cached in assets/sounds/_bank.npz."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.strokes, self.taps, self.room = [], [], None
        files = sorted(p for p in self.folder.glob("*") if p.suffix.lower() in EXTS) if self.folder.exists() else []
        if not files:
            return
        cache = self.folder / "_bank.npz"
        stamp = np.array(["soft-v2"] + [f"{p.name}:{p.stat().st_size}" for p in files])   # bump when slicing changes
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if list(z["stamp"]) == list(stamp):
                self.strokes, self.taps, self.room = list(z["strokes"]), list(z["taps"]), z["room"]
                return
        rooms = []
        for p in files:
            self._slice(_decode(p), rooms)
        self.room = np.concatenate(rooms)[: SR * 20] if rooms else None
        np.savez_compressed(cache, stamp=stamp, strokes=np.array(self.strokes, dtype=object),
                            taps=np.array(self.taps, dtype=object), room=self.room if self.room is not None else np.zeros(0))

    def _slice(self, x, rooms):
        x = signal.sosfilt(signal.butter(2, 120, "hp", fs=SR, output="sos"), x)   # rumble off
        e = _envelope(x)
        floor = np.percentile(e, 10)
        on = e > max(floor * 4, np.percentile(e, 60) * 0.5)
        # close tiny gaps, then cut runs
        on = np.convolve(on, np.ones(int(0.03 * SR)), mode="same") > 0
        d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        quiet_from = 0
        for a, b in zip(starts, ends):
            if a - quiet_from > SR // 2:
                rooms.append(x[quiet_from + SR // 10: a - SR // 10])
            quiet_from = b
            dur = (b - a) / SR
            seg = _soften(x[max(0, a - int(0.01 * SR)): b + int(0.02 * SR)])
            if dur < 0.09:
                self.taps.append(seg.astype(np.float32))
            elif dur < 3.0:
                self.strokes.append(seg.astype(np.float32))

    @property
    def ok(self):
        return len(self.strokes) >= 8


def _stroke_events(tl, t0, t1, ppm):
    """(start, end, speed curve times, speeds mm/s, landed) for each glide overlapping [t0, t1).

    A glide is one unbroken stretch of nib-on-paper: consecutive sub-strokes the pen never lifts
    between (scribbles, hatching, edge passes) are one continuous sound, not a train of tiny
    separate fades, which would read as scratching."""
    i0, i1 = np.searchsorted(tl.t, [t0 - 3, t1 + 0.5])
    down = np.asarray(tl.down[i0:i1], bool)
    t, x, y = tl.t[i0:i1], tl.x[i0:i1], tl.y[i0:i1]
    ev = []
    d = np.diff(np.concatenate([[0], down.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        idx = np.arange(a, b)
        if len(idx) < 2:
            continue
        ta, tb = float(t[idx[0]]), float(t[idx[-1]])
        if tb < t0 or ta > t1:
            continue
        sp = np.hypot(np.diff(x[idx]), np.diff(y[idx])) / np.maximum(np.diff(t[idx]), 1e-4) / ppm   # mm/s
        ev.append((ta, tb, t[idx[1:]], sp, True))    # every glide starts with the nib landing
    return ev


def _synth_stroke(n, speed, rng, tilt, pressure):
    """Alcohol-marker nib gliding on paper: smooth shaped noise whose loudness and brightness follow
    the pen's speed. No grain or flutter: only a slow, gentle swell from hand pressure."""
    sp = np.clip(_smooth(speed, 0.03) / 70.0, 0.08, 1.2)
    bright = _shaped_noise(n, rng, dark=0.0)
    dark = _shaped_noise(n, rng, dark=2.0 + 2.0 * tilt)          # slow strokes sound softer, darker
    mix = np.clip((sp - 0.2) / 0.7, 0, 1)
    x = _silk(dark * (1 - mix) + bright * mix)
    x /= np.sqrt(np.mean(x * x)) + 1e-9
    return x * sp ** 0.5 * pressure


def _fit(seg, n, rng):
    """Bank stroke resized to n samples (resample up to +-25%, else loop/trim), with fades."""
    ratio = n / len(seg)
    if 0.75 < ratio < 1.25:
        out = signal.resample(seg, n)
    elif len(seg) >= n:
        a = rng.integers(0, len(seg) - n + 1)
        out = seg[a:a + n]
    else:
        reps = int(np.ceil(n / len(seg)))
        out = np.tile(seg, reps)[:n]
    return out


def _fade(x, attack_s=ATTACK_S, release_s=RELEASE_S):
    """Raised-cosine touch-down and lift-off: the nib eases on and off the paper."""
    for k, sl in ((min(len(x) // 3, int(attack_s * SR)), slice(None)), (min(len(x) // 2, int(release_s * SR)), slice(None, None, -1))):
        if k > 1:
            r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
            x[sl][:k] *= r
    return x


def _soft_tap(rng):
    """Felt nib touching down: a muted, low 'tup', never a click."""
    k = int(0.03 * SR)
    burst = signal.sosfiltfilt(signal.butter(2, [120, 1200], "bp", fs=SR, output="sos"), rng.normal(0, 1, k))
    env = (1 - np.exp(-np.arange(k) / (0.002 * SR))) * np.exp(-np.arange(k) / (0.007 * SR))
    tap = burst * env
    return (tap / (np.abs(tap).max() + 1e-9)).astype(np.float32)


def render(tl, shot, bank, out_path, ppm, seed=0):
    """Write the shot's audio (mono float32 wav, SR) covering [shot.t0, shot.t1)."""
    rng = np.random.default_rng(seed + zlib.crc32(shot.id.encode()))
    t0, t1 = shot.t0, shot.t1
    N = int(round((t1 - t0) * SR))
    out = np.zeros(N, np.float32)
    # room tone
    if bank and bank.room is not None and len(bank.room) > SR:
        a = rng.integers(0, max(1, len(bank.room) - N))
        room = np.resize(bank.room[a:a + N], N)
        room = room / (np.sqrt(np.mean(room ** 2)) + 1e-9)
    else:   # soft, dark room hush
        room = signal.sosfiltfilt(signal.butter(2, 1500, "lp", fs=SR, output="sos"), rng.normal(0, 1, N))
        room /= np.sqrt(np.mean(room ** 2)) + 1e-9
    room = room.astype(np.float32) * 10 ** (ROOM_DB / 20)
    strokes = np.zeros(N, np.float32)
    down = np.zeros(N, bool)
    if shot.hand:
        tilt = rng.uniform(0, 1)
        pre = 4 * SR                      # glides may start up to 3 s before the shot
        p = _smooth(rng.normal(0, 1, N + pre + SR), 0.08)
        pressure = (1 + 0.10 * p / (p.std() + 1e-9)).astype(np.float32)   # slow hand-pressure swell, ~+-10%
        for ta, tb, ts, sp, landed in _stroke_events(tl, t0, t1, ppm):
            a, b = int((ta - t0) * SR), int((tb - t0) * SR)
            n = b - a
            if n < SR // 50 or a + pre < 0:
                continue
            curve = np.interp(np.arange(n) / SR + ta, ts, sp) if len(ts) > 1 else np.full(n, 60.0)
            if bank and bank.ok:
                seg = bank.strokes[rng.integers(len(bank.strokes))]
                s = _fit(seg, n, rng) * np.clip(_smooth(curve, 0.03) / 70.0, 0.15, 1.4) ** 0.6
            else:
                s = _synth_stroke(n, curve, rng, tilt, pressure[a + pre: a + pre + n])
            s = _fade(s.astype(np.float32)) * 10 ** (rng.uniform(-1.5, 1.5) / 20)
            if landed:                                      # the nib touching down, softly
                if bank and bank.taps:
                    tap = _fade(bank.taps[rng.integers(len(bank.taps))][: SR // 20].copy(), 0.002, 0.02) * 0.2
                else:
                    tap = _soft_tap(rng) * 0.25
                s[: len(tap)] += tap[: len(s)]
            lo, hi = max(0, a), min(N, b)
            if hi > lo:
                strokes[lo:hi] += s[lo - a: hi - a]
                down[lo:hi] = True
    if down.sum() > SR // 10:      # same loudness in every shot, whatever the pen speed
        strokes *= 10 ** (STROKE_DB / 20) / (np.sqrt(np.mean(strokes[down] ** 2)) + 1e-9)
    out = room + strokes
    peak = np.abs(out).max()       # soft-knee limiter: the loudest strokes never clip
    if peak > 0.7:
        out = np.where(np.abs(out) > 0.6, np.sign(out) * (0.6 + 0.3 * np.tanh((np.abs(out) - 0.6) / 0.3)), out)
    _write_wav(out_path, out.astype(np.float32))
    return out_path


def _write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(str(path), SR, x.astype(np.float32))
