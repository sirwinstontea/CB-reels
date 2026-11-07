"""Small ffmpeg helpers: write frames to mp4, read frames back."""
import subprocess

import numpy as np

FPS = 30


class Writer:
    def __init__(self, path, w, h, fps=FPS, crf=16, preset="medium"):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, rgb):
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    import json
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"w": s["width"], "h": s["height"], "fps": float(num) / float(den),
            "duration": float(j["format"]["duration"])}


def read_frames(path, w=None, h=None, fps=None):
    """Yield RGB uint8 frames, optionally resized / resampled to fps."""
    info = probe(path)
    w, h = w or info["w"], h or info["h"]
    vf = [f"scale={w}:{h}:flags=area"]
    if fps:
        vf.insert(0, f"fps={fps}")
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", ",".join(vf),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()
