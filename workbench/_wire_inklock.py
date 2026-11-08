from pathlib import Path

# fix inklock imports
p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "inklock.py"
t = p.read_text(encoding="utf-8")
t = t.replace(
    """from .spot import (
    DE_CHANGED, FPS, LIMITS, Q, Writer, _align, _apply_color, _color_fit, _count, _lab,
    probe, read_frames,
)
""",
    """from .video import Writer, read_frames, probe, FPS
from .spot import DE_CHANGED, LIMITS, Q, _align, _apply_color, _color_fit, _count, _lab
""",
)
p.write_text(t, encoding="utf-8")

# make spot.lock_ink delegate
sp = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "spot.py"
st = sp.read_text(encoding="utf-8")
start = st.find("def lock_ink(")
end = st.find('\nif __name__ == "__main__":', start)
if start < 0 or end < 0:
    raise SystemExit("spot lock_ink missing")
delegate = '''def lock_ink(result_path, plate_path, inkmask_path, out_path=None):
    """Delegate to inklock (align + colour-fit + mask composite)."""
    from . import inklock
    return inklock.lock_ink(result_path, plate_path, inkmask_path, out_path=out_path)

'''
sp.write_text(st[:start] + delegate + st[end:], encoding="utf-8")
print("wired")
