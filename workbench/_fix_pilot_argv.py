from pathlib import Path

p = Path(__file__).resolve().parent / "pilot.py"
t = p.read_text(encoding="utf-8")
old = "mode, name = sys.argv[1], sys.argv[2]\ns = pipeline.load_settings()\n"
new = (
    'USAGE = "usage: python pilot.py draft|shot|full NAME [shot_id] [model]"\n'
    "if len(sys.argv) < 3:\n"
    "    raise SystemExit(USAGE)\n"
    "mode, name = sys.argv[1], sys.argv[2]\n"
    "s = pipeline.load_settings()\n"
)
if old not in t:
    raise SystemExit("argv header missing")
t = t.replace(old, new, 1)
old2 = 'elif mode == "shot":\n    sid = sys.argv[3]\n'
new2 = (
    'elif mode == "shot":\n'
    "    if len(sys.argv) < 4:\n"
    '        raise SystemExit(USAGE + "  (shot mode needs shot_id, e.g. s01)")\n'
    "    sid = sys.argv[3]\n"
)
if old2 not in t:
    raise SystemExit("shot block missing")
t = t.replace(old2, new2, 1)
p.write_text(t, encoding="utf-8")
print("pilot argv fixed")
