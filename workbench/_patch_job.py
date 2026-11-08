from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "job.py"
t = p.read_text(encoding="utf-8")
old = (
    '    ("rough", "Make rough clips"),\n'
    '    ("courier", "Send to Higgsfield"),\n'
)
new = (
    '    ("rough", "Make rough clips"),\n'
    '    ("texture", "Marker texture (Higgs)"),\n'
    '    ("courier", "Send to Higgsfield"),\n'
)
if old not in t:
    raise SystemExit("stages anchor missing")
t = t.replace(old, new, 1)
old2 = (
    "    def set(self, stage, status=None, msg=None, progress=None):\n"
    '        s = self.state["stages"][stage]\n'
)
new2 = (
    "    def set(self, stage, status=None, msg=None, progress=None):\n"
    '        if stage not in self.state["stages"]:\n'
    "            label = dict(STAGES).get(stage, stage)\n"
    '            self.state["stages"][stage] = {"label": label, "status": "waiting", "msg": "", "progress": 0.0}\n'
    '        s = self.state["stages"][stage]\n'
)
if old2 not in t:
    raise SystemExit("set anchor missing")
t = t.replace(old2, new2, 1)
p.write_text(t, encoding="utf-8")
print("job.py ok")
