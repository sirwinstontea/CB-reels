"""A job = one page on its way to a reel. Holds the stage diagram's state and the log.

state.json  - what the window shows: each stage's status, message, progress, cost so far
log.jsonl   - one line per event, append-only; used to resume without paying twice
"""
import json
import time
from pathlib import Path

STAGES = [
    ("page", "Read the page"),
    ("colors", "Choose colors"),
    ("strokes", "Plan strokes"),
    ("ink", "Color the digital paper"),
    ("rough", "Make rough clips"),
    ("texture", "Marker texture (Higgs)"),
    ("courier", "Send to Higgsfield"),
    ("spot", "Spot the difference"),
    ("judge", "The judge"),
    ("sound", "Add the sound"),
    ("cut", "Cut the reel"),
    ("final", "Final check"),
]
STATUSES = ("waiting", "running", "passed", "retrying", "failed", "skipped")


class Job:
    def __init__(self, root, name):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.state_path = self.dir / "state.json"
        self.log_path = self.dir / "log.jsonl"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"name": name, "created": time.time(), "usd": 0.0, "done": False, "error": "",
                          "stages": {k: {"label": label, "status": "waiting", "msg": "", "progress": 0.0}
                                     for k, label in STAGES}}
            self._save()

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)          # atomic, so the window never reads half a file

    def log(self, event):
        event = {"t": round(time.time(), 2), **event}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def events(self):
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def set(self, stage, status=None, msg=None, progress=None):
        if stage not in self.state["stages"]:
            label = dict(STAGES).get(stage, stage)
            self.state["stages"][stage] = {"label": label, "status": "waiting", "msg": "", "progress": 0.0}
        s = self.state["stages"][stage]
        if status:
            assert status in STATUSES
            s["status"] = status
        if msg is not None:
            s["msg"] = msg
        if progress is not None:
            s["progress"] = round(float(progress), 3)
        self._save()
        self.log({"stage": stage, "status": s["status"], "msg": s["msg"]})

    def add_cost(self, usd):
        self.state["usd"] = round(self.state["usd"] + usd, 4)
        self._save()

    def finish(self, ok, error=""):
        self.state["done"] = True
        self.state["ok"] = ok
        self.state["error"] = error
        self._save()
