"""Coloring Reel Machine: the window.

Drop a page in, watch each stage light up, get a Windows notification when a reel is ready
or when something fails for good. Jobs run one at a time in the background; closing the
window stops after the current stage is saved (a restarted job resumes without paying twice).
"""
import json
import os
import queue
import threading
import time
from pathlib import Path

import webview
from webview.dom import DOMEventHandler

from engine import pipeline
from engine.courier import KEYRING_SERVICE as HF_SERVICE, MODELS as HF_MODELS
from engine.job import STAGES
from engine.judge import KEYRING_SERVICE as AN_SERVICE
from engine.paths import DATA, HOME as ROOT
WHERE = {"courier": "Higgsfield", "judge": "Claude"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


class Api:
    def __init__(self):
        self.q = queue.Queue()
        self.jobs = []          # [{"id", "page", "state"}]
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- jobs
    def _worker(self):
        while True:
            item = self.q.get()

            def on_update(state, item=item):
                with self.lock:
                    item["state"] = json.loads(json.dumps(state))
            try:
                out = pipeline.run(item["page"], name=item["id"], on_update=on_update)
                item["out"] = str(out)
            except Exception as e:
                item["error"] = str(e)

    def add_pages(self, paths):
        added = []
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in IMG_EXT or not p.exists():
                continue
            jid = f"{p.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
            item = {"id": jid, "page": str(p), "state": None, "queued": time.time()}
            with self.lock:
                self.jobs.append(item)
            self.q.put(item)
            added.append(jid)
        return added

    def pick_pages(self):
        w = webview.windows[0]
        res = w.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                   file_types=("Coloring pages (*.png;*.jpg;*.jpeg;*.webp)",))
        return self.add_pages(res or [])

    def snapshot(self):
        with self.lock:
            jobs = [{k: v for k, v in j.items()} for j in self.jobs]
        return {"stages": [{"key": k, "label": l, "where": WHERE.get(k, "This PC")} for k, l in STAGES],
                "jobs": jobs}

    def open_path(self, path):
        p = Path(path)
        if p.exists():
            os.startfile(str(p if p.is_dir() else p.parent))

    def open_file(self, path):
        if Path(path).exists():
            os.startfile(path)

    # ---- settings and keys
    def settings(self):
        return pipeline.load_settings()

    def save_settings(self, values):
        s = pipeline.load_settings()
        for k in ("budget_reel_usd", "budget_day_usd"):
            if k in values:
                s[k] = max(0.0, float(values[k]))
        if "use_judge" in values:
            s["use_judge"] = bool(values["use_judge"])
        if "models" in values and values["models"]:
            s["models"] = [m for m in values["models"] if m in HF_MODELS][:3] or s["models"]
        if "elements" in values:
            s["elements"] = [e.strip() for e in str(values["elements"]).split(",") if e.strip()][:4]
        if "max_paid_shots" in values:
            s["max_paid_shots"] = max(0, min(8, int(values["max_paid_shots"])))
        if "model_zooms" in values:
            s["model_zooms"] = [z for z in values["model_zooms"] if z in ("close", "medium", "wide")] or s["model_zooms"]
        keep = {k: s[k] for k in ("budget_reel_usd", "budget_day_usd", "use_judge", "models", "elements", "resolution",
                                  "model_zooms", "max_paid_shots") if k in s}
        (ROOT / "settings.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
        return s

    def key_status(self):
        import keyring
        return {"higgsfield": bool(keyring.get_password(HF_SERVICE, "api")),
                "anthropic": bool(keyring.get_password(AN_SERVICE, "api")) or bool(os.environ.get("ANTHROPIC_API_KEY"))}

    def save_key(self, which, key):
        """Keys go straight to Windows Credential Manager; never written to a file or log."""
        import keyring
        key = (key or "").strip()
        if which == "higgsfield":
            if ":" not in key:
                return {"ok": False, "msg": "Paste it as KEY_ID:KEY_SECRET"}
            keyring.set_password(HF_SERVICE, "api", key)
        elif which == "anthropic":
            if not key.startswith("sk-ant-"):
                return {"ok": False, "msg": "That doesn't look like an Anthropic key (sk-ant-…)"}
            keyring.set_password(AN_SERVICE, "api", key)
        else:
            return {"ok": False, "msg": "unknown key"}
        return {"ok": True, "msg": "Saved to Windows Credential Manager"}


def _bind_drop(window, api):
    def on_drop(e):
        files = e.get("dataTransfer", {}).get("files", [])
        paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
        if paths:
            api.add_pages(paths)

    window.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True)
    window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)


def main():
    api = Api()
    w = webview.create_window("Coloring Reel Machine", str(DATA / "app" / "ui.html"), js_api=api,
                              width=1040, height=820, min_size=(420, 560), background_color="#F6F3EE")
    webview.start(_bind_drop, (w, api))


if __name__ == "__main__":
    main()
