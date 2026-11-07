"""Make one reel from the command line (the app does the same thing behind its window).

python make_reel.py <page.png> [job name]
"""
import sys

from engine import pipeline

if __name__ == "__main__":
    last = {}

    def show(state):
        for k, st in state["stages"].items():
            key = (st["status"], st["msg"])
            if last.get(k) != key:
                last[k] = key
                print(f"[{st['status']:8s}] {st['label']}: {st['msg']}", flush=True)

    out = pipeline.run(sys.argv[1], name=sys.argv[2] if len(sys.argv) > 2 else None, on_update=show)
    print("reel:", out)
