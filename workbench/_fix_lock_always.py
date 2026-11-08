from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "spot.py"
t = p.read_text(encoding="utf-8")
old = """    if tmp:
        if rep["ok"] and tmp.exists():
            os.replace(tmp, out_path)
            rep["repaired"] = str(out_path)
        elif tmp.exists():
            tmp.unlink()
    return rep
"""
# only change the lock_ink ending — find unique context
idx = t.find("def lock_ink(")
if idx < 0:
    raise SystemExit("no lock_ink")
tail = t[idx:]
old2 = """    if tmp:
        if rep["ok"] and tmp.exists():
            os.replace(tmp, out_path)
            rep["repaired"] = str(out_path)
        elif tmp.exists():
            tmp.unlink()
    return rep
"""
new2 = """    # always keep the mask composite: plate outside ink, model inside (even if spot fails)
    if tmp and tmp.exists():
        os.replace(tmp, out_path)
        rep["repaired"] = str(out_path)
    return rep
"""
if old2 not in tail:
    raise SystemExit("lock_ink footer missing")
t = t[:idx] + tail.replace(old2, new2, 1)
p.write_text(t, encoding="utf-8")
print("lock_ink always writes repaired")
