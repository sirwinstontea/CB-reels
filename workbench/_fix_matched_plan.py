from pathlib import Path
p = Path(__file__).resolve().parent / "matched_shot.py"
t = p.read_text(encoding="utf-8")
old = """    if target.exists():
        plan = colors.plan_from_target(pg, target, markers)
    else:
        plan = colors.plan_fallback(pg, markers)
"""
new = """    if target.exists():
        plan, gate = colors.plan_from_target(pg, target, markers, min_align=0.55)
        print("color gate", gate, flush=True)
        if plan is None:
            plan = colors.plan_fallback(pg, markers)
    else:
        plan = colors.plan_fallback(pg, markers)
"""
if old not in t:
    raise SystemExit("block missing")
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("ok")
