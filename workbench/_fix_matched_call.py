from pathlib import Path
p = Path(__file__).resolve().parent / "matched_shot.py"
t = p.read_text(encoding="utf-8")
t = t.replace(
    """    if target.exists():
        plan = colors.plan_from_target(pg, markers, cv2.cvtColor(cv2.imread(str(target)), cv2.COLOR_BGR2RGB))
    else:
        plan = colors.plan_fallback(pg, markers)
    (OUT / "colorplan.json").write_text(json.dumps({str(k): v for k, v in (plan.items() if isinstance(plan, dict) else [])}, default=str))
""",
    """    if target.exists():
        plan = colors.plan_from_target(pg, target, markers)
    else:
        plan = colors.plan_fallback(pg, markers)
    # plan may be dict rid->idx or richer
    dump = plan if isinstance(plan, dict) else {"plan": str(plan)}
    (OUT / "colorplan.json").write_text(json.dumps({str(k): v for k, v in dump.items()}, default=str))
""",
)
# remove monkeypatch at bottom
if "if not hasattr(colors, \"plan_from_target\")" in t:
    t = t.split('if __name__ == "__main__":')[0] + 'if __name__ == "__main__":\n    main()\n'
p.write_text(t, encoding="utf-8")
print("fixed")
