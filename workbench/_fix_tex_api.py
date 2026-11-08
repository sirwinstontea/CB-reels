from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "texture.py"
t = p.read_text(encoding="utf-8")
old = """        for name in names:
            p = folder / name
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                out.append(p)
"""
new = """        for name in names:
            # prefer 9:16 api/ stills (Omni rejects odd aspect ratios)
            stem = Path(name).stem
            p = folder / "api" / f"{stem}.jpg"
            if not p.exists():
                p = folder / name
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                out.append(p)
"""
if old not in t:
    raise SystemExit("refs loop missing")
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("texture.py prefers api refs")
