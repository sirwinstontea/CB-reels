from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "texmod.py"
t = p.read_text(encoding="utf-8")
# replace _grain_tile with fine procedural tooth only
old = '''def _grain_tile(tex_root: Path | None = None, seed: int = 7) -> np.ndarray:
    """Zero-mean float32 tile ~[-1,1] from ref stills' high-pass, fallback to noise."""
    root = Path(tex_root or DATA / "assets" / "marker_textures" / "alcohol")
    rng = np.random.default_rng(seed)
    acc = np.zeros((TILE, TILE), np.float32)
    n = 0
    for p in list(root.glob("*.png")) + list(root.glob("*.jpg")):
        if p.parent.name == "api":
            continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None or min(im.shape) < 64:
            continue
        im = cv2.resize(im, (TILE, TILE), interpolation=cv2.INTER_AREA).astype(np.float32)
        hp = im - cv2.GaussianBlur(im, (0, 0), 6.0)
        acc += _unit(hp)
        n += 1
    if n == 0:
        acc = rng.normal(0, 1, (TILE, TILE)).astype(np.float32)
        acc = cv2.GaussianBlur(acc, (0, 0), 1.2) - cv2.GaussianBlur(acc, (0, 0), 5.0)
    return _unit(acc / max(1, n)).astype(np.float32)
'''
new = '''def _grain_tile(tex_root: Path | None = None, seed: int = 7) -> np.ndarray:
    """Fine paper tooth only (photo refs add lighting/wrinkles that look like foil)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (TILE, TILE)).astype(np.float32)
    tooth = cv2.GaussianBlur(a, (0, 0), 0.9)
    mid = cv2.GaussianBlur(a, (0, 0), 3.5)
    return _unit(0.7 * tooth + 0.3 * mid).astype(np.float32)
'''
if old not in t:
    raise SystemExit("grain fn missing")
t = t.replace(old, new, 1)
# softer defaults: strength 0.08, edge 0.06 — alcohol is subtle
t = t.replace("strength=0.14, edge=0.10", "strength=0.08, edge=0.06")
t = t.replace("strength=0.14)", "strength=0.08)")
p.write_text(t, encoding="utf-8")
print("grain fixed")
