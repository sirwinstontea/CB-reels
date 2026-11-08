from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "pipeline.py"
t = p.read_text(encoding="utf-8")
t = t.replace(
    "from . import colorai, colors, handref, director, editor, final, ink, judge, rough, scene, shading, sound, spot, strokes",
    "from . import colorai, colors, handref, director, editor, final, ink, judge, rough, scene, shading, sound, spot, strokes, texture",
    1,
)
if '"apply_ink_texture"' not in t:
    t = t.replace(
        '    "hand_refs": str(ROOT / "assets" / "hand"),\n'
        '    "max_fallback_share": 0.5,\n'
        "}",
        '    "hand_refs": str(ROOT / "assets" / "hand"),\n'
        '    "marker_textures": str(ROOT / "assets" / "marker_textures"),\n'
        '    "apply_ink_texture": True,\n'
        '    "texture_model": "kling-omni-edit",\n'
        '    "max_fallback_share": 0.5,\n'
        "}",
        1,
    )
if "HAND_PROMPT" not in t:
    t = t.replace("PROMPT = ", "HAND_PROMPT = ", 1)
    t = t.replace("PROMPT.format(cap=cap)", "HAND_PROMPT.format(cap=cap)", 1)
p.write_text(t, encoding="utf-8")
print("pipe1 ok")
