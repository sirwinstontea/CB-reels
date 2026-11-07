"""Mask-locked marker texture via Higgsfield.

Flat plate ink is the shape; this stage asks Omni to add real marker texture only
inside the ink mask. spot.lock_ink restores everything outside that mask to the plate.
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import DATA

CATALOG = "catalog.json"


def catalog(root: Path | None = None) -> dict:
    root = Path(root or DATA / "assets" / "marker_textures")
    return json.loads((root / CATALOG).read_text(encoding="utf-8"))


def medium_of(marker: dict) -> str:
    return str(marker.get("medium") or "alcohol")


def label_of(medium: str, cat: dict | None = None) -> str:
    cat = cat or catalog()
    return (cat.get(medium) or {}).get("label") or medium.replace("_", " ")


def refs_for(mediums: list[str], root: Path | None = None, want_blend: bool = False) -> list[Path]:
    """Reference stills for the media used in a shot (blend refs only if asked)."""
    root = Path(root or DATA / "assets" / "marker_textures")
    cat = catalog(root)
    out, seen = [], set()
    for m in mediums:
        info = cat.get(m) or cat.get("alcohol") or {}
        folder = root / m if (root / m).is_dir() else root / "alcohol"
        names = list(info.get("refs") or [])
        if want_blend:
            names += list(info.get("blend_refs") or [])
        for name in names:
            p = folder / name
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                out.append(p)
    return out[:4]  # Omni image_urls cap


def prompt(mediums: list[str], want_blend: bool = False) -> str:
    cat = catalog()
    labels = sorted({label_of(m, cat) for m in mediums}) or ["alcohol marker"]
    kinds = ", ".join(labels)
    blend = (
        " Where two marker colors meet or overlap, blend them softly like wet alcohol ink "
        "(see the two-color blend reference); do not invent new colors outside the flat fills."
        if want_blend else
        " Keep each region's hue close to the flat fill already on the page."
    )
    return (
        f"Apply realistic {kinds} texture ON THE COLORED INK ONLY. "
        "LOCK: do not move, redraw, thicken, or erase any black line art; "
        "do not color blank paper, the desk, or anything outside the already-filled ink regions; "
        "do not change framing, camera, or lighting. "
        "TEXTURE: make paper grain, soft feathered stroke edges, translucent dye soak, "
        "subtle mottle, and slightly darker overlaps clearly visible — match the reference photos "
        f"of {kinds} on paper. Flat digital fills should become convincing marker ink on paper."
        f"{blend}"
    )


def shot_mediums(markers: list[dict], indices: list[int]) -> list[str]:
    out = []
    for i in indices:
        if 0 <= i < len(markers) and not markers[i].get("opaque"):
            out.append(medium_of(markers[i]))
    return sorted(set(out)) or ["alcohol"]
