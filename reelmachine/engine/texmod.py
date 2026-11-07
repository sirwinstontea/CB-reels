"""Ref-driven marker texture on flat plate ink (no Omni residual — that leaked ghosts/bleed).

Builds a seamless grain + soft edge darken from alcohol-marker reference stills, then
multiplies it onto plate pixels inside the ink mask only. Outside the mask = exact plate.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .paths import DATA
from .scene import OUT_W, OUT_H
from .spot import Q, _count
from .video import Writer, read_frames, probe, FPS

TILE = 512


def _unit(a):
    return (a - a.mean()) / (a.std() + 1e-6)


def _grain_tile(tex_root: Path | None = None, seed: int = 7) -> np.ndarray:
    """Fine paper tooth only (photo refs add lighting/wrinkles that look like foil)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (TILE, TILE)).astype(np.float32)
    tooth = cv2.GaussianBlur(a, (0, 0), 0.9)
    mid = cv2.GaussianBlur(a, (0, 0), 3.5)
    return _unit(0.7 * tooth + 0.3 * mid).astype(np.float32)


def _tiled(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    th, tw = tile.shape
    return np.tile(tile, (-(-h // th), -(-w // tw)))[:h, :w]


def apply(result_path, plate_path, inkmask_path, out_path, blur_sigma=3.0, strength=0.08):
    """API compat: `result_path` ignored (Omni unused). Texture from refs only."""
    return apply_refs(plate_path, inkmask_path, out_path, strength=strength)


def apply_refs(plate_path, inkmask_path, out_path, tex_root=None, strength=0.08, edge=0.06):
    """Multiply ref grain onto ink; darken slightly near ink edges (alcohol fringe)."""
    info = probe(plate_path)
    W, H = int(info["w"]), int(info["h"])
    n = _count(plate_path)
    masks = np.load(inkmask_path)["masks"]
    n = min(n, len(masks))
    g = max(1, 6 // Q)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    ink = np.array([cv2.dilate(m.astype(np.uint8), kk) for m in masks[:n]]) > 0
    grain = _tiled(_grain_tile(tex_root), H, W)

    tmp = Path(str(out_path) + ".part.mp4")
    wr = Writer(tmp, W, H)
    outside_mae = []
    for i, p in enumerate(read_frames(plate_path)):
        if i >= n:
            break
        p = p.astype(np.float32)
        w = cv2.resize(ink[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        w = cv2.GaussianBlur(np.clip(w, 0, 1), (0, 0), 0.8)
        # edge fringe: darker near boundary of ink
        edge_m = cv2.GaussianBlur(w, (0, 0), 2.5) - w
        edge_m = np.clip(edge_m, 0, 1)
        mod = 1.0 + strength * grain - edge * edge_m
        textured = np.clip(p * mod[..., None], 0, 255)
        out = textured * w[..., None] + p * (1.0 - w[..., None])
        # hard burn black lines from plate (lines are darkest pixels)
        lines = (np.min(p, axis=2) < 40).astype(np.float32)
        lines = cv2.dilate(lines, np.ones((2, 2), np.uint8))
        out = out * (1.0 - lines[..., None]) + p * lines[..., None]
        o = w < 0.05
        outside_mae.append(float(np.mean(np.abs(out - p)[o])) if o.any() else 0.0)
        wr.write(out.astype(np.uint8))
    wr.close()
    os.replace(tmp, out_path)
    return {
        "ok": float(np.mean(outside_mae)) < 1.0,
        "outside_mae": round(float(np.mean(outside_mae)), 4),
        "repaired": str(out_path),
        "src": probe(plate_path),
        "frames": n,
        "mode": "refs",
        "strength": strength,
    }
