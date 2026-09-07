"""Bake StyleForge .cube LUTs (CubeLUT/Premiere compatible). Run from repo root."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.renderers.lut import write_cube
from app.renderers.photo_look import LUT_DIR, LOOKS, color_grade_rgb

SIZE = 17


def bake_one(style_id: str) -> Path:
    grid = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)
    table = np.zeros((SIZE, SIZE, SIZE, 3), dtype=np.float32)
    for r, rv in enumerate(grid):
        for g, gv in enumerate(grid):
            row = np.stack(
                [
                    np.full(SIZE, rv, dtype=np.float32),
                    np.full(SIZE, gv, dtype=np.float32),
                    grid,
                ],
                axis=-1,
            )[None, ...]
            table[r, g] = color_grade_rgb(row, style_id)[0]
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    path = LUT_DIR / LOOKS[style_id]["lut"]
    write_cube(path, table, LOOKS[style_id]["name"])
    return path


def main() -> None:
    for style_id in LOOKS:
        path = bake_one(style_id)
        print("wrote", path)


if __name__ == "__main__":
    main()
