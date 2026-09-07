"""Adobe/IRIDAS .cube 3D LUT — same format as CubeLUT.cn / Premiere Lumetri."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_cube(path: Path) -> tuple[int, np.ndarray]:
    size = None
    samples: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("TITLE"):
            continue
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line.startswith("DOMAIN_"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            samples.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not size:
        raise ValueError(f"LUT_3D_SIZE missing: {path}")
    table = np.asarray(samples, dtype=np.float32).reshape(size, size, size, 3)
    return size, table


def apply_cube_rgb(rgb: np.ndarray, table: np.ndarray) -> np.ndarray:
    """rgb: float32 HxWx3 in 0..1, table indexed [R,G,B] -> RGB."""
    size = table.shape[0]
    x = np.clip(rgb, 0.0, 1.0) * (size - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = x - i0.astype(np.float32)
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    c000 = table[r0, g0, b0]
    c001 = table[r0, g0, b1]
    c010 = table[r0, g1, b0]
    c011 = table[r0, g1, b1]
    c100 = table[r1, g0, b0]
    c101 = table[r1, g0, b1]
    c110 = table[r1, g1, b0]
    c111 = table[r1, g1, b1]

    c00 = c000 * (1 - fb) + c001 * fb
    c01 = c010 * (1 - fb) + c011 * fb
    c10 = c100 * (1 - fb) + c101 * fb
    c11 = c110 * (1 - fb) + c111 * fb
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return c0 * (1 - fr) + c1 * fr


def write_cube(path: Path, table: np.ndarray, title: str) -> None:
    size = table.shape[0]
    lines = [
        f'TITLE "{title}"',
        "# StyleForge original LUT. Cube format compatible with CubeLUT.cn / Premiere.",
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    for r in range(size):
        for g in range(size):
            for b in range(size):
                rr, gg, bb = table[r, g, b]
                lines.append(f"{rr:.6f} {gg:.6f} {bb:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
