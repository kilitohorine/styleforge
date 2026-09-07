"""Write small before/after strips into samples/ for README."""

from pathlib import Path

import cv2
import numpy as np

from app.renderers.photo_look import LOOKS, apply_look, make_comparison


def scene(kind: str, h: int = 180, w: int = 240) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    if kind == "portrait":
        img[:, :] = (70, 95, 140)
        cv2.circle(img, (w // 2, h // 2 - 10), 42, (160, 175, 210), -1)
        cv2.circle(img, (w // 2 - 12, h // 2 - 16), 6, (40, 40, 50), -1)
        cv2.circle(img, (w // 2 + 12, h // 2 - 16), 6, (40, 40, 50), -1)
    elif kind == "landscape":
        img[..., 0] = (40 + xx * 0.3).astype(np.uint8)
        img[..., 1] = (90 + yy * 0.4).astype(np.uint8)
        img[..., 2] = (160 - yy * 0.2).astype(np.uint8)
        img[h // 2 :, :] = (40, 90, 50)
    else:
        img[:, :] = (20, 18, 28)
        img[int(h * 0.65) :, :] = (18, 22, 30)
        cv2.rectangle(img, (40, 30), (70, 120), (180, 40, 90), -1)
        cv2.rectangle(img, (150, 50), (200, 130), (30, 160, 190), -1)
    return img


def main() -> None:
    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)
    for kind in ("portrait", "landscape", "night"):
        src = scene(kind)
        cv2.imwrite(str(out_dir / f"{kind}_src.jpg"), src)
        for sid in LOOKS:
            dst = apply_look(src, sid)
            cmp = make_comparison(src, dst)
            cv2.imwrite(str(out_dir / f"{kind}_{sid}.jpg"), cmp)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
