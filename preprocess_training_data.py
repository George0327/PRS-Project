"""
Binarize training dataset images to black/white (OTSU, inverted) and
save to TrainingData_BW/CNN letter Dataset/<class>/...

Run:
    python preprocess_training_data.py
"""

import os
import cv2
from pathlib import Path

VALID_CHARS = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"  # no I, O

def binarize_and_save(src_root: Path, dst_root: Path) -> None:
    for ch in VALID_CHARS:
        src_dir = src_root / ch
        dst_dir = dst_root / ch
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.is_dir():
            print(f"Skip missing: {src_dir}")
            continue

        count = 0
        for name in os.listdir(src_dir):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            src_path = src_dir / name
            img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize to 32x32 to match classifier preprocessing
            img32 = cv2.resize(img, (32, 32))
            # OTSU inverted threshold (white chars on black)
            _, bw = cv2.threshold(img32, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            # small open to clean artifacts
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

            out_path = dst_dir / name
            cv2.imwrite(str(out_path), bw)
            count += 1
        print(f"{ch}: saved {count} binarized images")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    src_root = script_dir / "TrainingData" / "CNN letter Dataset"
    dst_root = script_dir / "TrainingData_BW" / "CNN letter Dataset"
    dst_root.mkdir(parents=True, exist_ok=True)

    print("Binarizing training dataset -> TrainingData_BW ...")
    binarize_and_save(src_root, dst_root)
    print("Done.")
