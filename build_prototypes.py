"""Build kNN prototypes from labelled dish crops.

Folder layout (one folder per dish; names must match data/dish_table.csv):
    data/prototypes/roti/*.png
    data/prototypes/dal/*.png
    ...
Tip: run `run_pipeline.py --save-crops`, then sort outputs/crops/ into these folders, so the
prototypes look exactly like what the pipeline feeds the encoder.
"""
import argparse
from pathlib import Path

from PIL import Image

from thali.matching import FeatureEncoder, PrototypeKNN
from thali.pipeline import load_config, resolve_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--src", default="data/prototypes")
    args = ap.parse_args()

    cfg, root = load_config(args.config)
    images, labels = [], []
    for dish_dir in sorted(p for p in Path(args.src).iterdir() if p.is_dir()):
        files = [f for f in sorted(dish_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        images += [Image.open(f).convert("RGB") for f in files]
        labels += [dish_dir.name] * len(files)
        print(f"{dish_dir.name:<12}{len(files):>4} images")
    if not images:
        raise SystemExit(f"No images found under {args.src}/<dish>/ - see the docstring for the layout.")

    encoder = FeatureEncoder(cfg["classifier"]["encoder"], resolve_device(cfg["device"]))
    out = root / cfg["classifier"]["prototypes"]
    PrototypeKNN(encoder(images), labels).save(out)
    print(f"Saved {len(images)} prototypes to {out}")


if __name__ == "__main__":
    main()
