"""Run the thali pipeline on one image or a folder of images.

    python run_pipeline.py --image data/samples/thali1.jpg
    python run_pipeline.py --image data/samples --save-crops
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from thali.pipeline import ThaliPipeline, draw_result

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save-crops", action="store_true", help="save each segment crop (useful for building prototypes)")
    args = ap.parse_args()

    src = Path(args.image)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = ThaliPipeline.from_config(args.config)
    print(f"device: {pipe.device} | labels from: {'kNN prototypes' if pipe.knn else 'GroundingDINO (no prototypes yet)'}")

    for path in paths:
        image = Image.open(path).convert("RGB")
        result = pipe.run(image)

        print(f"\n{path.name}")
        print(f"  {'dish':<12}{'conf':>6}{'grams':>8}{'kcal':>8}{'prot':>7}{'carb':>7}{'fat':>7}")
        for i in result["items"]:
            nums = [f"{i[n]:>7.1f}" if i[n] is not None else f"{'-':>7}" for n in ("protein_g", "carbs_g", "fat_g")]
            kcal = f"{i['kcal']:>8.0f}" if i["kcal"] is not None else f"{'-':>8}"
            print(f"  {i['dish'][:12]:<12}{i['confidence']:>6.2f}{i['grams']:>8.0f}{kcal}{''.join(nums)}")
        t = result["totals"]
        print(f"  {'TOTAL':<18}{t['grams']:>8.0f}{t['kcal']:>8.0f}{t['protein_g']:>7.1f}{t['carbs_g']:>7.1f}{t['fat_g']:>7.1f}")

        draw_result(image, result).save(out / f"{path.stem}_result.png")
        with open(out / f"{path.stem}_result.json", "w", encoding="utf-8") as f:
            json.dump({k: result[k] for k in ("items", "totals", "plate_box")}, f, indent=2)
        if args.save_crops:
            crop_dir = out / "crops"
            crop_dir.mkdir(exist_ok=True)
            for n, (crop, item) in enumerate(zip(result["crops"], result["items"])):
                crop.save(crop_dir / f"{path.stem}_{n:02d}_{item['dish'].replace(' ', '-')}.png")

    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
