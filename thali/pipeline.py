"""End-to-end pipeline: image -> segments -> dish labels -> grams -> nutrition."""
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from .estimation import NUTRIENTS, DishTable, GeometricWeightEstimator
from .matching import FeatureEncoder, PrototypeKNN
from .segmentation import GroundedSAM2


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_config(path) -> tuple[dict, Path]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f), path.parent


class ThaliPipeline:
    def __init__(self, cfg: dict, root: Path):
        self.device = resolve_device(cfg["device"])
        c = cfg["classifier"]
        self.segmenter = GroundedSAM2(cfg["segmentation"], self.device)
        self.encoder = FeatureEncoder(c["encoder"], self.device)
        self.knn = PrototypeKNN.load(root / c["prototypes"], k=c["k"], min_similarity=c["min_similarity"])
        self.dishes = DishTable(root / cfg["dish_table"])
        self.weigher = GeometricWeightEstimator(self.dishes, **cfg["weight"])

    @classmethod
    def from_config(cls, path):
        return cls(*load_config(path))

    def run(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        segments, plate_box = self.segmenter(image)
        crops = [s.crop(image) for s in segments]

        if self.knn is not None and segments:
            preds, source = self.knn.predict(self.encoder(crops)), "knn"
        else:
            preds, source = [(s.det_label, s.score) for s in segments], "grounding_dino"

        items = []
        for seg, (label, conf) in zip(segments, preds):
            grams = self.weigher(image, seg, label, plate_box)
            items.append({
                "dish": label,
                "confidence": round(float(conf), 3),
                "label_source": source,
                "box": [round(float(v), 1) for v in seg.box],
                "grams": grams,
                **self.dishes.nutrition(label, grams),
            })

        totals = {n: round(sum(i[n] or 0 for i in items), 1) for n in ("grams", *NUTRIENTS)}
        return {
            "items": items,
            "totals": totals,
            "plate_box": None if plate_box is None else [round(float(v), 1) for v in plate_box],
            "segments": segments,
            "crops": crops,
        }


def draw_result(image: Image.Image, result: dict) -> Image.Image:
    rng = np.random.default_rng(0)
    canvas = np.array(image.convert("RGB")).astype(np.float32)
    for seg in result["segments"]:
        colour = rng.integers(60, 255, 3)
        canvas[seg.mask] = 0.55 * canvas[seg.mask] + 0.45 * colour
    out = Image.fromarray(canvas.astype(np.uint8))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=max(14, out.width // 60))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    if result["plate_box"]:
        draw.rectangle(result["plate_box"], outline=(255, 255, 255), width=2)
    for item in result["items"]:
        x0, y0, x1, y1 = item["box"]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
        kcal = f"{item['kcal']:.0f} kcal" if item["kcal"] is not None else "? kcal"
        draw.text((x0 + 3, y0 + 3), f"{item['dish']} {item['grams']:.0f}g {kcal}", fill=(255, 255, 0), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return out
