"""Stages 3 and 4 - weight estimation and calorie lookup."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .segmentation import Segment

NUTRIENTS = ("kcal", "protein_g", "carbs_g", "fat_g")


class DishTable:
    """Per-dish nutrition (per 100 g) and the geometry priors the placeholder weight estimator uses."""

    def __init__(self, path: Path):
        with open(path, newline="", encoding="utf-8") as f:
            self.rows = {r["dish"]: {k: float(v) for k, v in r.items() if k != "dish"} for r in csv.DictReader(f)}

    def get(self, dish: str):
        return self.rows.get(dish)

    def nutrition(self, dish: str, grams: float) -> dict:
        row = self.get(dish)
        if row is None:
            return {n: None for n in NUTRIENTS}
        per_100 = (row["kcal_per_100g"], row["protein_g"], row["carbs_g"], row["fat_g"])
        return {n: round(v * grams / 100, 1) for n, v in zip(NUTRIENTS, per_100)}


class GeometricWeightEstimator:
    """Placeholder until the learned Weight Estimation Network is trained.

    grams = mask area (cm^2) x typical depth (cm) x density (g/cm^3), using the detected
    plate as the pixel-to-cm reference. The learned network will keep this call signature.
    """

    def __init__(self, dishes: DishTable, plate_diameter_cm: float, fallback_px_per_cm: float):
        self.dishes = dishes
        self.plate_diameter_cm = plate_diameter_cm
        self.fallback_px_per_cm = fallback_px_per_cm

    def px_per_cm(self, plate_box) -> float:
        if plate_box is None:
            return self.fallback_px_per_cm
        x0, y0, x1, y1 = plate_box
        return ((x1 - x0) + (y1 - y0)) / 2 / self.plate_diameter_cm

    def __call__(self, image: Image.Image, segment: Segment, label: str, plate_box) -> float:
        row = self.dishes.get(label) or {"depth_cm": 1.0, "density_g_cm3": 1.0}
        area_cm2 = segment.area / self.px_per_cm(plate_box) ** 2
        return round(float(area_cm2 * row["depth_cm"] * row["density_g_cm3"]), 1)
