"""Stage 2 - Matching: encode each food crop, then kNN against labelled dish prototypes.

The encoder is a pretrained backbone for now (DINOv2). Once the colour/texture-aware ViT
(thali/ct_vit.py) is trained, it replaces FeatureEncoder with the same interface:
list of PIL crops in, L2-normalised (N, D) embeddings out.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FeatureEncoder:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=images[i : i + batch_size], return_tensors="pt").to(self.device)
            feats.append(F.normalize(self.model(**inputs).pooler_output, dim=-1).float().cpu())
        return torch.cat(feats).numpy() if feats else np.zeros((0, 0), dtype=np.float32)


class PrototypeKNN:
    def __init__(self, embeddings: np.ndarray, labels: list[str], k: int = 5, min_similarity: float = 0.35):
        self.embeddings = embeddings
        self.labels = labels
        self.k = k
        self.min_similarity = min_similarity

    @classmethod
    def load(cls, path: Path, **kwargs):
        if not Path(path).exists():
            return None
        data = np.load(path)
        return cls(data["embeddings"], data["labels"].tolist(), **kwargs)

    def save(self, path: Path):
        np.savez(path, embeddings=self.embeddings, labels=np.array(self.labels))

    def predict(self, feats: np.ndarray) -> list[tuple[str, float]]:
        """Similarity-weighted vote over the k nearest prototypes. Returns (label, confidence)."""
        sims = feats @ self.embeddings.T
        preds = []
        for row in sims:
            top = np.argsort(row)[::-1][: self.k]
            if row[top[0]] < self.min_similarity:
                preds.append(("unknown", float(row[top[0]])))
                continue
            votes = defaultdict(float)
            for j in top:
                votes[self.labels[j]] += max(row[j], 0.0)
            label = max(votes, key=votes.get)
            preds.append((label, votes[label] / sum(votes.values())))
        return preds
