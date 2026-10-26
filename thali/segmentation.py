"""Stage 1 - GroundedSAM2: GroundingDINO predicts boxes, SAM2 turns each box into a mask."""
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou, nms


@dataclass
class Segment:
    box: np.ndarray   # [x0, y0, x1, y1] in pixels
    mask: np.ndarray  # bool, H x W
    score: float      # GroundingDINO box confidence
    det_label: str    # phrase GroundingDINO matched, e.g. "dal"

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def crop(self, image: Image.Image, pad: int = 4) -> Image.Image:
        """Crop around the mask with everything outside the mask blacked out."""
        h, w = self.mask.shape
        x0, y0, x1, y1 = self.box.round().astype(int)
        x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1, y1 = min(x1 + pad, w), min(y1 + pad, h)
        rgb = np.array(image.convert("RGB"))[y0:y1, x0:x1].copy()
        rgb[~self.mask[y0:y1, x0:x1]] = 0
        return Image.fromarray(rgb)


class GroundedSAM2:
    def __init__(self, cfg: dict, device: str):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg["grounding_dino"])
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["grounding_dino"]).to(device).eval()
        self.sam = SAM2ImagePredictor.from_pretrained(cfg["sam2"], device=device)

    @torch.inference_mode()
    def detect(self, image: Image.Image, prompt: str):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.dino(**inputs)
        r = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.cfg["box_threshold"],
            text_threshold=self.cfg["text_threshold"],
            target_sizes=[image.size[::-1]],
        )[0]
        labels = r.get("text_labels", r["labels"])
        return r["boxes"].float().cpu(), r["scores"].float().cpu(), list(labels)

    def plate_box(self, image: Image.Image):
        boxes, scores, _ = self.detect(image, self.cfg["plate_prompt"])
        if len(boxes) == 0:
            return None
        return boxes[scores.argmax()].numpy()

    def __call__(self, image: Image.Image):
        """Returns (segments, plate_box). plate_box is None if no plate was found."""
        image = image.convert("RGB")
        plate = self.plate_box(image)
        boxes, scores, labels = self.detect(image, self.cfg["food_prompt"])

        if plate is not None and len(boxes):
            plate_t = torch.as_tensor(plate)[None]
            plate_area = (plate_t[0, 2] - plate_t[0, 0]) * (plate_t[0, 3] - plate_t[0, 1])
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = (areas < self.cfg["max_box_fraction_of_plate"] * plate_area) & (box_iou(boxes, plate_t)[:, 0] < 0.8)
            boxes, scores, labels = boxes[keep], scores[keep], [l for l, k in zip(labels, keep) if k]

        keep = nms(boxes, scores, self.cfg["nms_iou"]).tolist() if len(boxes) else []
        boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]
        if len(boxes) == 0:
            return [], plate

        amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), amp:
            self.sam.set_image(np.array(image))
            masks, _, _ = self.sam.predict(box=boxes.numpy(), multimask_output=False)
        masks = masks.reshape(len(boxes), *masks.shape[-2:]) > 0

        segments = [
            Segment(box=b.numpy(), mask=m, score=float(s), det_label=l.strip())
            for b, m, s, l in zip(boxes, masks, scores, labels)
        ]
        return segments, plate
