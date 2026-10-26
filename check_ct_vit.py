"""Sanity-check the colour/texture-aware ViT and measure training memory on this GPU.

    python check_ct_vit.py                 # random weights, quick
    python check_ct_vit.py --pretrained    # downloads ImageNet ViT weights
"""
import argparse

import torch

from thali.ct_vit import ColourTextureViT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--classes", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColourTextureViT(num_classes=args.classes, backbone=args.backbone, pretrained=args.pretrained).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device: {device}")

    x = torch.rand(args.batch, 3, 224, 224, device=device)
    y = torch.randint(0, args.classes, (args.batch,), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        emb, logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float(), y)
    loss.backward()
    opt.step()

    print(f"embedding: {tuple(emb.shape)} | logits: {tuple(logits.shape)} | loss: {loss.item():.3f}")
    if device == "cuda":
        print(f"peak training memory at batch {args.batch}: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
