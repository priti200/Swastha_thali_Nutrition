# Thali Dish, Weight & Nutrition Estimation Pipeline

```
image ─► GroundedSAM2 ─► food segments ─► encoder + kNN prototypes ─► dish label
             │                  │                                         │
             └── plate box ─────┴──────────► weight estimator ◄───────────┘
                                                    │ grams
                                                    ▼
                                             calorie lookup ─► kcal / protein / carbs / fat
```

| Stage | File | Status |
|---|---|---|
| 1. Segmentation (GroundingDINO boxes → SAM2 masks) | `thali/segmentation.py` | Pretrained, works now |
| 2. Matching (DINOv2 embeddings → kNN over prototypes) | `thali/matching.py` | Pretrained encoder; needs a few labelled crops per dish |
| 2b. Colour/texture-aware ViT | `thali/ct_vit.py` | Architecture only — train when the dataset is ready, then swap in for `FeatureEncoder` |
| 3. Weight estimation | `thali/estimation.py` | **Placeholder**: area × depth × density, plate used as scale. Replace with the learned network |
| 4. Calorie lookup | `data/dish_table.csv` | **Approximate values** — verify against IFCT 2017 and the mess recipes |

Until `data/prototypes.npz` exists, dish names come from GroundingDINO's matched prompt word.

## Setup (Windows, RTX 3050 6 GB)

Keep the virtual environment **outside OneDrive** so it isn't synced:

```powershell
python -m venv C:\venvs\thali
C:\venvs\thali\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

`sam2` installs from GitHub (needs `git`). If its optional CUDA extension fails to build on Windows,
that's fine — it only affects mask post-processing; set `$env:SAM2_BUILD_CUDA=0` before installing to skip it.
First run downloads model weights (~1 GB) from Hugging Face.

## Usage

```powershell
# 1. Put a few thali photos in data/samples/ and run the pipeline
python run_pipeline.py --image data/samples --save-crops

# 2. Sort outputs/crops/*.png into data/prototypes/<dish>/ (folder names = dish names in dish_table.csv)
python build_prototypes.py

# 3. Run again - labels now come from kNN matching
python run_pipeline.py --image data/samples

# Check the colour/texture ViT and its training memory on your GPU
python check_ct_vit.py --pretrained --batch 16
```

Outputs per image in `outputs/`: `<name>_result.png` (masks + labels + grams + kcal) and `<name>_result.json`.

## Tuning

All settings are in `config.yaml`: the GroundingDINO prompt (add your mess's dish names), detection
thresholds, the real plate diameter (measure it — it sets the pixel→cm scale), and the kNN `k` /
`min_similarity`.

## When the dataset arrives

1. Train `ColourTextureViT` on labelled crops (classification + embedding loss); wrap it with the
   `FeatureEncoder` interface (list of PIL crops → normalised embeddings).
2. Train a weight network with the `GeometricWeightEstimator.__call__` signature
   `(image, segment, label, plate_box) -> grams`, using the weighed-portion labels.
3. Evaluate: mask IoU (segmentation), top-1 accuracy (dish), MAE in grams and kcal (weight / calories).
