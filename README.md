# EEG-CLIP Project

Mapping THINGS-EEG2 brain signals into CLIP semantic image space for EEG-based image retrieval and optional visualization.

## Current Scope

- Dataset: THINGS-EEG2
- First data source: preprocessed EEG data
- Image representation: frozen CLIP image embeddings
- Core method: CAPE, cognitive anchor-guided prototype embeddings
- Primary output: ranked image retrieval from EEG trials

## Project Layout

- `data/things_eeg2/`: THINGS-EEG2 files, metadata, and splits
- `embeddings/`: cached CLIP, anchor, and EEG embeddings
- `models/`: EEG encoder, projection head, CAPE model, and losses
- `preprocessing/`: scripts for metadata, EEG preparation, CLIP extraction, and anchors
- `training/`: train, validate, and evaluate entrypoints
- `configs/`: shared and experiment-specific configuration
- `notebooks/`: exploration and analysis notebooks
- `results/`: checkpoints, logs, retrieval outputs, figures, and tables
- `utils/`: shared metrics, retrieval, visualization, and reproducibility helpers

## Data Preparation

Use the project virtual environment for local commands:

```powershell
.\.venv\Scripts\python.exe preprocessing\generate_splits.py
```

Use `--overwrite` only when intentionally regenerating existing split files.

This generates:

- `data/things_eeg2/splits/train_split.csv`
- `data/things_eeg2/splits/val_split.csv`
- `data/things_eeg2/splits/test_split.csv`
- `data/things_eeg2/splits/split_summary.json`

The split generator verifies the documented EEG tensor shapes, preserves
metadata index alignment, assigns train/validation membership at image level,
and expands repetitions into trial rows only after the image split is fixed.

Validate lazy EEG trial flattening with:

```powershell
.\.venv\Scripts\python.exe preprocessing\preprocess_eeg.py --split val --subject sub-01 --limit 16 --batch-size 8 --write-report
```

The trial loader indexes source EEG tensors as
`preprocessed_eeg_data[image_index, repetition_index]` and returns model-ready
EEG batches shaped `(batch, 17, 100)` without copying the full dataset into new
tensor files.

Extract frozen CLIP ViT-B/32 image embeddings with:

```powershell
.\.venv\Scripts\python.exe preprocessing\extract_clip_embeddings.py --splits train test --batch-size 64
```

Current caches:

- `embeddings/clip/train_image_embeddings.npz`
- `embeddings/clip/test_image_embeddings.npz`

Each cache stores L2-normalized 512-d image embeddings plus the metadata fields
needed to preserve EEG/image alignment during retrieval.

Build uniform category anchors with:

```powershell
.\.venv\Scripts\python.exe preprocessing\build_anchors.py
```

Current anchor cache:

- `embeddings/anchors/uniform_category_anchors.npz`

## Model Components

The default EEG encoder is implemented in `models/eegnet.py`. It accepts EEG
trial batches shaped `(batch, 17, 100)` and returns 256-dimensional latent
vectors without a classification head.

The projection head in `models/projection_head.py` maps EEGNet latents from
`256 -> 512 -> 512` and L2-normalizes the output for CLIP-space retrieval.

Quick validation:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from models import EEGCLIPModel; x=torch.randn(4,17,100); y=EEGCLIPModel()(x); print(y.shape, y.norm(dim=1))"
```

Validate training wiring with one optimizer step:

```powershell
.\.venv\Scripts\python.exe training\train.py --smoke-step --split train --subject sub-01 --limit 32 --batch-size 8 --objective direct
.\.venv\Scripts\python.exe training\train.py --smoke-step --split val --subject sub-01 --limit 32 --batch-size 8 --objective anchor
```

The direct objective uses multi-positive image IDs so repeated EEG trials of the
same image are not treated as negatives for each other.

Run a bounded EXP-001 sanity training job with:

```powershell
.\.venv\Scripts\python.exe training\train.py --experiment-id EXP-001-SANITY --split train --val-split val --subject sub-01 --limit-train 128 --limit-val 64 --batch-size 16 --max-epochs 2 --patience 2 --objective direct
```

Use a fresh `--experiment-id` for each run unless intentionally passing
`--overwrite-existing`. Checkpoints are written to `results/checkpoints/<experiment-id>/`
and epoch logs are written to `results/logs/<experiment-id>.jsonl`.
