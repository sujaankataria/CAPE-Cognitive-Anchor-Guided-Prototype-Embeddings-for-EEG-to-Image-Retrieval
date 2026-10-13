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
.\venv\Scripts\python.exe preprocessing\generate_splits.py
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
.\venv\Scripts\python.exe preprocessing\preprocess_eeg.py --split val --subject sub-01 --limit 16 --batch-size 8 --write-report
```

The trial loader indexes source EEG tensors as
`preprocessed_eeg_data[image_index, repetition_index]` and returns model-ready
EEG batches shaped `(batch, 17, 100)` without copying the full dataset into new
tensor files.

Extract frozen CLIP ViT-B/32 image embeddings with:

```powershell
.\venv\Scripts\python.exe preprocessing\extract_clip_embeddings.py --splits train test --batch-size 64
```

Current caches:

- `embeddings/clip/train_image_embeddings.npz`
- `embeddings/clip/test_image_embeddings.npz`

Each cache stores L2-normalized 512-d image embeddings plus the metadata fields
needed to preserve EEG/image alignment during retrieval.

Build uniform category anchors with:

```powershell
.\venv\Scripts\python.exe preprocessing\build_anchors.py
```

Current anchor cache:

- `embeddings/anchors/uniform_category_anchors.npz`

Build typicality-weighted category anchors only when a verified weights CSV is
available:

```powershell
.\venv\Scripts\python.exe preprocessing\build_anchors.py --anchor-type typicality --typicality-weights path\to\typicality_weights.csv
```

The typicality CSV must contain `image_index` and `typicality_weight` columns.
Weights must be finite and non-negative, and every training image index must
have one valid weight. The default weighted-anchor cache is
`embeddings/anchors/typicality_category_anchors.npz`. Uniform anchors remain the
default and are not overwritten by typicality runs.

Generate anchor geometry visualizations and neighborhood tables with:

```powershell
.\venv\Scripts\python.exe utils\visualization.py --anchors embeddings\anchors\uniform_category_anchors.npz --seed 42 --top-k 10
```

Current outputs:

- `results/figures/anchor_tsne.png`
- `results/figures/anchor_umap_style.png`
- `results/tables/anchor_tsne_coordinates.csv`
- `results/tables/anchor_umap_style_coordinates.csv`
- `results/tables/anchor_neighborhoods.csv`

The UMAP-style plot uses a deterministic nearest-neighbor spectral embedding
fallback because `umap-learn` is not currently installed in the project
environment.

## Model Components

The default EEG encoder is implemented in `models/eegnet.py`. It accepts EEG
trial batches shaped `(batch, 17, 100)` and returns 256-dimensional latent
vectors without a classification head.

The projection head in `models/projection_head.py` maps EEGNet latents from
`256 -> 512 -> 512` and L2-normalizes the output for CLIP-space retrieval.

Quick validation:

```powershell
.\venv\Scripts\python.exe -c "import torch; from models import EEGCLIPModel; x=torch.randn(4,17,100); y=EEGCLIPModel()(x); print(y.shape, y.norm(dim=1))"
```

Validate training wiring with one optimizer step:

```powershell
.\venv\Scripts\python.exe training\train.py --smoke-step --split train --subject sub-01 --limit 32 --batch-size 8 --objective direct
.\venv\Scripts\python.exe training\train.py --smoke-step --split val --subject sub-01 --limit 32 --batch-size 8 --objective anchor
```

The direct objective uses multi-positive image IDs so repeated EEG trials of the
same image are not treated as negatives for each other.

The training entrypoint currently supports four objectives:

- `direct`: direct EEG-image contrastive alignment
- `anchor`: uniform category-anchor contrastive alignment
- `prototype`: category-anchor alignment with EMA dynamic EEG prototypes
- `hybrid`: direct EEG-image alignment with weak anchor regularization

Run the CAPE-Hybrid rescue objective with:

```powershell
.\venv\Scripts\python.exe training\train.py --experiment-id HYBRID-SANITY --split train --val-split val --subject sub-01 --limit-train 128 --limit-val 64 --batch-size 16 --max-epochs 2 --patience 2 --objective hybrid --lambda-anchor 0.2 --lambda-prototype 0.0 --prototype-warmup-epochs 5
```

The default hybrid objective preserves direct image-level retrieval supervision
and adds a weak anchor regularizer:

```text
L = L_direct + 0.2 * L_anchor
```

Optional prototype regularization can be enabled with `--lambda-prototype`, but
it is disabled by default because the historical prototype-only objective
underperformed direct and anchor baselines.

Run the dynamic prototype baseline with:

```powershell
.\venv\Scripts\python.exe training\train.py --experiment-id EXP-003 --split train --val-split val --subject sub-01 --batch-size 128 --max-epochs 100 --patience 15 --objective prototype --learning-rate 0.0001 --weight-decay 0.0001 --temperature 0.07 --prototype-alpha 0.7 --prototype-beta 0.99 --seed 42
```

Evaluate a saved checkpoint on the official test split with:

```powershell
.\venv\Scripts\python.exe training\validate.py --split test --subject sub-01 --batch-size 128 --checkpoint results\checkpoints\EXP-003\best.pt
```

For subject-independent runs, pass comma-separated subject lists. This trains on
sub-01 through sub-03 and validates on held-out sub-10:

```powershell
.\venv\Scripts\python.exe training\train.py --experiment-id EXP-005 --split train --val-split val --train-subjects sub-01,sub-02,sub-03 --val-subjects sub-10 --batch-size 128 --max-epochs 100 --patience 15 --objective direct --learning-rate 0.0001 --weight-decay 0.0001 --temperature 0.07 --seed 42
```

The completed historical full subject-independent sweep uses sub-01 through
sub-09 for training and sub-10 for validation/test. Use the same subject lists
with `--objective direct`, `--objective anchor`, or `--objective prototype` to
reproduce EXP-007, EXP-008, or EXP-009 respectively. Use `--objective hybrid`
for the new rescue comparison.

Run the matching sub-10 subject-specific comparison with:

```powershell
.\venv\Scripts\python.exe training\train.py --experiment-id EXP-006 --split train --val-split val --subject sub-10 --batch-size 128 --max-epochs 100 --patience 15 --objective direct --learning-rate 0.0001 --weight-decay 0.0001 --temperature 0.07 --seed 42
```

Bounded multi-subject sanity checks should use per-subject limits to avoid
accidentally sampling only the first subject:

```powershell
.\venv\Scripts\python.exe training\train.py --experiment-id EXP-005-SANITY-BALANCED --split train --val-split val --train-subjects sub-01,sub-02 --val-subjects sub-10 --limit-train-per-subject 128 --limit-val-per-subject 128 --batch-size 32 --max-epochs 2 --patience 2 --objective direct --overwrite-existing
```

Train the EEG classification baseline with:

```powershell
.\venv\Scripts\python.exe training\train_classifier.py --experiment-id EXP-004 --split train --val-split val --subject sub-01 --batch-size 128 --max-epochs 100 --patience 15 --learning-rate 0.0001 --weight-decay 0.0001 --seed 42
```

This baseline trains an EEGNet classifier over the 1654 training categories.
For retrieval, classifier probabilities are multiplied by the uniform category
anchors to form a CLIP-space query. Official test classification accuracy is not
reported because official test concepts are disjoint from the training
categories, but official test retrieval can still be evaluated:

```powershell
.\venv\Scripts\python.exe training\train_classifier.py --eval-only --val-split test --subject sub-01 --batch-size 128 --checkpoint results\checkpoints\EXP-004\best.pt
```

Run a bounded EXP-001 sanity training job with:

```powershell
.\venv\Scripts\python.exe training\train.py --experiment-id EXP-001-SANITY --split train --val-split val --subject sub-01 --limit-train 128 --limit-val 64 --batch-size 16 --max-epochs 2 --patience 2 --objective direct
```

Use a fresh `--experiment-id` for each run unless intentionally passing
`--overwrite-existing`. Checkpoints are written to `results/checkpoints/<experiment-id>/`
and epoch logs are written to `results/logs/<experiment-id>.jsonl`.

## Staged Publication Reruns

The corrected publication protocol uses ten untouched test-subject folds. Each
fold assigns eight subjects to training, one distinct subject to validation,
and one subject to final testing. Uniform anchors are restricted to the 14,886
training-split images.

Create or verify the full 150-run manifest without starting training:

```powershell
.\venv\Scripts\python.exe scripts\run_publication_queue.py --manifest-only
```

Run an isolated sub-10 pilot before spending full publication GPU time:

```powershell
.\venv\Scripts\python.exe scripts\run_publication_queue.py --pilot --hours 2 --subjects sub-10 --objectives direct,anchor,hybrid --seeds 42,123,256,512,1024 --exclude-subject-specific --limit-train-per-subject 512 --limit-val-per-subject 256 --batch-size 64 --max-epochs 5 --patience 2
```

Pilot mode writes its manifest and metrics under `results/pilot/` and prefixes
experiment IDs with `PILOT-` so bounded screening checkpoints remain separate
from clean publication checkpoints.

Run a bounded daily session. The queue finishes the current epoch, writes
`latest.pt`, and resumes that run on the next invocation:

```powershell
.\venv\Scripts\python.exe scripts\run_publication_queue.py --hours 2
```

The recommended staged order for the rescue protocol is:

```powershell
# Stage A: primary direct-versus-hybrid comparison
.\venv\Scripts\python.exe scripts\run_publication_queue.py --hours 2 --objectives direct,hybrid --exclude-subject-specific

# Stage B: anchor and dynamic-prototype ablations
.\venv\Scripts\python.exe scripts\run_publication_queue.py --hours 2 --objectives anchor,prototype --exclude-subject-specific

# Stage C: subject-specific direct references
.\venv\Scripts\python.exe scripts\run_publication_queue.py --hours 2 --objectives direct
```

Stage C skips completed cross-subject direct runs and continues with the
remaining subject-specific references. Use `--subjects sub-01,sub-02` or
`--seeds 42` to narrow a day's queue without changing the full manifest.

Generate final subject-level statistics only after all clean runs complete:

```powershell
.\venv\Scripts\python.exe scripts\publication_analysis.py
```

Use `--allow-partial` only to inspect progress. Partial statistics must not be
copied into the paper.

For the completed Stage A direct-versus-hybrid comparison, run semantic-locality
analysis over the saved checkpoints with:

```powershell
.\venv\Scripts\python.exe scripts\semantic_locality_analysis.py --batch-size 128
```

This writes run metrics, subject-level means, paired semantic tests, and
subject-delta figures under `results/publication/semantic_locality/`. The
current clean analysis shows small directional hybrid gains on semantic-group
Top-K and CLIP target-similarity metrics, but these effects are also marginal.
