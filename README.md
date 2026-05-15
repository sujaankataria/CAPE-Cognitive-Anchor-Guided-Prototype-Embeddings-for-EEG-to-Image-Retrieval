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
