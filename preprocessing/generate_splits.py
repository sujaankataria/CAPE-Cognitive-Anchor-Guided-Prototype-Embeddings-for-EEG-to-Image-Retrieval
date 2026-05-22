"""Generate deterministic THINGS-EEG2 image-level split CSVs.

The source EEG tensors remain untouched. This script expands image-level splits
into trial rows only after assigning each image to train, validation, or test.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "things_eeg2"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed_data"
IMAGE_SET_ROOT = DATA_ROOT / "image_set"
METADATA_PATH = DATA_ROOT / "metadata" / "image_metadata.npy"
SPLITS_ROOT = DATA_ROOT / "splits"

TRAIN_EEG_SHAPE = (16540, 4, 17, 100)
TEST_EEG_SHAPE = (200, 80, 17, 100)
SEED = 42
VAL_FRACTION = 0.10

CSV_COLUMNS = [
    "global_trial_id",
    "subject_id",
    "split",
    "eeg_file",
    "image_index",
    "repetition_index",
    "concept_id",
    "concept_name",
    "image_file",
    "image_path",
]


def load_npy_object(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def find_subject_dirs(subjects: list[str] | None = None) -> dict[str, Path]:
    if not PREPROCESSED_ROOT.is_dir():
        raise FileNotFoundError(f"Missing preprocessed EEG directory: {PREPROCESSED_ROOT}")

    candidates = sorted(path for path in PREPROCESSED_ROOT.iterdir() if path.is_dir())
    if subjects is not None:
        requested = set(subjects)
        candidates = [PREPROCESSED_ROOT / subject for subject in subjects]
    else:
        requested = None

    subject_dirs: dict[str, Path] = {}
    for candidate in candidates:
        subject_id = candidate.name
        possible_dirs = [candidate / subject_id, candidate]
        for subject_dir in possible_dirs:
            train_file = subject_dir / "preprocessed_eeg_training.npy"
            test_file = subject_dir / "preprocessed_eeg_test.npy"
            if train_file.is_file() and test_file.is_file():
                subject_dirs[subject_id] = subject_dir
                break

    if requested is not None:
        missing = sorted(requested - set(subject_dirs))
        if missing:
            raise FileNotFoundError(f"Missing EEG files for requested subjects: {missing}")

    if not subject_dirs:
        raise FileNotFoundError(f"No subject EEG files found under {PREPROCESSED_ROOT}")

    return subject_dirs


def verify_subject_tensor_shapes(subject_dirs: dict[str, Path]) -> dict[str, dict[str, str]]:
    summary: dict[str, dict[str, str]] = {}
    for subject_id, subject_dir in subject_dirs.items():
        train_file = subject_dir / "preprocessed_eeg_training.npy"
        test_file = subject_dir / "preprocessed_eeg_test.npy"
        train_eeg = load_npy_object(train_file)
        test_eeg = load_npy_object(test_file)

        for name, eeg, expected_shape in (
            ("training", train_eeg, TRAIN_EEG_SHAPE),
            ("test", test_eeg, TEST_EEG_SHAPE),
        ):
            if not isinstance(eeg, dict) or "preprocessed_eeg_data" not in eeg:
                raise ValueError(f"{subject_id} {name} EEG is not the expected dictionary format")
            actual_shape = tuple(eeg["preprocessed_eeg_data"].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{subject_id} {name} EEG shape {actual_shape} does not match {expected_shape}"
                )

        summary[subject_id] = {
            "training_eeg_file": str(train_file.relative_to(PROJECT_ROOT)),
            "training_shape": str(TRAIN_EEG_SHAPE),
            "training_dtype": str(train_eeg["preprocessed_eeg_data"].dtype),
            "test_eeg_file": str(test_file.relative_to(PROJECT_ROOT)),
            "test_shape": str(TEST_EEG_SHAPE),
            "test_dtype": str(test_eeg["preprocessed_eeg_data"].dtype),
        }

    return summary


def load_metadata() -> dict[str, list[str]]:
    metadata = load_npy_object(METADATA_PATH)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dictionary: {METADATA_PATH}")

    required_keys = [
        "train_img_files",
        "train_img_concepts",
        "test_img_files",
        "test_img_concepts",
    ]
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"Metadata missing required key: {key}")

    if len(metadata["train_img_files"]) != TRAIN_EEG_SHAPE[0]:
        raise ValueError("Training metadata length does not match training EEG image count")
    if len(metadata["train_img_concepts"]) != TRAIN_EEG_SHAPE[0]:
        raise ValueError("Training concept metadata length does not match training EEG image count")
    if len(metadata["test_img_files"]) != TEST_EEG_SHAPE[0]:
        raise ValueError("Test metadata length does not match test EEG image count")
    if len(metadata["test_img_concepts"]) != TEST_EEG_SHAPE[0]:
        raise ValueError("Test concept metadata length does not match test EEG image count")

    return metadata


def split_concept(concept: str) -> tuple[str, str]:
    concept_id, _, concept_name = concept.partition("_")
    if not concept_id or not concept_name:
        raise ValueError(f"Unexpected concept format: {concept!r}")
    return concept_id, concept_name


def resolve_image_path(source_split: str, concept: str, image_file: str) -> Path:
    path = IMAGE_SET_ROOT / source_split / concept / image_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing image file for metadata row: {path}")
    return path


def make_image_level_split(num_images: int, seed: int, val_fraction: float) -> tuple[set[int], set[int]]:
    val_count = int(round(num_images * val_fraction))
    if not 0 < val_count < num_images:
        raise ValueError(f"Invalid validation count {val_count} for {num_images} images")

    rng = np.random.default_rng(seed)
    shuffled = np.arange(num_images)
    rng.shuffle(shuffled)
    val_indices = set(int(index) for index in shuffled[:val_count])
    train_indices = set(int(index) for index in shuffled[val_count:])
    return train_indices, val_indices


def row_for_trial(
    *,
    global_trial_id: int,
    subject_id: str,
    split: str,
    eeg_file: Path,
    image_index: int,
    repetition_index: int,
    concept: str,
    image_file: str,
    source_image_split: str,
) -> dict[str, str | int]:
    concept_id, concept_name = split_concept(concept)
    image_path = resolve_image_path(source_image_split, concept, image_file)
    return {
        "global_trial_id": global_trial_id,
        "subject_id": subject_id,
        "split": split,
        "eeg_file": str(eeg_file.relative_to(PROJECT_ROOT)),
        "image_index": image_index,
        "repetition_index": repetition_index,
        "concept_id": concept_id,
        "concept_name": concept_name,
        "image_file": image_file,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, str | int]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def generate_splits(subjects: list[str] | None, overwrite: bool) -> dict[str, Any]:
    metadata = load_metadata()
    subject_dirs = find_subject_dirs(subjects)
    tensor_summary = verify_subject_tensor_shapes(subject_dirs)

    train_image_indices, val_image_indices = make_image_level_split(
        TRAIN_EEG_SHAPE[0],
        seed=SEED,
        val_fraction=VAL_FRACTION,
    )
    if train_image_indices & val_image_indices:
        raise AssertionError("Train and validation image indices overlap")

    split_rows = {"train": [], "val": [], "test": []}
    global_trial_id = 0

    for subject_id, subject_dir in subject_dirs.items():
        training_eeg_file = subject_dir / "preprocessed_eeg_training.npy"
        test_eeg_file = subject_dir / "preprocessed_eeg_test.npy"

        for image_index in sorted(train_image_indices | val_image_indices):
            split = "val" if image_index in val_image_indices else "train"
            concept = metadata["train_img_concepts"][image_index]
            image_file = metadata["train_img_files"][image_index]
            for repetition_index in range(TRAIN_EEG_SHAPE[1]):
                split_rows[split].append(
                    row_for_trial(
                        global_trial_id=global_trial_id,
                        subject_id=subject_id,
                        split=split,
                        eeg_file=training_eeg_file,
                        image_index=image_index,
                        repetition_index=repetition_index,
                        concept=concept,
                        image_file=image_file,
                        source_image_split="training_images",
                    )
                )
                global_trial_id += 1

        for image_index in range(TEST_EEG_SHAPE[0]):
            concept = metadata["test_img_concepts"][image_index]
            image_file = metadata["test_img_files"][image_index]
            for repetition_index in range(TEST_EEG_SHAPE[1]):
                split_rows["test"].append(
                    row_for_trial(
                        global_trial_id=global_trial_id,
                        subject_id=subject_id,
                        split="test",
                        eeg_file=test_eeg_file,
                        image_index=image_index,
                        repetition_index=repetition_index,
                        concept=concept,
                        image_file=image_file,
                        source_image_split="test_images",
                    )
                )
                global_trial_id += 1

    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(SPLITS_ROOT / "train_split.csv", split_rows["train"], overwrite=overwrite)
    write_csv(SPLITS_ROOT / "val_split.csv", split_rows["val"], overwrite=overwrite)
    write_csv(SPLITS_ROOT / "test_split.csv", split_rows["test"], overwrite=overwrite)

    train_concepts = Counter(metadata["train_img_concepts"][index] for index in train_image_indices)
    val_concepts = Counter(metadata["train_img_concepts"][index] for index in val_image_indices)
    concept_overlap = sorted(set(train_concepts) & set(val_concepts))

    summary = {
        "seed": SEED,
        "validation_fraction": VAL_FRACTION,
        "subjects": sorted(subject_dirs),
        "split_protocol": "deterministic image-level split; repetitions are expanded only after image assignment",
        "source_metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        "tensor_summary": tensor_summary,
        "image_counts": {
            "train_images": len(train_image_indices),
            "val_images": len(val_image_indices),
            "test_images": TEST_EEG_SHAPE[0],
        },
        "repetitions_per_image": {
            "train": TRAIN_EEG_SHAPE[1],
            "val": TRAIN_EEG_SHAPE[1],
            "test": TEST_EEG_SHAPE[1],
        },
        "trial_row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "leakage_checks": {
            "train_val_image_overlap": 0,
            "validation_from_training_images_only": True,
            "official_test_untouched": True,
            "train_val_concept_overlap_count": len(concept_overlap),
            "train_val_concept_overlap_note": "Concept overlap is expected because the documented rule forbids image/repetition leakage, not category overlap.",
        },
        "outputs": {
            "train": str((SPLITS_ROOT / "train_split.csv").relative_to(PROJECT_ROOT)),
            "val": str((SPLITS_ROOT / "val_split.csv").relative_to(PROJECT_ROOT)),
            "test": str((SPLITS_ROOT / "test_split.csv").relative_to(PROJECT_ROOT)),
            "summary": str((SPLITS_ROOT / "split_summary.json").relative_to(PROJECT_ROOT)),
        },
    }

    summary_path = SPLITS_ROOT / "split_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to include. Defaults to all detected subjects.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split CSVs and summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_splits(subjects=args.subjects, overwrite=args.overwrite)
    print("Generated THINGS-EEG2 split files")
    print(f"  subjects: {', '.join(summary['subjects'])}")
    print(f"  train rows: {summary['trial_row_counts']['train']}")
    print(f"  val rows: {summary['trial_row_counts']['val']}")
    print(f"  test rows: {summary['trial_row_counts']['test']}")
    print(f"  summary: {summary['outputs']['summary']}")


if __name__ == "__main__":
    main()
