"""Validate lazy EEG trial flattening from generated split CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.eeg_trials import EEGTrialDataset

SPLITS_ROOT = PROJECT_ROOT / "data" / "things_eeg2" / "splits"
RESULTS_ROOT = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_ROOT / "eeg_trial_flattening_report.json"


def split_csv_path(split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return SPLITS_ROOT / f"{split}_split.csv"


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    eeg = batch["eeg"]
    if not isinstance(eeg, torch.Tensor):
        raise TypeError(f"Expected EEG batch tensor, got {type(eeg).__name__}")
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG batch shape (batch, channels, time), got {tuple(eeg.shape)}")
    if tuple(eeg.shape[1:]) != (17, 100):
        raise ValueError(f"Expected trial shape (17, 100), got {tuple(eeg.shape[1:])}")

    return {
        "batch_shape": list(eeg.shape),
        "dtype": str(eeg.dtype),
        "device": str(eeg.device),
        "min": float(eeg.min().item()),
        "max": float(eeg.max().item()),
        "mean": float(eeg.mean().item()),
        "std": float(eeg.std(unbiased=False).item()),
    }


def validate_flat_trials(split: str, subject: str | None, limit: int, batch_size: int) -> dict[str, Any]:
    dataset = EEGTrialDataset(split_csv_path(split), subject=subject, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    batch_summary = summarize_batch(batch)

    first = dataset[0]
    report = {
        "split": split,
        "subject": subject,
        "requested_limit": limit,
        "dataset_rows": len(dataset),
        "batch_size": batch_size,
        "flattening_rule": "split CSV rows index EEG as preprocessed_eeg_data[image_index, repetition_index] -> (channels, time)",
        "trial_shape": [17, 100],
        "batch_summary": batch_summary,
        "first_trial": {
            "global_trial_id": first["global_trial_id"],
            "subject_id": first["subject_id"],
            "image_index": first["image_index"],
            "repetition_index": first["repetition_index"],
            "concept_id": first["concept_id"],
            "concept_name": first["concept_name"],
            "image_file": first["image_file"],
            "image_path": first["image_path"],
        },
        "loaded_eeg_metadata": dataset.loaded_eeg_metadata(),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Split CSV to validate.",
    )
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Subject to validate. Use an empty string to include all subjects.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="Maximum rows to load from the filtered split CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for DataLoader validation.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"Write validation report to {REPORT_PATH.relative_to(PROJECT_ROOT)}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject = args.subject or None
    report = validate_flat_trials(
        split=args.split,
        subject=subject,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.write_report:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
