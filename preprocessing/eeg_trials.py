"""Lazy EEG trial access backed by generated THINGS-EEG2 split CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHANNELS = 17
EXPECTED_TIMES = 100
CSV_DTYPES = {
    "subject_id": "string",
    "split": "string",
    "eeg_file": "string",
    "concept_id": "string",
    "concept_name": "string",
    "image_file": "string",
    "image_path": "string",
}


def load_npy_object(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing EEG file: {path}")
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def _normalize_subject_filter(subject: str | Sequence[str] | None) -> list[str] | None:
    if subject is None:
        return None
    if isinstance(subject, str):
        values = [part.strip() for part in subject.split(",")]
    else:
        values = [str(part).strip() for part in subject]
    subjects = [value for value in values if value]
    return subjects or None


def load_split_table(
    path: Path,
    subject: str | Sequence[str] | None = None,
    limit: int | None = None,
    limit_per_subject: int | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split CSV: {path}")

    table = pd.read_csv(path, dtype=CSV_DTYPES)
    expected_columns = {
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
    }
    missing = expected_columns - set(table.columns)
    if missing:
        raise ValueError(f"Split table missing required columns: {sorted(missing)}")

    subjects = _normalize_subject_filter(subject)
    if subjects is not None:
        table = table.loc[table["subject_id"].isin(subjects)].copy()
        if table.empty:
            raise ValueError(f"No rows for subject filter {subjects!r} in {path}")

    if limit_per_subject is not None:
        if limit_per_subject <= 0:
            raise ValueError("limit_per_subject must be positive")
        table = table.groupby("subject_id", sort=False, group_keys=False).head(limit_per_subject).copy()

    if limit is not None:
        table = table.head(limit).copy()

    return table.reset_index(drop=True)


class EEGTrialDataset(Dataset):
    """Return EEG trials as `(channels, time)` tensors from split CSV rows."""

    def __init__(
        self,
        split_csv: Path | str,
        *,
        subject: str | Sequence[str] | None = None,
        limit: int | None = None,
        limit_per_subject: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.table = load_split_table(
            self.split_csv,
            subject=subject,
            limit=limit,
            limit_per_subject=limit_per_subject,
        )
        self.dtype = dtype
        self._eeg_cache: dict[str, np.ndarray] = {}
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.table)

    def _resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _load_eeg_data(self, eeg_file: str) -> np.ndarray:
        if eeg_file not in self._eeg_cache:
            path = self._resolve_project_path(eeg_file)
            eeg_object = load_npy_object(path)
            if not isinstance(eeg_object, dict) or "preprocessed_eeg_data" not in eeg_object:
                raise ValueError(f"EEG file is not the expected dictionary format: {path}")

            eeg_data = eeg_object["preprocessed_eeg_data"]
            if eeg_data.ndim != 4:
                raise ValueError(f"Expected 4D EEG tensor, got shape {eeg_data.shape} from {path}")
            if eeg_data.shape[2:] != (EXPECTED_CHANNELS, EXPECTED_TIMES):
                raise ValueError(
                    "Unexpected EEG channel/time dimensions "
                    f"{eeg_data.shape[2:]} in {path}; expected {(EXPECTED_CHANNELS, EXPECTED_TIMES)}"
                )

            self._eeg_cache[eeg_file] = eeg_data
            times = eeg_object.get("times")
            times_summary = None
            if isinstance(times, np.ndarray):
                times_summary = {
                    "shape": tuple(times.shape),
                    "dtype": str(times.dtype),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                }

            self._metadata_cache[eeg_file] = {
                "ch_names": eeg_object.get("ch_names"),
                "times": times_summary,
                "shape": tuple(eeg_data.shape),
                "dtype": str(eeg_data.dtype),
            }

        return self._eeg_cache[eeg_file]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        eeg_file = str(row["eeg_file"])
        image_index = int(row["image_index"])
        repetition_index = int(row["repetition_index"])

        eeg_data = self._load_eeg_data(eeg_file)
        if image_index < 0 or image_index >= eeg_data.shape[0]:
            raise IndexError(f"image_index {image_index} out of bounds for shape {eeg_data.shape}")
        if repetition_index < 0 or repetition_index >= eeg_data.shape[1]:
            raise IndexError(f"repetition_index {repetition_index} out of bounds for shape {eeg_data.shape}")

        trial = eeg_data[image_index, repetition_index]
        if trial.shape != (EXPECTED_CHANNELS, EXPECTED_TIMES):
            raise ValueError(f"Flattened trial shape is {trial.shape}, expected {(EXPECTED_CHANNELS, EXPECTED_TIMES)}")

        return {
            "eeg": torch.as_tensor(trial, dtype=self.dtype),
            "global_trial_id": int(row["global_trial_id"]),
            "subject_id": str(row["subject_id"]),
            "split": str(row["split"]),
            "eeg_file": eeg_file,
            "image_index": image_index,
            "repetition_index": repetition_index,
            "concept_id": str(row["concept_id"]),
            "concept_name": str(row["concept_name"]),
            "image_file": str(row["image_file"]),
            "image_path": str(row["image_path"]),
        }

    def loaded_eeg_metadata(self) -> dict[str, dict[str, Any]]:
        return dict(self._metadata_cache)
