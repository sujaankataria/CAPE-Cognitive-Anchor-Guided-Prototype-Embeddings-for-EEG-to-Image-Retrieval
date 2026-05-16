"""Inspect THINGS-EEG2 preprocessed EEG, metadata, and image-set layout.

This script is intentionally read-only for dataset files. It writes only a short
inspection report under results/.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "things_eeg2"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed_data"
IMAGE_SET_ROOT = DATA_ROOT / "image_set"
METADATA_ROOT = DATA_ROOT / "metadata"
RESULTS_ROOT = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_ROOT / "dataset_inspection_report.txt"

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def add_line(lines: list[str], text: str = "") -> None:
    print(text)
    lines.append(text)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required directory: {path}")


def find_subject_dir(subject: str) -> Path:
    """Support both sub-01/file.npy and sub-01/sub-01/file.npy layouts."""
    direct_dir = PREPROCESSED_ROOT / subject
    nested_dir = direct_dir / subject

    for candidate in (nested_dir, direct_dir):
        if (
            (candidate / "preprocessed_eeg_training.npy").is_file()
            and (candidate / "preprocessed_eeg_test.npy").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find both preprocessed EEG files for "
        f"{subject} under {PREPROCESSED_ROOT}"
    )


def load_npy(path: Path) -> Any:
    require_file(path)
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        try:
            return value.item()
        except ValueError:
            return value
    return value


def numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return value
    return None


def describe_eeg_shape(array: np.ndarray) -> str:
    if array.ndim == 3:
        return "Likely trials x channels x time."
    if array.ndim == 4:
        return "Likely images x repetitions x channels x time."
    return f"Other structure: {array.ndim} dimensions."


def summarize_value(value: Any, limit: int = 5) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return f"scalar array dtype={value.dtype}, value={pformat(value.item(), compact=True, width=100)}"
        if value.size == 0:
            return f"array shape={value.shape}, dtype={value.dtype}, empty"
        preview = [value.flat[index].item() if hasattr(value.flat[index], "item") else value.flat[index] for index in range(min(limit, value.size))]
        return f"array shape={value.shape}, dtype={value.dtype}, first {len(preview)}={pformat(preview, compact=True, width=100)}"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        preview = list(value[:limit])
        return f"{type(value).__name__} length={len(value)}, first {len(preview)}={pformat(preview, compact=True, width=100)}"

    return pformat(value, compact=True, width=100)


def describe_numeric_array(prefix: str, array: np.ndarray, lines: list[str]) -> None:
    add_line(lines, f"  {prefix} shape: {array.shape}")
    add_line(lines, f"  {prefix} dtype: {array.dtype}")
    add_line(lines, f"  {prefix} shape interpretation: {describe_eeg_shape(array)}")

    arr = numeric_array(array)
    if arr is None:
        add_line(lines, f"  {prefix} numeric stats: unavailable; dtype is not numeric")
        return

    add_line(lines, f"  {prefix} min: {np.nanmin(arr):.6g}")
    add_line(lines, f"  {prefix} max: {np.nanmax(arr):.6g}")
    add_line(lines, f"  {prefix} mean: {np.nanmean(arr):.6g}")
    add_line(lines, f"  {prefix} std: {np.nanstd(arr):.6g}")


def describe_eeg_object(name: str, value: Any, lines: list[str]) -> None:
    add_line(lines, f"{name}:")
    add_line(lines, f"  type: {type(value).__name__}")

    if isinstance(value, Mapping):
        keys = list(value.keys())
        add_line(lines, "  shape: unavailable for dict/object mapping")
        add_line(lines, f"  keys ({len(keys)}): {keys}")
        for key in keys:
            entry = value[key]
            if isinstance(entry, np.ndarray):
                describe_numeric_array(str(key), entry, lines)
            else:
                add_line(lines, f"  {key}: {summarize_value(entry)}")
        add_line(lines)
        return

    if not isinstance(value, np.ndarray):
        add_line(lines, "  shape: unavailable; loaded object is not a numpy array")
        add_line(lines, f"  preview: {summarize_value(value)}")
        add_line(lines)
        return

    describe_numeric_array("array", value, lines)
    add_line(lines)


def find_metadata_file() -> Path:
    require_dir(METADATA_ROOT)
    candidates = sorted(path for path in METADATA_ROOT.glob("*.npy") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No .npy metadata file found in {METADATA_ROOT}")
    if len(candidates) > 1:
        print("Multiple metadata .npy files found; using the first sorted path:")
        for path in candidates:
            print(f"  {path}")
    return candidates[0]


def preview_sequence(value: Sequence[Any], limit: int = 5) -> list[str]:
    preview = []
    for item in list(value[:limit]):
        preview.append(pformat(item, compact=True, width=100))
    return preview


def describe_metadata(value: Any, lines: list[str]) -> None:
    add_line(lines, "Metadata:")
    add_line(lines, f"  type: {type(value).__name__}")

    if isinstance(value, np.ndarray):
        add_line(lines, f"  shape: {value.shape}")
        add_line(lines, f"  dtype: {value.dtype}")
        if value.dtype.names:
            add_line(lines, f"  structured fields: {list(value.dtype.names)}")
        if value.shape == () and value.dtype == object:
            add_line(lines, f"  scalar object preview: {pformat(value.item(), compact=True, width=100)}")
        else:
            flat_preview = value.flat[: min(5, value.size)]
            add_line(lines, "  first entries:")
            for entry in flat_preview:
                add_line(lines, f"    {pformat(entry, compact=True, width=100)}")
    elif isinstance(value, Mapping):
        add_line(lines, "  shape: unavailable for dict/object mapping")
        keys = list(value.keys())
        add_line(lines, f"  keys ({len(keys)}): {keys[:20]}")
        add_line(lines, "  first entries:")
        for key in keys[:5]:
            add_line(lines, f"    {key!r}: {summarize_value(value[key])}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        add_line(lines, f"  shape: sequence length {len(value)}")
        add_line(lines, "  first entries:")
        for entry in preview_sequence(value):
            add_line(lines, f"    {entry}")
    else:
        add_line(lines, "  shape: unavailable")
        add_line(lines, f"  preview: {pformat(value, compact=True, width=100)}")

    add_line(lines)


def count_images(path: Path) -> tuple[int, int]:
    require_dir(path)
    class_dirs = sum(1 for child in path.iterdir() if child.is_dir())
    image_files = sum(
        1
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )
    return image_files, class_dirs


def inspect_dataset(subject: str) -> str:
    lines: list[str] = []

    add_line(lines, "CAPE Dataset Inspection")
    add_line(lines, "=" * 24)
    add_line(lines, f"Project root: {PROJECT_ROOT}")
    add_line(lines, f"Subject: {subject}")
    add_line(lines)

    subject_dir = find_subject_dir(subject)
    add_line(lines, f"Subject directory: {subject_dir}")
    add_line(lines)

    training_eeg_path = subject_dir / "preprocessed_eeg_training.npy"
    test_eeg_path = subject_dir / "preprocessed_eeg_test.npy"
    training_eeg = load_npy(training_eeg_path)
    test_eeg = load_npy(test_eeg_path)

    describe_eeg_object("Training EEG", training_eeg, lines)
    describe_eeg_object("Test EEG", test_eeg, lines)

    metadata_path = find_metadata_file()
    add_line(lines, f"Metadata file: {metadata_path}")
    metadata = load_npy(metadata_path)
    describe_metadata(metadata, lines)

    training_image_count, training_class_count = count_images(IMAGE_SET_ROOT / "training_images")
    test_image_count, test_class_count = count_images(IMAGE_SET_ROOT / "test_images")
    add_line(lines, "Image set:")
    add_line(lines, f"  training image files: {training_image_count}")
    add_line(lines, f"  training class/category directories: {training_class_count}")
    add_line(lines, f"  test image files: {test_image_count}")
    add_line(lines, f"  test class/category directories: {test_class_count}")
    add_line(lines)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    add_line(lines, f"Saved report to: {REPORT_PATH}")

    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Subject folder to inspect, for example sub-01.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_dataset(args.subject)


if __name__ == "__main__":
    main()
