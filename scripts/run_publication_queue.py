"""Run leakage-free publication experiments within a bounded daily time budget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.publication_protocol import (
    CLEAN_ANCHORS,
    OBJECTIVES,
    SEEDS,
    SUBJECTS,
    PublicationRun,
    build_publication_runs,
)
from training.datasets import load_anchor_cache, validate_anchor_image_scope


RESULTS_ROOT = PROJECT_ROOT / "results" / "publication"
MANIFEST_PATH = RESULTS_ROOT / "experiment_manifest.csv"
PROTOCOL_PATH = RESULTS_ROOT / "protocol.json"
METRICS_ROOT = RESULTS_ROOT / "metrics"
SPLITS_ROOT = PROJECT_ROOT / "data" / "things_eeg2" / "splits"
MANIFEST_FIELDS = (
    "experiment_id",
    "setting",
    "objective",
    "seed",
    "train_subjects",
    "val_subject",
    "test_subject",
    "status",
    "epochs_completed",
    "best_epoch",
    "best_validation_mrr",
    "metrics_path",
    "updated_at",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_clean_anchor() -> dict[str, Any]:
    train = pd.read_csv(SPLITS_ROOT / "train_split.csv", usecols=["image_index"])
    val = pd.read_csv(SPLITS_ROOT / "val_split.csv", usecols=["image_index"])
    allowed = {int(value) for value in train["image_index"].unique()}
    forbidden = {int(value) for value in val["image_index"].unique()}
    if allowed & forbidden:
        raise ValueError("Train and validation image indices overlap")
    anchor_cache = load_anchor_cache(CLEAN_ANCHORS)
    validate_anchor_image_scope(
        anchor_cache,
        allowed_image_indices=allowed,
        forbidden_image_indices=forbidden,
    )
    return {
        "path": str(CLEAN_ANCHORS.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(CLEAN_ANCHORS),
        "allowed_image_count": len(allowed),
        "forbidden_image_count": len(forbidden),
    }


def read_manifest(runs: list[PublicationRun]) -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
            existing = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    records = {}
    for run in runs:
        record = {field: "" for field in MANIFEST_FIELDS}
        record.update(run.to_record())
        record["status"] = "pending"
        record.update(existing.get(run.experiment_id, {}))
        records[run.experiment_id] = record
    return records


def parse_csv_tuple(value: str | None, *, cast: type = str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    parsed = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Comma-separated filters must contain at least one value")
    return parsed


def add_experiment_prefix(runs: list[PublicationRun], prefix: str) -> list[PublicationRun]:
    if not prefix:
        return runs
    return [replace(run, experiment_id=f"{prefix}{run.experiment_id}") for run in runs]


def write_manifest(records: dict[str, dict[str, str]]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records.values())
    temp_path.replace(MANIFEST_PATH)


def checkpoint_state(experiment_id: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / "results" / "checkpoints" / experiment_id / "latest.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def train_run(run: PublicationRun, session_minutes: float, args: argparse.Namespace) -> dict[str, Any]:
    latest = checkpoint_state(run.experiment_id)
    command = [
        sys.executable,
        "training/train.py",
        "--experiment-id",
        run.experiment_id,
        "--split",
        "train",
        "--val-split",
        "val",
        "--train-subjects",
        ",".join(run.train_subjects),
        "--val-subjects",
        run.val_subject,
        "--objective",
        run.objective,
        "--seed",
        str(run.seed),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--temperature",
        str(args.temperature),
        "--prototype-alpha",
        str(args.prototype_alpha),
        "--prototype-beta",
        str(args.prototype_beta),
        "--lambda-anchor",
        str(args.lambda_anchor),
        "--lambda-prototype",
        str(args.lambda_prototype),
        "--prototype-warmup-epochs",
        str(args.prototype_warmup_epochs),
        "--anchors-path",
        str(CLEAN_ANCHORS),
        "--max-session-minutes",
        str(max(session_minutes, 0.1)),
    ]
    if args.limit_train_per_subject is not None:
        command.extend(["--limit-train-per-subject", str(args.limit_train_per_subject)])
    if args.limit_val_per_subject is not None:
        command.extend(["--limit-val-per-subject", str(args.limit_val_per_subject)])
    if latest is not None:
        command.append("--resume")
    return run_command(command)


def evaluate_run(run: PublicationRun, batch_size: int) -> dict[str, Any]:
    checkpoint = PROJECT_ROOT / "results" / "checkpoints" / run.experiment_id / "best.pt"
    metrics = run_command(
        [
            sys.executable,
            "training/validate.py",
            "--split",
            "test",
            "--subject",
            run.test_subject,
            "--batch-size",
            str(batch_size),
            "--checkpoint",
            str(checkpoint),
        ]
    )
    metrics.update(run.to_record())
    metrics["checkpoint"] = str(checkpoint.relative_to(PROJECT_ROOT))
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    output = METRICS_ROOT / f"{run.experiment_id}.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def reconcile_record(run: PublicationRun, record: dict[str, str]) -> None:
    state = checkpoint_state(run.experiment_id)
    metrics_path = METRICS_ROOT / f"{run.experiment_id}.json"
    if metrics_path.is_file():
        record["status"] = "completed"
        record["metrics_path"] = str(metrics_path.relative_to(PROJECT_ROOT))
    elif state is not None:
        record["status"] = "trained" if bool(state.get("completed")) else "paused"
        record["epochs_completed"] = str(state.get("epoch", ""))
        record["best_epoch"] = str(state.get("best_epoch", ""))
        record["best_validation_mrr"] = str(state.get("best_metric", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--prototype-alpha", type=float, default=0.7)
    parser.add_argument("--prototype-beta", type=float, default=0.99)
    parser.add_argument("--lambda-anchor", type=float, default=0.2)
    parser.add_argument("--lambda-prototype", type=float, default=0.0)
    parser.add_argument("--prototype-warmup-epochs", type=int, default=5)
    parser.add_argument("--limit-train-per-subject", type=int, default=None)
    parser.add_argument("--limit-val-per-subject", type=int, default=None)
    parser.add_argument("--subjects", default=None, help="Optional comma-separated test subjects.")
    parser.add_argument("--objectives", default=None, help="Optional comma-separated objectives.")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seeds.")
    parser.add_argument("--exclude-subject-specific", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Write an isolated selected-run manifest under results/pilot.",
    )
    parser.add_argument(
        "--pilot-prefix",
        default="PILOT-",
        help="Experiment ID prefix used with --pilot to avoid checkpoint collisions.",
    )
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hours <= 0:
        raise ValueError("--hours must be positive")
    selected_subjects = parse_csv_tuple(args.subjects) or SUBJECTS
    selected_objectives = parse_csv_tuple(args.objectives) or OBJECTIVES
    selected_seeds = parse_csv_tuple(args.seeds, cast=int) or SEEDS
    include_subject_specific = not args.exclude_subject_specific
    if args.pilot:
        global RESULTS_ROOT, MANIFEST_PATH, PROTOCOL_PATH, METRICS_ROOT
        RESULTS_ROOT = PROJECT_ROOT / "results" / "pilot"
        MANIFEST_PATH = RESULTS_ROOT / "experiment_manifest.csv"
        PROTOCOL_PATH = RESULTS_ROOT / "protocol.json"
        METRICS_ROOT = RESULTS_ROOT / "metrics"
        all_runs = build_publication_runs(
            subjects=selected_subjects,
            seeds=selected_seeds,
            objectives=selected_objectives,
            include_subject_specific=include_subject_specific,
        )
        all_runs = add_experiment_prefix(all_runs, args.pilot_prefix)
        runs = all_runs
    else:
        subject_filter = set(selected_subjects) if args.subjects is not None else None
        objective_filter = set(selected_objectives) if args.objectives is not None else None
        seed_filter = set(selected_seeds) if args.seeds is not None else None
        all_runs = build_publication_runs()
        runs = [
            run
            for run in all_runs
            if (subject_filter is None or run.test_subject in subject_filter)
            and (objective_filter is None or run.objective in objective_filter)
            and (seed_filter is None or run.seed in seed_filter)
            and (include_subject_specific or run.setting != "subject_specific")
        ]
    if not runs:
        raise ValueError("The requested filters selected no publication runs")
    anchor = validate_clean_anchor()
    records = read_manifest(all_runs)
    for run in all_runs:
        reconcile_record(run, records[run.experiment_id])
    write_manifest(records)
    PROTOCOL_PATH.write_text(
        json.dumps(
            {
                "rule": "8 train subjects, 1 distinct validation subject, 1 untouched test subject",
                "test_subjects": sorted({run.test_subject for run in all_runs}),
                "seeds": sorted({run.seed for run in all_runs}),
                "objectives": sorted({run.objective for run in all_runs}),
                "run_count": len(all_runs),
                "anchor": anchor,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if args.manifest_only:
        print(
            json.dumps(
                {
                    "manifest": str(MANIFEST_PATH),
                    "total_runs": len(all_runs),
                    "selected_runs": len(runs),
                    "anchor": anchor,
                },
                indent=2,
            )
        )
        return

    deadline = time.monotonic() + args.hours * 3600.0
    sessions = 0
    for run in runs:
        record = records[run.experiment_id]
        reconcile_record(run, record)
        if record["status"] == "completed":
            continue
        if args.max_runs is not None and sessions >= args.max_runs:
            break
        remaining_minutes = (deadline - time.monotonic()) / 60.0
        if remaining_minutes <= 1.0:
            break
        if record["status"] != "trained":
            record["status"] = "running"
            write_manifest(records)
            train_run(run, remaining_minutes, args)
            sessions += 1
            reconcile_record(run, record)
            write_manifest(records)
        if record["status"] == "trained":
            evaluate_run(run, args.batch_size)
            reconcile_record(run, record)
            write_manifest(records)
        if time.monotonic() >= deadline:
            break

    summary = {
        "sessions_started": sessions,
        "completed": sum(row["status"] == "completed" for row in records.values()),
        "paused": sum(row["status"] == "paused" for row in records.values()),
        "pending": sum(row["status"] == "pending" for row in records.values()),
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
