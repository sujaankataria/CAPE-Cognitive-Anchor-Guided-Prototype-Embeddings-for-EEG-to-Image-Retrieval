"""Generate subject-level publication statistics from completed clean runs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.publication_protocol import OBJECTIVES, SEEDS, SUBJECTS


RESULTS_ROOT = PROJECT_ROOT / "results" / "publication"
MANIFEST_PATH = RESULTS_ROOT / "experiment_manifest.csv"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
PRIMARY_METRIC = "mrr"
SECONDARY_METRICS = ("top1", "top5", "top10")
PRIMARY_BASELINE = "direct"
PRIMARY_METHOD = "hybrid"


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = abs(float(values.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistics.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int = 20260608,
    iterations: int = 20000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(iterations, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * pvalue)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def load_completed_metrics(allow_partial: bool) -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing publication manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"seed": int})
    expected = len(SUBJECTS) * len(SEEDS) * (len(OBJECTIVES) + 1)
    completed = manifest.loc[manifest["status"] == "completed"]
    if not allow_partial and len(completed) != expected:
        raise RuntimeError(
            f"Publication analysis requires all {expected} publication runs; found {len(completed)}"
        )
    rows = []
    for record in completed.to_dict("records"):
        metrics_path = PROJECT_ROOT / str(record["metrics_path"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(metrics)
    if not rows:
        raise RuntimeError("No completed publication metrics are available")
    return pd.DataFrame(rows)


def build_analysis(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cross_metrics = metrics.loc[metrics["setting"] == "cross_subject"].copy()
    subject_means = (
        cross_metrics.groupby(["test_subject", "objective"], as_index=False)[
            [PRIMARY_METRIC, *SECONDARY_METRICS]
        ]
        .mean()
        .sort_values(["test_subject", "objective"])
    )
    seed_counts = (
        cross_metrics.groupby(["test_subject", "objective"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "seed_count"})
    )
    subject_means = subject_means.merge(
        seed_counts,
        on=["test_subject", "objective"],
        how="left",
    )
    subject_means["complete_seed_set"] = subject_means["seed_count"] == len(SEEDS)
    tests = []
    raw_secondary = {}
    for metric in (PRIMARY_METRIC, *SECONDARY_METRICS):
        complete_subject_means = subject_means.loc[subject_means["complete_seed_set"]].copy()
        pivot = complete_subject_means.pivot(
            index="test_subject",
            columns="objective",
            values=metric,
        )
        missing = [
            objective
            for objective in (PRIMARY_BASELINE, PRIMARY_METHOD)
            if objective not in pivot.columns
        ]
        if missing:
            tests.append(
                {
                    "metric": metric,
                    "n_subjects": 0,
                    f"{PRIMARY_BASELINE}_mean": np.nan,
                    f"{PRIMARY_METHOD}_mean": np.nan,
                    f"{PRIMARY_METHOD}_minus_{PRIMARY_BASELINE}": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "sign_flip_p": np.nan,
                    "cohens_dz": np.nan,
                    "status": f"missing objective(s): {','.join(missing)}",
                }
            )
            continue
        pivot = pivot.dropna(subset=[PRIMARY_BASELINE, PRIMARY_METHOD])
        if pivot.empty:
            tests.append(
                {
                    "metric": metric,
                    "n_subjects": 0,
                    f"{PRIMARY_BASELINE}_mean": np.nan,
                    f"{PRIMARY_METHOD}_mean": np.nan,
                    f"{PRIMARY_METHOD}_minus_{PRIMARY_BASELINE}": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "sign_flip_p": np.nan,
                    "cohens_dz": np.nan,
                    "status": "no complete paired seed sets available",
                }
            )
            continue
        differences = (pivot[PRIMARY_METHOD] - pivot[PRIMARY_BASELINE]).to_numpy()
        ci_low, ci_high = bootstrap_mean_ci(differences)
        pvalue = exact_sign_flip_pvalue(differences)
        if metric in SECONDARY_METRICS:
            raw_secondary[metric] = pvalue
        standard_deviation = (
            float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
        )
        tests.append(
            {
                "metric": metric,
                "n_subjects": len(differences),
                f"{PRIMARY_BASELINE}_mean": float(pivot[PRIMARY_BASELINE].mean()),
                f"{PRIMARY_METHOD}_mean": float(pivot[PRIMARY_METHOD].mean()),
                f"{PRIMARY_METHOD}_minus_{PRIMARY_BASELINE}": float(differences.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sign_flip_p": pvalue,
                "cohens_dz": float(differences.mean() / standard_deviation)
                if standard_deviation > 0
                else 0.0,
                "status": "ok",
            }
        )
    adjusted = holm_adjust(raw_secondary)
    tests_frame = pd.DataFrame(tests)
    tests_frame["holm_p"] = tests_frame.apply(
        lambda row: adjusted.get(row["metric"], row["sign_flip_p"]),
        axis=1,
    )
    random_baselines = {
        "top1": 1.0 / 200.0,
        "top5": 5.0 / 200.0,
        "top10": 10.0 / 200.0,
        "mrr": float(sum(1.0 / rank for rank in range(1, 201)) / 200.0),
    }
    return subject_means, tests_frame, random_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_completed_metrics(args.allow_partial)
    subject_means, tests, random_baselines = build_analysis(metrics)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS_ROOT / "run_metrics.csv", index=False)
    subject_means.to_csv(ANALYSIS_ROOT / "subject_means.csv", index=False)
    tests.to_csv(ANALYSIS_ROOT / "paired_subject_tests.csv", index=False)
    (ANALYSIS_ROOT / "random_baselines.json").write_text(
        json.dumps(random_baselines, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    complete_mask = subject_means["complete_seed_set"].fillna(False)
    complete_subjects = subject_means.loc[complete_mask]
    complete_paired_subjects = []
    for subject, frame in complete_subjects.groupby("test_subject"):
        objectives = set(frame["objective"].tolist())
        if {PRIMARY_BASELINE, PRIMARY_METHOD}.issubset(objectives):
            complete_paired_subjects.append(subject)
    status = {
        "subjects_with_metrics": sorted(subject_means["test_subject"].unique().tolist()),
        "complete_subjects": sorted(complete_paired_subjects),
        "complete_paired_subject_count": len(complete_paired_subjects),
        "run_count": int(len(metrics)),
        "primary_metric": PRIMARY_METRIC,
        "analysis_complete": len(complete_paired_subjects) == len(SUBJECTS),
    }
    (ANALYSIS_ROOT / "analysis_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
