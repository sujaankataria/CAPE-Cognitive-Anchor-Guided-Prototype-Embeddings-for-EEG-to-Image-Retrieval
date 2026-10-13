"""Semantic-locality analysis for clean publication retrieval checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EEGCLIPModel
from training.datasets import EEGCLIPPairDataset
from training.validate import collect_eeg_embeddings
from utils.metrics import retrieval_metrics
from utils.retrieval import CLIPImageRetrievalDatabase, rank_images


SPLITS_ROOT = PROJECT_ROOT / "data" / "things_eeg2" / "splits"
PUBLICATION_ROOT = PROJECT_ROOT / "results" / "publication"
MANIFEST_PATH = PUBLICATION_ROOT / "experiment_manifest.csv"
METRICS_ROOT = PUBLICATION_ROOT / "metrics"
OUTPUT_ROOT = PUBLICATION_ROOT / "semantic_locality"
OLD_SEMANTIC_LOCALITY = PROJECT_ROOT / "results" / "semantic_locality.csv"

GROUPS = (
    "food",
    "animal",
    "vehicle",
    "tool",
    "household object",
    "natural object",
    "other",
)


def split_csv_path(split: str) -> Path:
    return SPLITS_ROOT / f"{split}_split.csv"


def load_existing_concept_groups() -> dict[str, str]:
    """Reuse the broad-group taxonomy from the earlier semantic-locality file."""

    if not OLD_SEMANTIC_LOCALITY.is_file():
        return {}
    columns = [
        "ground_truth_concept",
        "ground_truth_group",
        "top1_concept",
        "top1_group",
    ]
    frame = pd.read_csv(OLD_SEMANTIC_LOCALITY, usecols=columns)
    mapping: dict[str, str] = {}
    for concept, group in zip(frame["ground_truth_concept"], frame["ground_truth_group"], strict=False):
        if isinstance(concept, str) and isinstance(group, str):
            mapping.setdefault(concept, group)
    for concept, group in zip(frame["top1_concept"], frame["top1_group"], strict=False):
        if isinstance(concept, str) and isinstance(group, str):
            mapping.setdefault(concept, group)
    return mapping


def fallback_group(concept_name: str) -> str:
    """Deterministic broad grouping for concepts absent from the legacy map."""

    name = concept_name.lower()
    food_terms = {
        "banana",
        "basil",
        "batter",
        "bread",
        "bun",
        "cake",
        "calamari",
        "cashew",
        "cheese",
        "coconut",
        "coffee",
        "cookie",
        "dessert",
        "egg",
        "fruit",
        "garlic",
        "hamburger",
        "lamb",
        "lettuce",
        "meat",
        "mussel",
        "okra",
        "omelet",
        "onion",
        "orange",
        "pear",
        "pepper",
        "pie",
        "popcorn",
        "popsicle",
        "pretzel",
        "radish",
        "raspberry",
        "sausage",
        "scallion",
        "scallop",
        "seaweed",
        "strawberry",
        "turkey",
        "walnut",
        "wheat",
        "wine",
    }
    animal_terms = {
        "antelope",
        "beaver",
        "bug",
        "cat",
        "caterpillar",
        "cheetah",
        "cobra",
        "crab",
        "crow",
        "dalmatian",
        "dragonfly",
        "eagle",
        "eel",
        "elephant",
        "flamingo",
        "goose",
        "gopher",
        "gorilla",
        "grasshopper",
        "hummingbird",
        "manatee",
        "mosquito",
        "ostrich",
        "panther",
        "pheasant",
        "piglet",
        "possum",
        "pug",
        "rhinoceros",
        "rooster",
        "seagull",
        "tick",
    }
    vehicle_terms = {
        "aircraft",
        "bike",
        "boat",
        "buggy",
        "cart",
        "ferry",
        "gondola",
        "jeep",
        "minivan",
        "sailboat",
        "scooter",
        "sled",
        "station",
        "submarine",
        "unicycle",
        "wheelchair",
    }
    tool_terms = {
        "backscratcher",
        "bat",
        "blowtorch",
        "bottle_opener",
        "brace",
        "bullet",
        "chain",
        "chopsticks",
        "cleaver",
        "dagger",
        "fork",
        "grenade",
        "hammer",
        "handbrake",
        "ladle",
        "metal_detector",
        "pickax",
        "pocketknife",
        "sandpaper",
        "slingshot",
        "spatula",
        "spoon",
        "stethoscope",
        "tongs",
        "tool",
        "vise",
        "wok",
    }
    household_terms = {
        "bench",
        "breadbox",
        "candlestick",
        "cd_player",
        "chest",
        "chime",
        "coffeemaker",
        "crib",
        "drum",
        "freezer",
        "highchair",
        "jukebox",
        "kettle",
        "lampshade",
        "laundry",
        "modem",
        "music_box",
        "nightstand",
        "rug",
        "table",
        "tape_recorder",
        "television",
    }
    natural_terms = {
        "bush",
        "ice",
        "marijuana",
        "orchid",
        "seed",
    }
    for term in food_terms:
        if term in name:
            return "food"
    for term in animal_terms:
        if term in name:
            return "animal"
    for term in vehicle_terms:
        if term in name:
            return "vehicle"
    for term in tool_terms:
        if term in name:
            return "tool"
    for term in household_terms:
        if term in name:
            return "household object"
    for term in natural_terms:
        if term in name:
            return "natural object"
    return "other"


def concept_group_array(concept_names: np.ndarray, existing: dict[str, str]) -> np.ndarray:
    groups = []
    for raw_name in concept_names:
        name = str(raw_name)
        groups.append(existing.get(name, fallback_group(name)))
    return np.asarray(groups, dtype=object)


def reciprocal_rank_for_match(mask: np.ndarray) -> np.ndarray:
    reciprocal = np.zeros(mask.shape[0], dtype=np.float64)
    for row_index, row in enumerate(mask):
        matches = np.flatnonzero(row)
        if matches.size:
            reciprocal[row_index] = 1.0 / float(matches[0] + 1)
    return reciprocal


def semantic_metrics(
    ranked_indices: np.ndarray,
    scores: np.ndarray,
    target_indices: np.ndarray,
    database: CLIPImageRetrievalDatabase,
    database_groups: np.ndarray,
) -> dict[str, float]:
    target_groups = database_groups[target_indices]
    ranked_groups = database_groups[ranked_indices]
    group_hits = ranked_groups == target_groups[:, None]

    target_embeddings = database.embeddings[target_indices]
    ranked_embeddings = database.embeddings[ranked_indices]
    target_to_ranked = np.einsum("qd,qkd->qk", target_embeddings, ranked_embeddings)

    metrics = {
        "semantic_group_top1": float(group_hits[:, :1].any(axis=1).mean()),
        "semantic_group_top5": float(group_hits[:, :5].any(axis=1).mean()),
        "semantic_group_top10": float(group_hits[:, :10].any(axis=1).mean()),
        "semantic_group_mrr": float(reciprocal_rank_for_match(group_hits).mean()),
        "target_top1_clip_similarity": float(target_to_ranked[:, 0].mean()),
        "target_top5_clip_similarity_mean": float(target_to_ranked[:, :5].mean(axis=1).mean()),
        "target_top5_clip_similarity_max": float(target_to_ranked[:, :5].max(axis=1).mean()),
        "target_top10_clip_similarity_mean": float(target_to_ranked[:, :10].mean(axis=1).mean()),
        "retrieval_top1_score": float(scores[:, 0].mean()),
    }
    for group in GROUPS:
        group_mask = target_groups == group
        metrics[f"group_{group.replace(' ', '_')}_query_count"] = int(group_mask.sum())
        if group_mask.any():
            metrics[f"group_{group.replace(' ', '_')}_semantic_top1"] = float(
                group_hits[group_mask, :1].any(axis=1).mean()
            )
            metrics[f"group_{group.replace(' ', '_')}_semantic_top5"] = float(
                group_hits[group_mask, :5].any(axis=1).mean()
            )
        else:
            metrics[f"group_{group.replace(' ', '_')}_semantic_top1"] = math.nan
            metrics[f"group_{group.replace(' ', '_')}_semantic_top5"] = math.nan
    return metrics


def load_model(checkpoint_path: Path, device: torch.device) -> EEGCLIPModel:
    model = EEGCLIPModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_run(
    run: pd.Series,
    *,
    batch_size: int,
    device: torch.device,
    existing_groups: dict[str, str],
    limit_queries: int | None,
) -> dict[str, Any]:
    checkpoint_path = PROJECT_ROOT / str(run["checkpoint"])
    dataset = EEGCLIPPairDataset(
        split_csv_path("test"),
        split="test",
        subject=str(run["test_subject"]),
        limit=limit_queries,
        anchors_path=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = load_model(checkpoint_path, device)
    eeg_embeddings, target_indices, target_concepts, cosine_loss = collect_eeg_embeddings(model, loader, device)
    database = CLIPImageRetrievalDatabase.from_npz(dataset.image_cache["path"])
    database_groups = concept_group_array(database.concept_name, existing_groups)
    ranked_indices, scores = rank_images(eeg_embeddings, database.embeddings, top_k=database.embeddings.shape[0])

    exact = retrieval_metrics(
        ranked_indices,
        target_indices,
        target_concepts=target_concepts,
        database_concepts=database.concept,
    )
    semantic = semantic_metrics(ranked_indices, scores, target_indices, database, database_groups)

    return {
        "experiment_id": str(run["experiment_id"]),
        "objective": str(run["objective"]),
        "seed": int(run["seed"]),
        "test_subject": str(run["test_subject"]),
        "setting": str(run["setting"]),
        "num_queries": int(eeg_embeddings.shape[0]),
        "cosine_loss": float(cosine_loss),
        **exact,
        **semantic,
    }


def sign_flip_pvalue(differences: np.ndarray) -> float:
    diffs = np.asarray(differences, dtype=np.float64)
    observed = abs(float(diffs.mean()))
    signs = np.array(np.meshgrid(*[[-1.0, 1.0] for _ in diffs])).T.reshape(-1, len(diffs))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= observed - 1e-15).mean())


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        value = min(1.0, (total - rank) * pvalue)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def paired_tests(subject_means: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "semantic_group_top1",
        "semantic_group_top5",
        "semantic_group_top10",
        "semantic_group_mrr",
        "target_top1_clip_similarity",
        "target_top5_clip_similarity_max",
        "target_top10_clip_similarity_mean",
    ]
    rows = []
    raw_pvalues = {}
    for metric in metrics:
        pivot = subject_means.pivot(index="test_subject", columns="objective", values=metric).dropna()
        if not {"direct", "hybrid"}.issubset(pivot.columns):
            continue
        diffs = pivot["hybrid"].to_numpy(dtype=np.float64) - pivot["direct"].to_numpy(dtype=np.float64)
        pvalue = sign_flip_pvalue(diffs)
        raw_pvalues[metric] = pvalue
        rows.append(
            {
                "metric": metric,
                "n_subjects": int(len(diffs)),
                "direct_mean": float(pivot["direct"].mean()),
                "hybrid_mean": float(pivot["hybrid"].mean()),
                "hybrid_minus_direct": float(diffs.mean()),
                "sign_flip_p": pvalue,
            }
        )
    adjusted = holm_adjust(raw_pvalues)
    for row in rows:
        row["holm_p"] = adjusted[row["metric"]]
    return pd.DataFrame(rows)


def write_plots(subject_means: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mrr", "Exact MRR"),
        ("top1", "Exact Top-1"),
        ("semantic_group_top1", "Semantic-Group Top-1"),
        ("semantic_group_top5", "Semantic-Group Top-5"),
        ("semantic_group_mrr", "Semantic-Group MRR"),
        ("target_top1_clip_similarity", "Target-vs-Top1 CLIP Similarity"),
    ]
    for metric, title in metrics:
        pivot = subject_means.pivot(index="test_subject", columns="objective", values=metric).dropna()
        if not {"direct", "hybrid"}.issubset(pivot.columns):
            continue
        subjects = list(pivot.index)
        x = np.arange(len(subjects))
        direct = pivot["direct"].to_numpy(dtype=np.float64)
        hybrid = pivot["hybrid"].to_numpy(dtype=np.float64)
        delta = hybrid - direct

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot(x, direct, marker="o", label="direct")
        axes[0].plot(x, hybrid, marker="o", label="hybrid")
        axes[0].set_title(title)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(subjects, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        colors = ["#2f7d4f" if value >= 0 else "#a33d3d" for value in delta]
        axes[1].bar(x, delta, color=colors)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title(f"Hybrid - Direct ({delta.mean():+.6f} mean)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(subjects, rotation=45, ha="right")
        axes[1].grid(axis="y", alpha=0.25)

        safe_metric = metric.replace("/", "_")
        fig.savefig(figures_dir / f"{safe_metric}_subject_delta.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", default="direct,hybrid")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objectives = {part.strip() for part in args.objectives.split(",") if part.strip()}
    manifest = pd.read_csv(MANIFEST_PATH)
    metrics = pd.read_csv(PUBLICATION_ROOT / "analysis" / "run_metrics.csv")
    runs = manifest[
        (manifest["status"] == "completed")
        & (manifest["setting"] == "cross_subject")
        & (manifest["objective"].isin(objectives))
    ].merge(
        metrics[["experiment_id", "checkpoint"]],
        on="experiment_id",
        how="left",
    )
    runs = runs.sort_values(["test_subject", "objective", "seed"]).reset_index(drop=True)
    if args.max_runs is not None:
        runs = runs.head(args.max_runs)
    if runs.empty:
        raise RuntimeError("No completed publication runs matched the requested filters")
    missing = runs["checkpoint"].isna()
    if missing.any():
        raise RuntimeError(f"Missing checkpoint paths for: {runs.loc[missing, 'experiment_id'].tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing_groups = load_existing_concept_groups()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, run in runs.iterrows():
        print(f"Evaluating {run['experiment_id']} on {device}", flush=True)
        rows.append(
            evaluate_run(
                run,
                batch_size=args.batch_size,
                device=device,
                existing_groups=existing_groups,
                limit_queries=args.limit_queries,
            )
        )
    run_metrics = pd.DataFrame(rows)
    run_metrics.to_csv(output_dir / "run_semantic_metrics.csv", index=False)

    mean_columns = [
        "mrr",
        "top1",
        "top5",
        "top10",
        "semantic_group_top1",
        "semantic_group_top5",
        "semantic_group_top10",
        "semantic_group_mrr",
        "target_top1_clip_similarity",
        "target_top5_clip_similarity_mean",
        "target_top5_clip_similarity_max",
        "target_top10_clip_similarity_mean",
    ]
    subject_means = (
        run_metrics.groupby(["test_subject", "objective"], as_index=False)[mean_columns]
        .mean()
        .sort_values(["test_subject", "objective"])
    )
    subject_means["seed_count"] = (
        run_metrics.groupby(["test_subject", "objective"])["seed"].nunique().to_numpy()
    )
    subject_means.to_csv(output_dir / "subject_semantic_means.csv", index=False)

    tests = paired_tests(subject_means)
    tests.to_csv(output_dir / "paired_semantic_tests.csv", index=False)
    write_plots(subject_means, output_dir)

    summary = {
        "analysis_complete": bool(
            len(run_metrics) == 60
            and set(run_metrics["objective"]) == {"direct", "hybrid"}
            and run_metrics.groupby(["test_subject", "objective"])["seed"].nunique().eq(3).all()
        ),
        "device": str(device),
        "run_count": int(len(run_metrics)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
