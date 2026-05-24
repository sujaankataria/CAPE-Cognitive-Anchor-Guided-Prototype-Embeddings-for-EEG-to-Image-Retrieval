"""Visualization helpers for embeddings and retrieval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHORS = PROJECT_ROOT / "embeddings" / "anchors" / "uniform_category_anchors.npz"
DEFAULT_FIGURES = PROJECT_ROOT / "results" / "figures"
DEFAULT_TABLES = PROJECT_ROOT / "results" / "tables"
DEFAULT_SEED = 42


def plot_retrieval_grid(*args, **kwargs):
    raise NotImplementedError("plot_retrieval_grid is not implemented yet.")


def _resolve_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_anchor_cache(path: Path | str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing anchor cache: {resolved}")
    with np.load(resolved, allow_pickle=True) as cache:
        embeddings = np.asarray(cache["anchor_embeddings"], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D anchor embeddings, got {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(f"Anchor embeddings are not normalized; norm range {norms.min()} to {norms.max()}")
        metadata_raw = cache["metadata"].item() if cache["metadata"].shape == () else str(cache["metadata"])
        return {
            "path": resolved,
            "embeddings": embeddings,
            "concept": cache["concept"].copy(),
            "concept_id": cache["concept_id"].copy(),
            "concept_name": cache["concept_name"].copy(),
            "metadata": json.loads(metadata_raw),
        }


def compute_tsne(embeddings: np.ndarray, *, seed: int = DEFAULT_SEED) -> np.ndarray:
    perplexity = min(30.0, max(5.0, (embeddings.shape[0] - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def compute_umap_style_projection(embeddings: np.ndarray, *, seed: int = DEFAULT_SEED) -> np.ndarray:
    reducer = SpectralEmbedding(
        n_components=2,
        n_neighbors=min(30, embeddings.shape[0] - 1),
        affinity="nearest_neighbors",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def save_projection_table(path: Path, projection: np.ndarray, anchors: dict[str, Any], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "method": method,
            "x": projection[:, 0],
            "y": projection[:, 1],
            "concept": anchors["concept"],
            "concept_id": anchors["concept_id"],
            "concept_name": anchors["concept_name"],
        }
    )
    table.to_csv(path, index=False)


def plot_projection(path: Path, projection: np.ndarray, anchors: dict[str, Any], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.scatter(projection[:, 0], projection[:, 1], s=7, alpha=0.72, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.2, linewidth=0.5)

    label_indices = np.linspace(0, projection.shape[0] - 1, num=min(12, projection.shape[0]), dtype=int)
    for index in label_indices:
        ax.annotate(
            str(anchors["concept_name"][index]),
            (projection[index, 0], projection[index, 1]),
            fontsize=6,
            alpha=0.8,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_anchor_neighbor_table(anchors: dict[str, Any], *, top_k: int = 10) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    embeddings = anchors["embeddings"]
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, -np.inf)
    order = np.argsort(-similarities, axis=1)[:, :top_k]

    rows = []
    for source_index, neighbor_indices in enumerate(order):
        for rank, neighbor_index in enumerate(neighbor_indices, start=1):
            rows.append(
                {
                    "source_concept": str(anchors["concept"][source_index]),
                    "source_concept_id": str(anchors["concept_id"][source_index]),
                    "source_concept_name": str(anchors["concept_name"][source_index]),
                    "neighbor_rank": rank,
                    "neighbor_concept": str(anchors["concept"][neighbor_index]),
                    "neighbor_concept_id": str(anchors["concept_id"][neighbor_index]),
                    "neighbor_concept_name": str(anchors["concept_name"][neighbor_index]),
                    "cosine_similarity": float(similarities[source_index, neighbor_index]),
                }
            )
    return pd.DataFrame(rows)


def run_anchor_visualizations(
    *,
    anchors_path: Path,
    figures_dir: Path,
    tables_dir: Path,
    seed: int = DEFAULT_SEED,
    top_k: int = 10,
) -> dict[str, Any]:
    anchors = load_anchor_cache(anchors_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tsne = compute_tsne(anchors["embeddings"], seed=seed)
    umap_style = compute_umap_style_projection(anchors["embeddings"], seed=seed)

    tsne_png = figures_dir / "anchor_tsne.png"
    umap_png = figures_dir / "anchor_umap_style.png"
    tsne_csv = tables_dir / "anchor_tsne_coordinates.csv"
    umap_csv = tables_dir / "anchor_umap_style_coordinates.csv"
    neighbors_csv = tables_dir / "anchor_neighborhoods.csv"

    plot_projection(tsne_png, tsne, anchors, "Anchor t-SNE Projection")
    plot_projection(umap_png, umap_style, anchors, "Anchor UMAP-Style Projection")
    save_projection_table(tsne_csv, tsne, anchors, "tsne")
    save_projection_table(umap_csv, umap_style, anchors, "umap_style_spectral")
    build_anchor_neighbor_table(anchors, top_k=top_k).to_csv(neighbors_csv, index=False)

    return {
        "anchor_count": int(anchors["embeddings"].shape[0]),
        "embedding_dim": int(anchors["embeddings"].shape[1]),
        "anchor_source": str(anchors["path"].relative_to(PROJECT_ROOT)),
        "figures": [str(tsne_png.relative_to(PROJECT_ROOT)), str(umap_png.relative_to(PROJECT_ROOT))],
        "tables": [
            str(tsne_csv.relative_to(PROJECT_ROOT)),
            str(umap_csv.relative_to(PROJECT_ROOT)),
            str(neighbors_csv.relative_to(PROJECT_ROOT)),
        ],
        "top_k_neighbors": top_k,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize category anchor geometry.")
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_anchor_visualizations(
        anchors_path=_resolve_path(args.anchors),
        figures_dir=_resolve_path(args.figures_dir),
        tables_dir=_resolve_path(args.tables_dir),
        seed=args.seed,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
