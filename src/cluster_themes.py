"""Cluster extracted question type keyphrases using BERTopic.

This script clusters keyphrases from question type extraction into ~N clusters per category.

Usage:
    python src/cluster_themes.py --input-csv output/extracted_types.csv --output-json output/clusters.json
    python src/cluster_themes.py --input-csv output/extracted_types.csv --output-json output/clusters.json --n-clusters 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN


# Categories to cluster (from QuestionTypeExtraction model)
THEME_CATEGORIES = [
    "question_types",
]


def parse_json_column(df: pd.DataFrame, column: str) -> List[str]:
    """Extract unique keyphrases from a JSON list column.

    Args:
        df: DataFrame with theme extraction results
        column: Column name containing JSON list strings

    Returns:
        List of unique keyphrases
    """
    all_keyphrases = []

    for value in df[column].dropna():
        try:
            keyphrases = json.loads(value)
            if isinstance(keyphrases, list):
                all_keyphrases.extend(keyphrases)
        except (json.JSONDecodeError, TypeError):
            continue

    # Deduplicate while preserving order, normalize to lowercase
    seen = set()
    unique_keyphrases = []
    for kp in all_keyphrases:
        kp_lower = kp.strip().lower()
        if kp_lower and kp_lower not in seen:
            seen.add(kp_lower)
            unique_keyphrases.append(kp_lower)

    return unique_keyphrases


def generate_embeddings(
    keyphrases: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Generate embeddings for keyphrases.

    Args:
        keyphrases: List of keyphrase strings
        model_name: Name of sentence transformer model

    Returns:
        Numpy array of embeddings
    """
    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"  Generating embeddings for {len(keyphrases)} keyphrases...")
    embeddings = model.encode(keyphrases, show_progress_bar=True)

    return embeddings


def cluster_keyphrases(
    keyphrases: List[str],
    embeddings: np.ndarray,
    n_clusters: int = 5,
    min_cluster_size: int = 2
) -> Dict[int, List[str]]:
    """Cluster keyphrases using BERTopic.

    Args:
        keyphrases: List of keyphrase strings
        embeddings: Precomputed embeddings
        n_clusters: Target number of clusters
        min_cluster_size: Minimum cluster size for HDBSCAN

    Returns:
        Dict mapping cluster_id to list of keyphrases
    """
    print(f"  Running BERTopic with target {n_clusters} clusters...")

    # Handle small datasets
    if len(keyphrases) < 5:
        print(f"  Too few keyphrases ({len(keyphrases)}), returning single cluster")
        return {0: keyphrases}

    # Configure UMAP for dimensionality reduction
    n_neighbors = min(15, len(keyphrases) - 1)
    n_components = min(5, len(keyphrases) - 2)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Configure HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        prediction_data=True
    )

    # Create BERTopic model with target number of topics
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_clusters,  # Reduce to target number
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False
    )

    # Fit model
    topics, _ = topic_model.fit_transform(keyphrases, embeddings)

    # Group keyphrases by topic
    clusters = {}
    for keyphrase, topic_id in zip(keyphrases, topics):
        # Map outliers (-1) to cluster 0
        cluster_id = topic_id if topic_id >= 0 else 0
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(keyphrase)

    print(f"  Created {len(clusters)} clusters")
    for cluster_id, members in sorted(clusters.items()):
        print(f"    Cluster {cluster_id}: {len(members)} keyphrases")

    return clusters


def cluster_all_themes(
    input_csv_path: str,
    output_json_path: str,
    n_clusters: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Dict:
    """Cluster keyphrases for all theme categories.

    Args:
        input_csv_path: Path to theme extraction CSV
        output_json_path: Path to output JSON file
        n_clusters: Target number of clusters per category
        embedding_model: Sentence transformer model name

    Returns:
        Dict with clustered keyphrases per category
    """
    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV not found: {input_csv_path}"

    # Create output directory
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=== Loading theme extraction results ===")
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} rows")

    # Process each category
    results = {}

    for category in THEME_CATEGORIES:
        print(f"\n=== Processing {category} ===")

        if category not in df.columns:
            print(f"  Column '{category}' not found, skipping")
            results[category] = []
            continue

        # Extract unique keyphrases
        keyphrases = parse_json_column(df, category)
        print(f"  Found {len(keyphrases)} unique keyphrases")

        if len(keyphrases) == 0:
            print(f"  No keyphrases found, skipping")
            results[category] = []
            continue

        # Generate embeddings
        embeddings = generate_embeddings(keyphrases, embedding_model)

        # Cluster
        clusters = cluster_keyphrases(keyphrases, embeddings, n_clusters)

        # Format output: list of {"keyphrases": [...]}
        results[category] = [
            {"keyphrases": sorted(members)}
            for cluster_id, members in sorted(clusters.items())
        ]

    # Save results
    print(f"\n=== Saving results to {output_json_path} ===")
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n=== Summary ===")
    for category, clusters in results.items():
        total_keyphrases = sum(len(c["keyphrases"]) for c in clusters)
        print(f"{category}: {len(clusters)} clusters, {total_keyphrases} keyphrases")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cluster theme keyphrases using BERTopic"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to theme extraction CSV file"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON file with clustered keyphrases"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Target number of clusters per category (default: 5)"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings"
    )

    args = parser.parse_args()

    cluster_all_themes(
        input_csv_path=args.input_csv,
        output_json_path=args.output_json,
        n_clusters=args.n_clusters,
        embedding_model=args.embedding_model
    )


if __name__ == "__main__":
    main()
