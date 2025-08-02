import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

datasets = [
    "Ahern_covid_UK",
    "Green_BCL",
    "Kock_AIDAv2",
    "Perez_lupus",
    "Tabula_immune",
]

embedding_key = "X_umap"
baseline_dir = "baseline/umap_distance_comparison/X_umap"
os.makedirs(baseline_dir, exist_ok=True)
n_pairs = 300000

file_paths = {
    "Ahern_covid_UK": "/iridisfs/ddnb/faircells/data/data_hvg/Ahern_covid_UK_hvg.h5ad",
    "Green_BCL": "/iridisfs/ddnb/faircells/data/data_hvg/Green_BCL_hvg.h5ad",
    "Kock_AIDAv2": "/iridisfs/ddnb/faircells/data/data_hvg/Kock_AIDAv2_hvg.h5ad",
    "Perez_lupus": "/iridisfs/ddnb/faircells/data/data_hvg/Perez_lupus_hvg.h5ad",
    "Tabula_immune": "/iridisfs/ddnb/faircells/data/data_hvg/Tabula_immune_hvg.h5ad",
}

for dataset in datasets:
    flush_print(f"Processing {dataset}...")
    adata = sc.read_h5ad(file_paths[dataset])

    if embedding_key not in adata.obsm:
        flush_print(f"Embedding key '{embedding_key}' not found in {dataset} — skipping.")
        continue

    coords = adata.obsm[embedding_key]
    n_cells = coords.shape[0]
    if n_cells < 10:
        flush_print(f"Not enough cells in {dataset} — skipping.")
        continue

    rng = np.random.default_rng(seed=42)
    max_pairs = min(n_pairs, n_cells ** 2)
    i_idx = rng.integers(0, n_cells, size=max_pairs)
    j_idx = rng.integers(0, n_cells, size=max_pairs)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    distances = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)

    stats = {
        'dataset': dataset,
        'embedding': embedding_key,
        'n_cells': n_cells,
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        '95th_percentile': float(np.percentile(distances, 95))
    }

    df = pd.DataFrame([stats])
    csv_path = os.path.join(baseline_dir, f"{dataset}_{embedding_key}_summary.csv")
    df.to_csv(csv_path, index=False)
    flush_print(f"Saved: {csv_path}")

    plt.figure(figsize=(8, 6))
    sns.histplot(distances, bins=100, kde=True, stat="density", color="steelblue")
    plt.title(f"{dataset} - {embedding_key} Distance Distribution")
    plt.xlabel("Pairwise Euclidean Distance")
    plt.ylabel("Density")
    plt.tight_layout()
    plot_path = os.path.join(baseline_dir, f"{dataset}_{embedding_key}_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    flush_print(f"Saved plot: {plot_path}")

flush_print("Done.")