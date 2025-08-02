import os
import sys
import re
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import scipy.sparse

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# === Config for Ahern_covid_UK ===
dataset = "Ahern_covid_UK"
hvg_file = "/iridisfs/ddnb/faircells/data/data_hvg/Ahern_covid_UK_hvg.h5ad"
embedding_key = "X_umap"
groupby_cols = ["sex", "self_reported_ethnicity", "disease", "development_stage", "Age"]
n_pairs = 300000

# === Quartile-based Age Binning Function ===
def bin_age_column_quartiles(series: pd.Series, new_col_name="age_quartile"):
    series = series.astype(str)
    def extract_age(x):
        if "-" in x or ">=" in x:
            match = re.match(r"(\d+)", x)
        else:
            match = re.search(r"(\d+\.?\d*)", x)
        return float(match.group(1)) if match else np.nan
    numeric_ages = series.map(extract_age).dropna()
    if numeric_ages.nunique() < 4:
        flush_print(f"Not enough unique numeric values in '{series.name}' to compute quartiles — skipping binning.")
        return pd.Series(["unknown"] * len(series), index=series.index, name=new_col_name)
    q1, q2, q3 = np.nanpercentile(numeric_ages, [25, 50, 75])
    def assign_bin(age):
        if np.isnan(age): return "unknown"
        elif age < q1: return f"<{int(q1)}"
        elif age < q2: return f"{int(q1)}-{int(q2)-1}"
        elif age < q3: return f"{int(q2)}-{int(q3)-1}"
        else: return f">={int(q3)}"
    return series.map(extract_age).map(assign_bin)

flush_print(f"\n=== Processing dataset: {dataset} ===")
flush_print("Loading HVG .h5ad file...")
adata_hvg = sc.read_h5ad(hvg_file)
coords = adata_hvg.obsm[embedding_key]
adata = adata_hvg
output_dir = os.path.join(dataset, "raw_outputs", "umap_distance_comparison", embedding_key)
os.makedirs(output_dir, exist_ok=True)

for groupby_col in groupby_cols:
    flush_print(f"\nProcessing groupby column: {groupby_col}")
    if groupby_col not in adata.obs.columns:
        flush_print(f"Skipping '{groupby_col}' — not found in .obs")
        continue
    col_normalized = groupby_col.strip().lower()
    if col_normalized in ["age", "development_age", "development stage", "development_stage"]:
        flush_print(f"Binning '{groupby_col}' into quartiles...")
        binned = bin_age_column_quartiles(adata.obs[groupby_col], new_col_name=f"{groupby_col}_quartile")
        adata.obs[f"{groupby_col}_quartile"] = binned
        groupby_col = f"{groupby_col}_quartile"
    group_stats = []
    all_group_dists = {}
    plt.figure(figsize=(10, 6))
    plotted_any = False
    for group in adata.obs[groupby_col].unique():
        if pd.isna(group) or str(group).lower() == "unknown":
            continue
        flush_print(f"Sampling distances for group: {group}")
        subset_idx = adata.obs[groupby_col] == group
        sub_coords = coords[subset_idx.values]
        if sub_coords.shape[0] < 10:
            flush_print(f"Skipping group {group} (too few cells)")
            continue
        rng = np.random.default_rng(seed=42)
        max_pairs = min(n_pairs, sub_coords.shape[0] ** 2)
        i_indices = rng.integers(0, sub_coords.shape[0], size=max_pairs)
        j_indices = rng.integers(0, sub_coords.shape[0], size=max_pairs)
        mask = i_indices != j_indices
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        distances = np.linalg.norm(sub_coords[i_indices] - sub_coords[j_indices], axis=1)
        if len(distances) == 0:
            continue
        all_group_dists[str(group)] = distances
        plotted_any = True
        group_stats.append({
            "group": group,
            "n_cells": sub_coords.shape[0],
            "mean_distance": float(f"{np.mean(distances):.3g}"),
            "median_distance": float(f"{np.median(distances):.3g}"),
            "std_distance": float(f"{np.std(distances):.3g}"),
            "min_distance": float(f"{np.min(distances):.3g}"),
            "max_distance": float(f"{np.max(distances):.3g}"),
            "95th_percentile": float(f"{np.percentile(distances, 95):.3g}")
        })
        sns.histplot(distances, bins=50, kde=True, stat="density", label=f"{group} (n={sub_coords.shape[0]})", alpha=0.4)
    if plotted_any:
        title = f"HVG UMAP Embedding Distance by {groupby_col}"
        plt.title(title)
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{dataset}_{groupby_col}_{embedding_key}_distance_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        flush_print(f"Saved plot: {plot_path}")
    else:
        plt.close()
        flush_print(f"No valid groups to plot for '{groupby_col}' — skipping plot.")
    summary_df = pd.DataFrame(group_stats)
    output_rows = [summary_df]
    if len(all_group_dists) >= 2:
        try:
            stat, pval = kruskal(*all_group_dists.values())
            flush_print(f"Kruskal–Wallis for '{groupby_col}': H = {stat:.3g}, p = {pval:.3g}")
            kruskal_row = pd.DataFrame([{
                "group": "Kruskal–Wallis",
                "n_cells": "",
                "mean_distance": "",
                "median_distance": "",
                "std_distance": "",
                "min_distance": "",
                "max_distance": "",
                "95th_percentile": "",
                "H_statistic": float(f"{stat:.3g}"),
                "p_value": float(f"{pval:.3g}")
            }])
            output_rows.append(pd.DataFrame([{}]))
            output_rows.append(kruskal_row)
        except Exception as e:
            flush_print(f"Kruskal–Wallis failed for '{groupby_col}': {e}")
    if len(all_group_dists) >= 2:
        flush_print(f"Running pairwise Mann–Whitney U tests and Cohen's d for '{groupby_col}'...")
        groups = list(all_group_dists.keys())
        results = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                d1, d2 = all_group_dists[g1], all_group_dists[g2]
                try:
                    stat, mw_pval = mannwhitneyu(d1, d2, alternative="two-sided")
                    n1, n2 = len(d1), len(d2)
                    pooled_sd = np.sqrt(((n1 - 1)*np.var(d1) + (n2 - 1)*np.var(d2)) / (n1 + n2 - 2))
                    cohens_d = (np.mean(d1) - np.mean(d2)) / pooled_sd if pooled_sd > 0 else 0
                    results.append({
                        "group": f"{g1} vs {g2}",
                        "U_statistic": stat,
                        "p_uncorrected": mw_pval,
                        "cohens_d": cohens_d
                    })
                except Exception as e:
                    flush_print(f"Failed MW test {g1} vs {g2}: {e}")
        pvals = [r["p_uncorrected"] for r in results]
        _, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
        for r, p_corr in zip(results, pvals_corr):
            r["p_fdr_corrected"] = p_corr
        mw_df = pd.DataFrame(results)
        output_rows.append(pd.DataFrame([{}]))
        output_rows.append(mw_df)
    full_output = pd.concat(output_rows, ignore_index=True)
    csv_path = os.path.join(output_dir, f"{dataset}_{groupby_col}_{embedding_key}_distance_summary.csv")
    full_output.to_csv(csv_path, index=False)
    flush_print(f"Saved combined summary CSV: {csv_path}")
flush_print("\nAll done.") 