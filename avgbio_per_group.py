import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os
import sys
import re
import pandas as pd

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# === Configuration ===
input_configs = {
    # "Ahern_covid_UK": {
    #     "file": "Ahern_covid_UK/scgpt_outputs/Ahern_covid_UK_scGPT.h5ad",
    #     "ground_truth_col": "major_subset"
    # },
    # "Green_BCL": {
    #     "file": "Green_BCL/scgpt_outputs/Green_BCL_scGPT.h5ad",
    #     "ground_truth_col": "author_cell_type"
    # },
    # "Kock_AIDAv2": {
    #     "file": "Kock_AIDAv2/scgpt_outputs/Kock_AIDAv2_scGPT.h5ad",
    #     "ground_truth_col": "Annotation_Level1"
    # },
    # "Perez_lupus": {
    #     "file": "Perez_lupus/scgpt_outputs/Perez_lupus_scGPT.h5ad",
    #     "ground_truth_col": "author_cell_type"
    # },
    "Tabula_immune": {
        "file": "Tabula_immune/scgpt_outputs/Tabula_immune_scGPT.h5ad",
        "ground_truth_col": "broad_cell_class"
    }
}


embedding_key = "X_scgpt"
groupby_cols = ["sex", "self_reported_ethnicity", "Age", "development_stage", "disease"]
min_cells_per_group = 100

# === Quartile-based Age Binning Function ===
def bin_age_column_quartiles(series: pd.Series, new_col_name="age_quartile"):
    series = series.astype(str)

    def extract_age(x):
        if "-" in x or ">=" in x:
            match = re.match(r"(\d+)", x)
        else:
            match = re.search(r"(\d+\.?\d*)", x)
        return float(match.group(1)) if match else np.nan

    numeric_ages = series.map(extract_age)
    q1, q2, q3 = np.nanpercentile(numeric_ages, [25, 50, 75])

    def assign_bin(age):
        if np.isnan(age): return "unknown"
        elif age < q1: return f"<{int(q1)}"
        elif age < q2: return f"{int(q1)}-{int(q2)-1}"
        elif age < q3: return f"{int(q2)}-{int(q3)-1}"
        else: return f">={int(q3)}"

    quartile_bins = numeric_ages.map(assign_bin)
    return pd.Series(quartile_bins, index=series.index, name=new_col_name)

# === Cohen's d ===
def cohens_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / (nx + ny - 2))
    d = (mean_x - mean_y) / pooled_std if pooled_std > 0 else 0.0
    return d

# === Main Loop Over Datasets ===
for dataset_name, config in input_configs.items():
    input_file = config["file"]
    ground_truth_col = config["ground_truth_col"]
    base_name = os.path.basename(input_file).replace("_scGPT.h5ad", "")
    output_dir = f"{base_name}/scgpt_outputs/avgbio_scores"
    os.makedirs(output_dir, exist_ok=True)

    flush_print(f"\n=== Processing: {dataset_name} ===")
    flush_print("Loading AnnData...")
    adata = ad.read_h5ad(input_file)
    embedding = adata.obsm[embedding_key]
    labels = adata.obs[ground_truth_col].values

    flush_print("Running Louvain clustering on embeddings...")
    adata.obsm["X_pca"] = embedding
    n_neighbors = min(15, embedding.shape[0] - 1)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors)

    best_nmi = -1
    best_res = 0.0
    best_labels = None
    for res in np.linspace(0.1, 2.0, 20):
        sc.tl.louvain(adata, resolution=res, key_added=f"louvain_{res:.2f}")
        cluster_labels = adata.obs[f"louvain_{res:.2f}"].values
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        if nmi > best_nmi:
            best_nmi = nmi
            best_res = res
            best_labels = cluster_labels

    adata.obs["best_louvain"] = best_labels
    flush_print(f"Best Louvain resolution: {best_res:.2f} (global NMI = {best_nmi:.3f})")

    for groupby_col in groupby_cols:
        flush_print(f"\n=== Evaluating by '{groupby_col}' ===")

        try:
            if groupby_col not in adata.obs.columns:
                flush_print(f"'{groupby_col}' not found in .obs — skipping.")
                continue

            col_normalized = groupby_col.strip().lower()
            if col_normalized in ["age", "development_age", "development stage", "development_stage"]:
                flush_print(f"Detected age-like column: '{groupby_col}' — applying quartile binning.")
                binned = bin_age_column_quartiles(adata.obs[groupby_col], new_col_name=f"{groupby_col}_quartile")
                adata.obs[f"{groupby_col}_quartile"] = binned
                groupby_col = f"{groupby_col}_quartile"

            groups_raw = adata.obs[groupby_col]
            valid_mask = (~groups_raw.isna()) & (groups_raw.astype(str).str.lower() != "unknown")
            groups = groups_raw[valid_mask].values
            labels_filtered = labels[valid_mask]
            embedding_filtered = embedding[valid_mask]
            best_labels_filtered = best_labels[valid_mask]

            group_scores = []

            for group in np.unique(groups):
                idx = np.where(groups == group)[0]
                if len(idx) < min_cells_per_group:
                    flush_print(f"Skipping group '{group}' (n={len(idx)})")
                    continue

                labels_group = labels_filtered[idx]
                if len(np.unique(labels_group)) < 2:
                    flush_print(f"Skipping group '{group}' due to label homogeneity.")
                    continue

                embedding_group = embedding_filtered[idx]
                best_labels_group = best_labels_filtered[idx]

                try:
                    asw = silhouette_score(embedding_group, labels_group)
                    asw_norm = (asw + 1) / 2
                except Exception as e:
                    flush_print(f"ASW failed for group '{group}': {str(e)}")
                    asw_norm = np.nan

                ari = adjusted_rand_score(labels_group, best_labels_group)
                nmi = normalized_mutual_info_score(labels_group, best_labels_group)
                avg_bio = np.nanmean([asw_norm, ari, nmi])

                flush_print(f"\nGroup: {group}")
                flush_print(f"ASW  : {asw_norm:.3f}")
                flush_print(f"ARI  : {ari:.3f}")
                flush_print(f"NMI  : {nmi:.3f}")
                flush_print(f"AvgBIO: {avg_bio:.3f}")

                group_scores.append({
                    "Group": group,
                    "n_cells": len(idx),
                    "ASW": asw_norm,
                    "ARI": ari,
                    "NMI": nmi,
                    "AvgBIO": avg_bio
                })

            output_path = os.path.join(output_dir, f"{base_name}_avgbio_gf_per_{groupby_col}.tsv")
            with open(output_path, "w") as f:
                f.write("Group\tn_cells\tASW\tARI\tNMI\tAvgBIO\n")
                for score in group_scores:
                    f.write(f"{score['Group']}\t{score['n_cells']}\t{score['ASW']:.4f}\t{score['ARI']:.4f}\t{score['NMI']:.4f}\t{score['AvgBIO']:.4f}\n")

                f.write("\n# Statistical Tests\n")
                try:
                    group_values = {s["Group"]: [s["AvgBIO"]] for s in group_scores if not np.isnan(s["AvgBIO"])}
                    if len(group_values) >= 2:
                        stat, pval = kruskal(*group_values.values())
                        f.write(f"Kruskal-Wallis_H\t{stat:.4f}\n")
                        f.write(f"Kruskal-Wallis_p\t{pval:.4e}\n")
                        flush_print(f"Kruskal–Wallis for {groupby_col}: H = {stat:.4f}, p = {pval:.4e}")

                        flush_print(f"Running pairwise Mann–Whitney U tests for '{groupby_col}'...")
                        groups = list(group_values.keys())
                        results = []
                        for i in range(len(groups)):
                            for j in range(i + 1, len(groups)):
                                g1, g2 = groups[i], groups[j]
                                try:
                                    dist1 = group_values[g1]
                                    dist2 = group_values[g2]
                                    stat, mw_pval = mannwhitneyu(dist1, dist2, alternative="two-sided")
                                    d_val = cohens_d(dist1, dist2)
                                    results.append({
                                        "group": f"{g1} vs {g2}",
                                        "U_statistic": stat,
                                        "p_uncorrected": mw_pval,
                                        "cohens_d": d_val
                                    })
                                except Exception as e:
                                    flush_print(f"Failed MW test {g1} vs {g2}: {e}")

                        if results:
                            pvals = [r["p_uncorrected"] for r in results]
                            _, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
                            for r, p_corr in zip(results, pvals_corr):
                                r["p_fdr"] = p_corr

                            f.write("\n# Mann-Whitney U + Cohen's d (FDR corrected)\n")
                            f.write("Group_Pair\tU_statistic\tp_uncorrected\tp_fdr\tcohens_d\n")
                            for r in results:
                                f.write(f"{r['group']}\t{r['U_statistic']:.4f}\t{r['p_uncorrected']:.4e}\t{r['p_fdr']:.4e}\t{r['cohens_d']:.4f}\n")

                except Exception as e:
                    flush_print(f"⚠️  Statistical tests failed for {groupby_col}: {e}")
                    f.write(f"Statistical Test Error: {e}\n")

            flush_print(f"Saved metrics to: {output_path}")

        except Exception as e:
            flush_print(f"⚠️  Error while processing '{groupby_col}': {str(e)} — skipping.\n")
