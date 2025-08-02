import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
import matplotlib.pyplot as plt
import os
import sys



# performs louvain clustering on all embeddings, finds the best resolution by NMI, and computes average biological scores
# === Configuration ===
input_file = "Ahern_covid_UK/zeroshot_outputs/Ahern_covid_UK_ZS_GF.h5ad"
base_name = os.path.basename(input_file).replace("_ZS_GF.h5ad", "")

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

embedding_key = "X_zeroshot"
ground_truth_col = "major_subset"
output_dir = f"{base_name}/zeroshot_outputs"
os.makedirs(output_dir, exist_ok=True)


# === Load Data ===
flush_print("Loading AnnData...")
adata = ad.read_h5ad(input_file)
embedding = adata.obsm[embedding_key]
labels = adata.obs[ground_truth_col].values

# === Preprocessing for Clustering ===
flush_print("Running Louvain clustering on embeddings...")
adata.obsm["X_pca"] = embedding  
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)

# Scan resolutions and compute NMI for each
best_nmi = -1
best_res = 0.0
best_labels = None

for res in np.linspace(0.1, 2.0, 20):
    sc.tl.louvain(adata, resolution=res, key_added=f"louvain_{res:.2f}")
    cluster_labels = adata.obs[f"louvain_{res:.2f}"].values
    nmi = nmi_score(labels, cluster_labels)
    if nmi > best_nmi:
        best_nmi = nmi
        best_res = res
        best_labels = cluster_labels

flush_print(f"Best Louvain resolution: {best_res:.2f} (NMI = {best_nmi:.3f})")
adata.obs["best_louvain"] = best_labels

# === Compute Scores ===
asw = silhouette_score(embedding, labels)
asw_norm = (asw +1 ) / 2
ari = adjusted_rand_score(labels, best_labels)
nmi = normalized_mutual_info_score(labels, best_labels)
avg_bio = (asw_norm + ari + nmi) / 3

flush_print(f"\nEvaluation Metrics:")
flush_print(f"ASW  : {asw:.3f}" + f" (normalized: {asw_norm:.3f})")
flush_print(f"ARI  : {ari:.3f}")
flush_print(f"NMI  : {nmi:.3f}")
flush_print(f"AvgBIO: {avg_bio:.3f}")

# === Save Results ===
with open(os.path.join(output_dir, f"{base_name}_avgbio_all.txt"), "w") as f:
    f.write(f"ASW: {asw_norm:.4f}\n")
    f.write(f"ARI: {ari:.4f}\n")
    f.write(f"NMI: {nmi:.4f}\n")
    f.write(f"AvgBIO: {avg_bio:.4f}\n")
    f.write(f"Best Louvain Resolution: {best_res:.2f}\n")

flush_print("\nMetrics saved to embedding_evaluation_metrics.txt")
