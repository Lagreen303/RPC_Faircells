import os
import scanpy as sc
import anndata as ad
import numpy as np
import sys
import scipy.sparse

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

input_file = "/iridisfs/ddnb/faircells/data/Ahern_covid_UK.h5ad"
output_dir = "/iridisfs/ddnb/faircells/data/data_hvg/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "Ahern_covid_UK_hvg.h5ad")

flush_print(f"Loading data from {input_file} ...")
adata = ad.read_h5ad(input_file)
flush_print(f"Loaded {adata.shape[0]} cells, {adata.shape[1]} genes.")

# Optionally reset to raw counts if available
if hasattr(adata, 'raw') and adata.raw is not None:
    flush_print("Resetting AnnData to raw counts...")
    adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)
    flush_print(f"Reset to raw: {adata.shape[0]} cells, {adata.shape[1]} genes.")

# Remove cells with all-zero counts
if scipy.sparse.issparse(adata.X):
    cell_sum = adata.X.sum(axis=1).A.ravel()
else:
    cell_sum = adata.X.sum(axis=1)
adata = adata[np.array(cell_sum != 0).ravel()]
flush_print(f"After removing zero-count cells: {adata.shape[0]} cells, {adata.shape[1]} genes.")

# Remove genes with all-zero counts
if scipy.sparse.issparse(adata.X):
    gene_sum = adata.X.sum(axis=0).A.ravel()
else:
    gene_sum = adata.X.sum(axis=0)
adata = adata[:, np.array(gene_sum != 0).ravel()]
flush_print(f"After removing zero-count genes: {adata.shape[0]} cells, {adata.shape[1]} genes.")

# Remove genes with any NaN in metadata
adata._inplace_subset_var(~adata.var.isnull().any(axis=1))
flush_print(f"After dropping NaN in var: {adata.shape[0]} cells, {adata.shape[1]} genes.")

if adata.shape[0] == 0 or adata.shape[1] == 0:
    flush_print("No cells or genes left after filtering. Exiting.")
    sys.exit(1)

flush_print("Normalizing total counts...")
sc.pp.normalize_total(adata)
flush_print("Log1p transforming...")
sc.pp.log1p(adata)

# Remove genes with all NaN or all zero after log1p
if scipy.sparse.issparse(adata.X):
    gene_nan = np.isnan(adata.X.A).all(axis=0)
    gene_zero = (adata.X.A == 0).all(axis=0)
else:
    gene_nan = np.isnan(adata.X).all(axis=0)
    gene_zero = (adata.X == 0).all(axis=0)
remove_mask = np.logical_not(np.logical_or(gene_nan, gene_zero))
adata = adata[:, remove_mask]
flush_print(f"After removing all-NaN or all-zero genes post-log1p: {adata.shape[0]} cells, {adata.shape[1]} genes.")

flush_print("Selecting highly variable genes...")
if scipy.sparse.issparse(adata.X):
    means = np.array(adata.X.mean(axis=0)).ravel()
else:
    means = adata.X.mean(axis=0)
flush_print(f"Number of genes: {adata.shape[1]}")
flush_print(f"Means NaN count: {np.isnan(means).sum()}")
flush_print(f"Means min: {np.nanmin(means)}, max: {np.nanmax(means)}")

# Remove genes with NaN means
valid_means = np.logical_not(np.isnan(means))
adata = adata[:, valid_means]
means = means[valid_means]

# Check for enough unique means
unique_means_count = len(np.unique(means))
if adata.shape[1] < 10 or unique_means_count < 10:
    flush_print("Too few valid genes with unique means for HVG selection. Exiting.")
    sys.exit(1)

# Use 2000 HVGs
n_hvg = 2000
flush_print(f"Selecting {n_hvg} highly variable genes from {adata.shape[1]} available genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
flush_print(f"After HVG: {adata.shape[0]} cells, {adata.shape[1]} genes.")

# Remove all genes with NaN names
adata._inplace_subset_var(~adata.var_names.isnull())
flush_print(f"Removed genes with NaN names. Remaining genes: {adata.shape[1]}")
# Remove all duplicated gene names
adata._inplace_subset_var(~adata.var_names.duplicated())
flush_print(f"Removed duplicated gene names. Remaining genes: {adata.shape[1]}")

flush_print("Scaling...")
sc.pp.scale(adata, max_value=10)
flush_print("PCA...")
sc.pp.pca(adata)
flush_print("Neighbors...")
sc.pp.neighbors(adata)
flush_print("UMAP...")
sc.tl.umap(adata)
flush_print("Saving...")
adata.write(output_file)
flush_print(f"Saved: {output_file}") 