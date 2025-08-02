from helical.models.scgpt import scGPT, scGPTConfig
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import anndata as ad
import sys
import pandas as pd
import re
import gzip
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scanpy as sc
import torch

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# === Load scGPT model ===
flush_print("Loading scGPT configuration...")
device = "cuda" if torch.cuda.is_available() else "cpu"
scgpt_config = scGPTConfig(batch_size=10, device=device)
scgpt = scGPT(configurer=scgpt_config)

# === Load data ===
flush_print("Loading data...")
input_file = "/iridisfs/ddnb/faircells/data/Perez_lupus.h5ad"
ann_data = ad.read_h5ad(input_file)
base_name = os.path.basename(input_file).replace(".h5ad", "")
output_dir = f"{base_name}/scgpt_outputs"
os.makedirs(output_dir, exist_ok=True)

# Ensure raw counts are used
if ann_data.raw is not None:
    ann_data.X = ann_data.raw.X.copy()

flush_print("Filtering cells with missing 'author_cell_type' labels...")
ann_data = ann_data[
    ann_data.obs["author_cell_type"].notnull() &
    (ann_data.obs["author_cell_type"] != "nan")
].copy()
ann_data.obs["author_cell_type"] = ann_data.obs["author_cell_type"].astype(str)


#GTF Mapping function
def parse_gtf(gtf_file):
    ensembl_to_symbol = {}
    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if fields[2] != 'gene':
                continue
            attributes = fields[8]
            gene_id_match = re.search('gene_id "([^"]+)"', attributes)
            gene_name_match = re.search('gene_name "([^"]+)"', attributes)
            if gene_id_match and gene_name_match:
                ensembl_id = gene_id_match.group(1).split('.')[0]  # Strip version
                gene_symbol = gene_name_match.group(1)
                ensembl_to_symbol[ensembl_id] = gene_symbol
    return ensembl_to_symbol

flush_print("Parsing GTF...")
mapping_dict = parse_gtf("/iridisfs/ddnb/faircells/AI_hackathon25/gencode.v48.basic.annotation.gtf.gz")

# =Map Ensembl IDs to gene symbols safely inside AnnData.var ===
flush_print("Mapping Ensembl IDs to gene symbols...")

# Map Ensembl IDs to gene symbols
ann_data.var['ensembl_clean'] = ann_data.var_names.str.replace(r'\\..*$', '', regex=True)
ann_data.var['gene_symbol'] = ann_data.var['ensembl_clean'].map(mapping_dict)
# Remove genes without gene symbol mapping
ann_data = ann_data[:, ann_data.var['gene_symbol'].notnull()]
# Remove duplicate gene symbols
ann_data = ann_data[:, ~ann_data.var['gene_symbol'].duplicated()]
# Assign gene symbols to var_names
ann_data.var_names = ann_data.var['gene_symbol']

# === Load scGPT vocabulary and subset ===
flush_print("Loading scGPT vocabulary...")
with open("scgpt_vocab.txt") as f:
    vocab_list = [line.strip() for line in f]

# Subset AnnData to genes present in vocabulary
genes_in_vocab = [g for g in ann_data.var_names if g in vocab_list]
ann_data = ann_data[:, genes_in_vocab]
# Reorder genes to match vocab order
ordered_genes = [g for g in vocab_list if g in ann_data.var_names]
ann_data = ann_data[:, ordered_genes]

flush_print("Processing data for scGPT...")
dataset = scgpt.process_data(ann_data)
flush_print("Computing embeddings...")
embeddings = scgpt.get_embeddings(dataset)

# Save embeddings into AnnData
if isinstance(embeddings, torch.Tensor):
    ann_data.obsm["X_scgpt"] = embeddings.cpu().numpy() if torch.cuda.is_available() else embeddings.numpy()
else:
    ann_data.obsm["X_scgpt"] = embeddings 

# Compute neighbors using scGPT embeddings
sc.pp.neighbors(ann_data, use_rep="X_scgpt")
# Compute UMAP based on those neighbors
sc.tl.umap(ann_data)
ann_data.obsm["X_umap_scgpt"] = ann_data.obsm["X_umap"]

# Extract UMAP coordinates into a dataframe
plot_df = pd.DataFrame(ann_data.obsm["X_umap_scgpt"], columns=["UMAP1", "UMAP2"])
plot_df["Cell Type"] = ann_data.obs["author_cell_type"].to_numpy()


print("obs columns:", ann_data.obs.columns)
print(ann_data.obs["author_cell_type"].head())


flush_print("Plotting and saving full UMAP...")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", ax=axs[0], s=100)
axs[0].set_title("UMAP: No Labels")
sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", hue="Cell Type", ax=axs[1], s=5)
axs[1].set_title("UMAP: Colored by Cell Type")
plt.tight_layout()
umap_plot_path = os.path.join(output_dir, f"{base_name}_scGPT_umap.png")
plt.savefig(umap_plot_path)
flush_print(f"UMAP plot saved to {umap_plot_path}")
plt.close()

#save the AnnData object with embeddings
flush_print("Saving AnnData with scGPT embeddings...")
output_file = os.path.join(output_dir,  f"{base_name}_scGPT.h5ad")
ann_data.write_h5ad(output_file)
flush_print(f"Saving AnnData to {output_file}...")


print(plot_df.shape)
print(plot_df.head())
print(plot_df.isna().sum())


print(f"AnnData shape after vocab filtering: {ann_data.shape}")
print("Embedding variance:", np.var(embeddings, axis=0).mean())

print("Embedding variance per dimension:")
print(np.var(embeddings, axis=0))


ann_data.write_h5ad(output_file)
