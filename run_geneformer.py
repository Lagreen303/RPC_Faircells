import sys
import os
import anndata as ad
from helical.models.geneformer import GeneformerConfig, Geneformer
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from pprint import pformat

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# === Output directory ===

input_file = "/iridisfs/ddnb/faircells/data/Green_BCL.h5ad"

# === Load and Prepare Data ===
flush_print("Reading AnnData...")
ann_data = ad.read_h5ad(input_file)
base_name = os.path.basename(input_file).replace(".h5ad", "")
ann_data = ann_data[:10]  # Limit to first 1000 cells for testing

output_dir = f"{base_name}/zeroshot_outputs"
os.makedirs(output_dir, exist_ok=True)

flush_print("Cleaning AnnData feature index...")
ann_data = ann_data[:, ann_data.var.index.notnull()].copy()
ann_data.var.index = ann_data.var.index.astype(str)

flush_print("Replacing AnnData with raw counts...")
ann_data = ad.AnnData(X=ann_data.raw.X, obs=ann_data.obs, var=ann_data.raw.var)
ann_data.var['ensembl_id'] = ann_data.var_names

flush_print("Filtering cells with missing 'celltype.l1' labels...")
ann_data = ann_data[
    ann_data.obs["celltype.l1"].notnull() &
    (ann_data.obs["celltype.l1"] != "nan")
].copy()
ann_data.obs["celltype.l1"] = ann_data.obs["celltype.l1"].astype(str)

flush_print(f"Final AnnData shape: {ann_data.shape[0]} cells, {ann_data.shape[1]} features.")

# === Split Data ===
flush_print("Shuffling and splitting into train/test...")
ann_data = ann_data[ann_data.obs.sample(frac=1, random_state=42).index]
train_data, test_data = train_test_split(ann_data, test_size=0.2, random_state=42)
train_labels = train_data.obs["celltype.l1"].values
test_labels = test_data.obs["celltype.l1"].values

# === Geneformer Setup ===
flush_print("Setting up Geneformer...")
device = "cuda" if torch.cuda.is_available() else "cpu"
config = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=8, nproc=8, device=device, accelerator=True)
geneformer = Geneformer(config)
flush_print("Geneformer configuration:\n" + pformat(vars(config)))

# === Embedding Extraction ===
flush_print("Processing training data...")
train_dataset = geneformer.process_data(train_data, gene_names='ensembl_id')
flush_print("Processing test data...")
test_dataset = geneformer.process_data(test_data, gene_names='ensembl_id')

flush_print("Generating train embeddings...")
flush_print(f"Initial train_data: {train_data.shape[0]} cells")
with torch.no_grad():
    train_emb = geneformer.get_embeddings(train_dataset)
flush_print(f"Tokenized dataset length: {len(train_dataset)}")

flush_print("Generating test embeddings...")
with torch.no_grad():
    test_emb = geneformer.get_embeddings(test_dataset)

flush_print(f"Train embedding shape: {train_emb.shape}")
flush_print(f"Test embedding shape: {test_emb.shape}")

# === Save Embeddings + Predictions ===
flush_print("Merging embeddings...")
train_idx = train_data.obs.index
test_idx = test_data.obs.index
emb_dict = {**dict(zip(train_idx, train_emb)), **dict(zip(test_idx, test_emb))}
ordered_embeddings = np.stack([emb_dict[i] for i in ann_data.obs.index])
if isinstance(ordered_embeddings, torch.Tensor):
    ann_data.obsm["X_zeroshot"] = ordered_embeddings.cpu().numpy() if torch.cuda.is_available() else ordered_embeddings.numpy()
else:
    ann_data.obsm["X_zeroshot"] = ordered_embeddings 

# === UMAP on full embedding ===
flush_print("Running UMAP on full dataset embeddings...")
reducer = umap.UMAP(min_dist=0.2, n_components=2, n_neighbors=3, random_state=42)
umap_coords = reducer.fit_transform(ordered_embeddings)
ann_data.obsm["X_umap_ZS"] = umap_coords

# Prepare plot DataFrame
plot_df = pd.DataFrame(umap_coords, columns=["px", "py"])
plot_df["Cell Type"] = ann_data.obs["celltype.l1"].values

flush_print("Plotting and saving full UMAP...")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
sns.scatterplot(data=plot_df, x="px", y="py", ax=axs[0], s=5)
axs[0].set_title("UMAP: No Labels")

sns.scatterplot(data=plot_df, x="px", y="py", hue="Cell Type", ax=axs[1], s=5)
axs[1].set_title("UMAP: Colored by Cell Type")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_name}_geneformer_umap_all_cells.png"), dpi=300)
plt.show()

# === Classification ===
flush_print("Fitting k-NN classifier...")
neigh = KNeighborsClassifier(n_neighbors=5, metric='cosine')
neigh.fit(train_emb, train_labels)

flush_print("Predicting test labels...")
pred_labels = neigh.predict(test_emb)

flush_print("Saving predicted labels to AnnData...")
ann_data.obs["predicted_celltype.l1"] = np.nan
ann_data.obs.loc[test_idx, "predicted_celltype.l1"] = pred_labels

output_file = os.path.join(output_dir,  f"{base_name}_ZS_GF.h5ad")
ann_data.write_h5ad(output_file)
flush_print(f"Saved final AnnData with embeddings + predictions:\n{output_file}")


flush_print("Classification Report:")
flush_print(classification_report(test_labels, pred_labels, digits=3))


flush_print("Plotting and saving confusion matrix...")
cm = confusion_matrix(test_labels, pred_labels)
labels = np.unique(np.concatenate((test_labels, pred_labels)))
fig, ax = plt.subplots(figsize=(20, 20))  # wider and taller
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Geneformer - kNN Classification")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_name}_geneformer_confusion_matrix.png"), dpi=300)
plt.show()

