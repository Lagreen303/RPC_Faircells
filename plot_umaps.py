import os
import sys
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

# === Utility ===
def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# === Config ===
input_configs = {
    "Ahern_covid_UK": {
        "zs_file": "Ahern_covid_UK/zeroshot_outputs/Ahern_covid_UK_ZS_GF.h5ad",
        "scgpt_file": "Ahern_covid_UK/scgpt_outputs/Ahern_covid_UK_scGPT.h5ad",
        "hvg_file": "/iridisfs/ddnb/faircells/data/data_hvg/Ahern_covid_UK_hvg.h5ad",
        "ground_truth_col": "major_subset"
    },
    "Green_BCL": {
        "zs_file": "Green_BCL/zeroshot_outputs/Green_BCL_ZS_GF.h5ad",
        "scgpt_file": "Green_BCL/scgpt_outputs/Green_BCL_scGPT.h5ad",
        "hvg_file": "/iridisfs/ddnb/faircells/data/data_hvg/Green_BCL_hvg.h5ad",
        "ground_truth_col": "author_cell_type"
    },
    "Kock_AIDAv2": {
        "zs_file": "Kock_AIDAv2/zeroshot_outputs/Kock_AIDAv2_ZS_GF.h5ad",
        "scgpt_file": "Kock_AIDAv2/scgpt_outputs/Kock_AIDAv2_scGPT.h5ad",
        "hvg_file": "/iridisfs/ddnb/faircells/data/data_hvg/Kock_AIDAv2_hvg.h5ad",
        "ground_truth_col": "Annotation_Level1"
    },
    "Perez_lupus": {
        "zs_file": "Perez_lupus/zeroshot_outputs/Perez_lupus_ZS_GF.h5ad",
        "scgpt_file": "Perez_lupus/scgpt_outputs/Perez_lupus_scGPT.h5ad",
        "hvg_file": "/iridisfs/ddnb/faircells/data/data_hvg/Perez_lupus_hvg.h5ad",
        "ground_truth_col": "author_cell_type"
    },
    "Tabula_immune": {
        "zs_file": "Tabula_immune/zeroshot_outputs/Tabula_immune_ZS_GF.h5ad",
        "scgpt_file": "Tabula_immune/scgpt_outputs/Tabula_immune_scGPT.h5ad",
        "hvg_file": "/iridisfs/ddnb/faircells/data/data_hvg/Tabula_immune_hvg.h5ad",
        "ground_truth_col": "broad_cell_class"
    }
}

model_labels = {
    "ZS_GF": "Geneformer",
    "scGPT": "scGPT",
    "raw": "HVG"
}

# === Collect UMAPs ===
os.makedirs("umapsmall", exist_ok=True)

# Set global font sizes
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 36,
    'axes.labelsize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'figure.titlesize': 40
})

for dataset, config in input_configs.items():
    flush_print(f"\n--- Processing {dataset} ---")
    ground_truth_col = config["ground_truth_col"]
    all_umaps = []

    # Load ZS (Geneformer)
    adata_zs = ad.read_h5ad(config["zs_file"])
    if ground_truth_col in adata_zs.obs:
        labels_zs = adata_zs.obs[ground_truth_col].astype(str)
        valid_mask_zs = labels_zs.notna() & (labels_zs != 'nan')
        labels_zs = labels_zs[valid_mask_zs]
        umap_zs = adata_zs.obsm["X_umap_ZS"][valid_mask_zs.values]
    else:
        labels_zs = pd.Series([], dtype=str)
        umap_zs = np.empty((0, 2))
    unique_celltypes = sorted(np.unique(labels_zs))
    palette = sns.color_palette("husl", len(unique_celltypes))
    lut = dict(zip(unique_celltypes, palette))
    df_zs = pd.DataFrame(umap_zs, columns=["UMAP1", "UMAP2"])
    df_zs["Model"] = model_labels["ZS_GF"]
    df_zs["Dataset"] = dataset
    df_zs["Celltype"] = labels_zs.values
    all_umaps.append(df_zs)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_zs, x="UMAP1", y="UMAP2", hue="Celltype", palette=lut, s=1.2)
    plt.title(f"{dataset} - Geneformer", fontsize=50, fontweight='bold')
    plt.xlabel("UMAP1", fontsize=28)
    plt.ylabel("UMAP2", fontsize=28)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24, title_fontsize=28)
    plt.tight_layout()
    plt.savefig(f"umapsmall/{dataset}_Geneformer.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Load scGPT
    adata_sc = ad.read_h5ad(config["scgpt_file"])
    if ground_truth_col in adata_sc.obs:
        labels_sc = adata_sc.obs[ground_truth_col].astype(str)
        valid_mask_sc = labels_sc.notna() & (labels_sc != 'nan')
        labels_sc = labels_sc[valid_mask_sc]
        umap_sc = adata_sc.obsm["X_umap_scgpt"][valid_mask_sc.values]
    else:
        labels_sc = pd.Series([], dtype=str)
        umap_sc = np.empty((0, 2))
    df_sc = pd.DataFrame(umap_sc, columns=["UMAP1", "UMAP2"])
    df_sc["Model"] = model_labels["scGPT"]
    df_sc["Dataset"] = dataset
    df_sc["Celltype"] = labels_sc.values
    all_umaps.append(df_sc)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_sc, x="UMAP1", y="UMAP2", hue="Celltype", palette=lut, s=1.2)
    plt.title(f"{dataset} - scGPT", fontsize=50, fontweight='bold')
    plt.xlabel("UMAP1", fontsize=28)
    plt.ylabel("UMAP2", fontsize=28)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24, title_fontsize=28)
    plt.tight_layout()
    plt.savefig(f"umapsmall/{dataset}_scGPT.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Load HVG UMAP
    adata_hvg = ad.read_h5ad(config["hvg_file"])
    if ground_truth_col in adata_hvg.obs:
        labels_hvg = adata_hvg.obs[ground_truth_col].astype(str)
        valid_mask_hvg = labels_hvg.notna() & (labels_hvg != 'nan')
        labels_hvg = labels_hvg[valid_mask_hvg]
        umap_hvg = adata_hvg.obsm["X_umap"][valid_mask_hvg.values]
    else:
        labels_hvg = pd.Series([], dtype=str)
        umap_hvg = np.empty((0, 2))
    df_hvg = pd.DataFrame(umap_hvg, columns=["UMAP1", "UMAP2"])
    df_hvg["Model"] = model_labels["raw"]
    df_hvg["Dataset"] = dataset
    df_hvg["Celltype"] = labels_hvg.values
    all_umaps.append(df_hvg)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_hvg, x="UMAP1", y="UMAP2", hue="Celltype", palette=lut, s=1.2)
    plt.title(f"{dataset} - HVG", fontsize=50, fontweight='bold')
    plt.xlabel("UMAP1", fontsize=28)
    plt.ylabel("UMAP2", fontsize=28)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24, title_fontsize=28)
    plt.tight_layout()
    plt.savefig(f"umapsmall/{dataset}_HVG.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Combine and plot per dataset
    combined = pd.concat(all_umaps, ignore_index=True)
    g = sns.FacetGrid(
        combined,
        row="Model",
        hue="Celltype",
        height=12,
        aspect=1.2,
        sharex=False,
        sharey=False,
        palette=lut
    )
    g.map_dataframe(sns.scatterplot, x="UMAP1", y="UMAP2", s=1.2)
    
    # Style the FacetGrid with larger fonts
    g.set_titles(row_template='{row_name}', size=36, weight='bold')
    g.set_axis_labels("UMAP1", "UMAP2", size=28)
    g.set_xticklabels(size=28)
    g.set_yticklabels(size=28)
    
    # Add legend with larger font
    g.add_legend(title="Cell Type", title_fontsize=32, fontsize=24)
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"{dataset}", fontsize=44, fontweight='bold')
    plt.savefig(f"umapsmall/{dataset}_combined_grid.png", dpi=300, bbox_inches='tight')
    plt.close()

flush_print("\n✅ All UMAPs processed and saved.")


from PIL import Image

# List of combined grid PNGs in the order of input_configs
combined_grid_files = [
    f"umapsmall/{dataset}_combined_grid.png"
    for dataset in input_configs.keys()
]

# Open all images
images = [Image.open(fname) for fname in combined_grid_files]

# Calculate total width and max height
total_width = sum(img.width for img in images)
max_height = max(img.height for img in images)

# Create a new blank image with the appropriate size
new_im = Image.new('RGB', (total_width, max_height))

# Paste each image side by side
x_offset = 0
for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.width

# Save the joined image
new_im.save("umapsmall/all_combined_side_by_side.png")
flush_print("\n✅ All combined grids joined and saved as umapsmall/all_combined_side_by_side.png")