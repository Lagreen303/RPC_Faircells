import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Paths
avgbio_path = "/iridisfs/ddnb/faircells/AI_hackathon25/summary_avgbio_scores.csv"
umap_path = "/iridisfs/ddnb/faircells/AI_hackathon25/summary_umap_group_stats.csv"
output_dir = "/iridisfs/ddnb/faircells/AI_hackathon25/plots/avgBIO_dotplots_by_dataset/"
os.makedirs(output_dir, exist_ok=True)

# Load and harmonize
df_avgbio = pd.read_csv(avgbio_path)
df_umap = pd.read_csv(umap_path)
for df in [df_avgbio, df_umap]:
    for col in ["dataset", "group", "groupby", "model"]:
        df[col] = df[col].astype(str).str.strip()
    df["groupby"] = df["groupby"].replace({
        "self": "ethnicity", "development": "age", "Age_quartile": "age", "Age": "age",
        "age_quartile": "age", "development_stage_quartile": "age", "self_reported_ethnicity": "ethnicity"
    })

# Merge n_cells
ncell_map = df_umap.dropna(subset=["group", "n_cells"])[["dataset", "group", "model", "n_cells"]].drop_duplicates()
df_avgbio = pd.merge(df_avgbio, ncell_map, on=["dataset", "group", "model"], how="left")

# Sorting helpers
age_order = ["<29", "<30", "30-38", "29-34", "39-47", "35-50", "<56", "56-58", "59-59",
             ">=48", ">=51", ">=60", "<41", "41-50", "51-70", ">=71"]
ethnicity_priority = {
    "european": 0, "indian": 1, "pakistani": 1, "bangladeshi": 1, "singaporean indian": 1,
    "asian": 2, "chinese": 2, "japanese": 2, "korean": 2, "thai": 2,
    "singaporean chinese": 2, "singaporean malay": 2,
    "african american or afro-caribbean": 3, "african american": 3, "black": 3,
    "hispanic or latin american": 4, "hispanic": 4, "latin": 4, "latino": 4
}
sex_priority = {"male": 0, "female": 1}
def group_sort_key(row):
    group = row["group"].lower()
    gb = row["groupby"]
    if gb == "age":
        return age_order.index(group) if group in age_order else len(age_order)
    elif gb == "sex":
        return sex_priority.get(group, 2)
    elif gb == "ethnicity":
        for k, v in ethnicity_priority.items():
            if k in group: return v
        return 5
    elif gb == "disease":
        return 0 if "normal" in group else 1
    return 99

# Main processing loop (collect all dataframes)
all_dfs = []
for dataset in sorted(df_avgbio["dataset"].unique()):
    df_ds = df_avgbio[df_avgbio["dataset"] == dataset].copy()

    # Get baseline (All) per model
    baselines = df_ds[df_ds["group"] == "All"][["model", "AvgBIO"]].rename(columns={"AvgBIO": "baseline"})

    # Remove baseline rows from output
    df_out = df_ds[df_ds["group"] != "All"].copy()
    df_out = pd.merge(df_out, baselines, on="model", how="left")
    df_out["delta"] = df_out["AvgBIO"] - df_out["baseline"]

    # Create unique full label: "groupby: group"
    df_out["group_label"] = df_out["groupby"].str.capitalize() + ": " + df_out["group"]
    df_out["sort_key"] = df_out.apply(group_sort_key, axis=1)

    # Sort labels within groupby then within dataset
    df_out = df_out.sort_values(["groupby", "sort_key", "group_label"])
    label_order = df_out["group_label"].drop_duplicates().tolist()
    df_out["group_label"] = pd.Categorical(df_out["group_label"], categories=label_order, ordered=True)

    all_dfs.append(df_out)

# Concatenate all datasets and save as one CSV
final_df = pd.concat(all_dfs, axis=0)
final_df.to_csv(os.path.join(output_dir, "avgBIO_dotdata_ALL.csv"), index=False)

print("✅ Saved all avgBIO dotplot dataframes as a single CSV.")

# Main plotting loop
for dataset in sorted(df_avgbio["dataset"].unique()):
    df_ds = df_avgbio[df_avgbio["dataset"] == dataset].copy()

    # Get baseline (All) per model
    baselines = df_ds[df_ds["group"] == "All"][["model", "AvgBIO"]].rename(columns={"AvgBIO": "baseline"})

    # Remove baseline rows from plot
    df_plot = df_ds[df_ds["group"] != "All"].copy()
    df_plot = pd.merge(df_plot, baselines, on="model", how="left")
    df_plot["delta"] = df_plot["AvgBIO"] - df_plot["baseline"]

    # Create unique full label: "groupby: group"
    df_plot["group_label"] = df_plot["groupby"].str.capitalize() + ": " + df_plot["group"]
    df_plot["sort_key"] = df_plot.apply(group_sort_key, axis=1)

    # Sort labels within groupby then within dataset
    df_plot = df_plot.sort_values(["groupby", "sort_key", "group_label"])
    label_order = df_plot["group_label"].drop_duplicates().tolist()
    df_plot["group_label"] = pd.Categorical(df_plot["group_label"], categories=label_order, ordered=True)

    # Plot
    plt.figure(figsize=(8, max(6, len(label_order) * 0.35)))
    norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.1)
    ax = sns.scatterplot(
        data=df_plot,
        x="model", y="group_label",
        hue="delta", size="n_cells",
        palette="coolwarm_r", hue_norm=norm, sizes=(30, 300),
        legend="brief"
    )

    plt.title(f"Δ avgBIO (Group − All)\nDataset: {dataset}")
    plt.xlabel("Model")
    plt.ylabel("Group")

    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"avgBIO_dotplot_{dataset}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

print("✅ Saved all avgBIO dotplots by dataset.")

