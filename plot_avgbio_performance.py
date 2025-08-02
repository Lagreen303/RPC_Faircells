import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input path and output dir
avgbio_path = "/iridisfs/ddnb/faircells/AI_hackathon25/summary_avgbio_scores.csv"
output_dir = "/iridisfs/ddnb/faircells/AI_hackathon25/plots/avgbio_all_summary/"
os.makedirs(output_dir, exist_ok=True)

# Load
df = pd.read_csv(avgbio_path)

# Clean and filter for baseline only
df["group"] = df["group"].astype(str).str.strip()
df["dataset"] = df["dataset"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()
df = df[df["group"] == "All"]

# Pivot table for CSV export
pivot_df = df.pivot_table(index="dataset", columns="model", values="AvgBIO")
pivot_df.to_csv(os.path.join(output_dir, "avgBIO_baseline_summary.csv"))
print("✅ Saved summary CSV of avgBIO (All) scores.")

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="dataset", y="AvgBIO", hue="model")
plt.title("Overall Dataset avgBIO Scores")
plt.ylabel("AvgBIO")
plt.xlabel("Dataset")
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(output_dir, "avgBIO_baseline_barplot.png"), dpi=300)
plt.close()

print("✅ Saved baseline avgBIO barplot.")
