import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from matplotlib.patches import Patch

# Data
data = [
    ["TasNet", "DNN", 2503296, 0.132, 0.887, "[0.860, 0.913]", 8.97, 60.90, "167.9 min"],
    ["Attention-Stem Multi-Head Multiscale TCN", "SNN", 78070, 0.171, 0.811, "[0.766, 0.857]", 4.52, 23.51, "81.2 min"],
    ["UNet1D", "DNN", 2710819, 0.159, 0.778, "[0.733, 0.820]", 4.21, 21.70, "67.4 min"],
    ["Attention-Stem Multi-Head Multiscale Branches", "SNN", 17638, 0.225, 0.777, "[0.739, 0.816]", 3.97, 5.77, "30.7 min"],
    ["SepFormer", "DNN", 601088, 0.184, 0.727, "[0.692, 0.761]", 0.15, 13.14, "66.0 min"],
    ["Attention-Stem Bilinear-Fusion Multi-Head Multiscale Branches", "SNN", 29302, 0.217, 0.693, "[0.628, 0.760]", 3.14, 9.38, "41.4 min"],
    ["Shallow Conv1D", "SNN", 5443, 0.294, 0.668, "[0.634, 0.703]", 2.46, 1.85, "10.4 min"],
    ["Autoencoder", "SNN", 23651, 0.145, 0.660, "[0.617, 0.701]", 3.08, 57.72, "40.1 min"],
    ["Multiple-Head Multiscale Branches", "SNN", 5859, 0.241, 0.659, "[0.614, 0.701]", 2.14, 4.47, "19.7 min"],
    ["Multiscale Dilated", "SNN", 12867, 0.276, 0.644, "[0.609, 0.680]", 2.10, 3.06, "10.5 min"],
    ["Multiscale Branches", "SNN", 5859, 0.241, 0.643, "[0.602, 0.684]", 1.77, 3.42, "17.3 min"],
    ["SingleHead MLP", "SNN", 643, 0.395, 0.557, "[0.544, 0.569]", 1.35, 0.32, "11.6 min"],
    ["Fuse Shallow", "SNN", 5859, 0.238, 0.545, "[0.491, 0.599]", 0.96, 3.16, "9.4 min"],
]

df = pd.DataFrame(data, columns=[
    "Model", "Type", "Params", "Loss", "Corr", "CI", "SNR", "Inf_ms", "Train"
])

# Sort by correlation descending
df = df.sort_values("Corr", ascending=False).reset_index(drop=True)

# Wrap long labels
df["Label"] = df["Model"].apply(lambda x: "\n".join(textwrap.wrap(x, width=18)))

# Color map
color_map = {"DNN": "#d62728", "SNN": "#1f77b4"}
colors = df["Type"].map(color_map)

# Plot
plt.figure(figsize=(15, 7))
bars = plt.bar(df["Label"], df["Corr"], color=colors, edgecolor="black", linewidth=0.4)

plt.title("Macro correlation per model", fontsize=16)
plt.ylabel("Macro correlation", fontsize=13)
plt.ylim(0, 0.95)
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", linestyle="-", alpha=0.25)

# Legend
legend_elements = [
    Patch(facecolor="#d62728", edgecolor="black", label="DNN"),
    Patch(facecolor="#1f77b4", edgecolor="black", label="SNN"),
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()

# Save figure
plt.savefig("macro_correlation_per_model.png", dpi=300, bbox_inches="tight")
plt.show()