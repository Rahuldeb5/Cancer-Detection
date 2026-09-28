"""Two heatmaps: lesion hit-count and mean nnU-Net Dice, both by
(long-axis position in the gland) x (diameter bin). Single-hue sequential
colormaps throughout -- both are magnitude quantities (a count, a 0-1 score),
not signed/diverging ones.

Usage (from repo root): .venv/bin/python3 src/lesion-sectioning/plot_hitrate_dice.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("src/lesion-sectioning/work")
DIAM_LABELS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-40mm", "40mm+"]
PC1_LABELS = ["head 0-20%", "20-40%", "mid 40-60%", "60-80%", "tail 80-100%"]


def heatmap(ax, data: pd.DataFrame, title: str, cmap: str, fmt: str, cbar_label: str) -> None:
    data = data.reindex(index=DIAM_LABELS, columns=PC1_LABELS)
    im = ax.imshow(data.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(PC1_LABELS)))
    ax.set_xticklabels(PC1_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(DIAM_LABELS)))
    ax.set_yticklabels(DIAM_LABELS)
    ax.set_title(title)
    ax.set_xlabel("position along pancreas long axis (PCA-reoriented, head->tail)")

    vals = data.values.astype(float)
    finite = vals[~np.isnan(vals)]
    thresh = finite.min() + (finite.max() - finite.min()) * 0.6 if finite.size else 0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > thresh else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def main() -> None:
    counts = pd.read_csv(RESULTS_DIR / "hitrate_counts_pc1_x_diambin.csv", index_col=0)
    dice_mean = pd.read_csv(RESULTS_DIR / "dice_mean_pc1_x_diambin.csv", index_col=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    heatmap(ax1, counts, "Lesion hit-rate: count by position x size", "Blues", "g", "n lesions")
    heatmap(ax2, dice_mean, "Mean nnU-Net Dice by position x size", "Blues", ".2f", "mean lesion Dice")
    fig.suptitle("Pancreatic lesion location (PCA long-axis, head->tail) vs. size and detectability")
    fig.tight_layout()

    out_path = RESULTS_DIR / "hitrate_dice_heatmap.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
