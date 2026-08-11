import pandas as pd
import numpy as np
import ast
from pathlib import Path

# 1. Load the corrected mask metadata (pancreatic_lesion-based) and original metadata
df_masks = pd.read_csv("mask_metadata.csv")
df_masks.columns = df_masks.columns.str.strip()
df_masks["id"] = df_masks["id"].astype(str).str.strip()

df_original = pd.read_csv("data/pants - PanTS_metadata (1).csv")
df_original.columns = df_original.columns.str.strip()
df_original["PanTS ID"] = df_original["PanTS ID"].astype(str).str.strip()

# 2. Pull native Z spacing (mm) out of the "spacing" tuple string
def parse_spacing(spacing_str):
    try:
        return np.array(ast.literal_eval(str(spacing_str)), dtype=float)
    except (ValueError, SyntaxError):
        return np.array([np.nan, np.nan, np.nan])

spacing_matrix = np.vstack(df_original["spacing"].apply(parse_spacing).values)
df_original["Spacing_Z_mm"] = spacing_matrix[:, 2]

# 3. Merge Z spacing onto every measured lesion component
merged = df_masks.merge(
    df_original[["PanTS ID", "Spacing_Z_mm"]],
    left_on="id", right_on="PanTS ID", how="left"
)

# 4. Only rows with an actual measured component (drop the num_components==0 placeholder rows)
real = merged[merged["min_axis_extent_mm"].notna()].copy()

# 5. Flag components where the slice thickness gives fewer than ~3 real voxels
#    across the lesion's narrowest axis -- i.e. the scan can't reliably resolve it
real["z_undersampled"] = real["Spacing_Z_mm"] > (real["min_axis_extent_mm"] / 3.0)

undersampled = real[real["z_undersampled"]].sort_values("id")

# 6. Write results: one file with full detail, one with just the unique study ids to cut
detail_path = Path("undersampled_components_temp.txt")
with open(detail_path, "w") as f:
    f.write("id,component_index,min_axis_extent_mm,Spacing_Z_mm\n")
    for _, row in undersampled.iterrows():
        f.write(
            f"{row['id']},{int(row['component_index'])},"
            f"{row['min_axis_extent_mm']:.4f},{row['Spacing_Z_mm']:.4f}\n"
        )

ids_path = Path("undersampled_ids_temp.txt")
with open(ids_path, "w") as f:
    f.write("\n".join(sorted(undersampled["id"].unique())))

print(f"{len(undersampled)} undersampled components across {undersampled['id'].nunique()} study ids")
print(f"wrote detail -> {detail_path}")
print(f"wrote ids     -> {ids_path}")
