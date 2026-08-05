import os

import pandas as pd

DEFAULT_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "AbdomenAtlas3.0MiniWithMeta.csv"
)


def load_abdomenatlas_metadata(csv_path: str = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_pancreatic_lesion_bdmap_ids(df: pd.DataFrame) -> list[str]:
    mask = df["number of pancreatic lesion instances"] >= 1
    return df.loc[mask, "BDMAP ID"].tolist()


def get_pancreatic_and_liver_or_kidney_lesion_bdmap_ids(df: pd.DataFrame) -> list[str]:
    pancreas = df["number of pancreatic lesion instances"] >= 1
    liver = df["number of liver lesion instances"] >= 1
    kidney = df["number of kidney lesion instances"] >= 1
    mask = pancreas & (liver | kidney)
    return df.loc[mask, "BDMAP ID"].tolist()

