import os

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "AbdomenAtlas3.0MiniWithMeta.csv"
)
DEFAULT_PANCREATIC_LESION_IDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "pancreatic_lesion_bdmap_ids.txt"
)
DEFAULT_COOCCURRING_LESION_IDS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "pancreatic_and_liver_or_kidney_lesion_bdmap_ids.txt",
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


def read_bdmap_ids(txt_path: str) -> list[str]:
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def split_pancreatic_lesion_dataset(
    train_size: int = 1000,
    seed: int = 42,
    pancreatic_ids_path: str = DEFAULT_PANCREATIC_LESION_IDS_PATH,
    cooccurring_ids_path: str = DEFAULT_COOCCURRING_LESION_IDS_PATH,
) -> tuple[list[str], list[str]]:
    pancreatic_ids = read_bdmap_ids(pancreatic_ids_path)
    cooccurring_ids = set(read_bdmap_ids(cooccurring_ids_path))
    is_cooccurring = [bdmap_id in cooccurring_ids for bdmap_id in pancreatic_ids]

    train_ids, test_ids = train_test_split(
        pancreatic_ids,
        train_size=train_size,
        random_state=seed,
        stratify=is_cooccurring,
    )
    return train_ids, test_ids

