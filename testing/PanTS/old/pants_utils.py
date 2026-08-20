import os

import pandas as pd

DEFAULT_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "PanTS", "PanTS_metadata.csv"
)
DEFAULT_SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "PanTS")

# PanTS's official split: no split column in the metadata, split is encoded
# directly in the PanTS ID range.
TRAIN_ID_RANGE = (1, 9000)
TEST_ID_RANGE = (9001, 9901)

TRAIN_NEG_COUNT = 309
TEST_NEG_COUNT = 50


def load_pants_metadata(csv_path: str = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["id_num"] = df["PanTS ID"].str.extract(r"(\d+)").astype(int)
    return df


def get_official_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_lo, train_hi = TRAIN_ID_RANGE
    test_lo, test_hi = TEST_ID_RANGE
    train_df = df[df["id_num"].between(train_lo, train_hi)]
    test_df = df[df["id_num"].between(test_lo, test_hi)]
    return train_df, test_df


def get_tumor_positive_ids(df: pd.DataFrame) -> list[str]:
    mask = df["tumor?"].astype(int) >= 1
    return df.loc[mask, "PanTS ID"].tolist()


def get_tumor_negative_pool(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["tumor?"].astype(int) == 0
    return df.loc[mask]


def read_pants_ids(txt_path: str) -> list[str]:
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def write_pants_ids(ids: list[str], txt_path: str) -> None:
    with open(txt_path, "w") as f:
        f.write("\n".join(ids) + "\n")


def phase_distribution(df: pd.DataFrame, ids: list[str]) -> pd.Series:
    """Ct-phase proportions for the given ids. Rows with a missing phase are
    dropped before normalizing, since there is no negative-pool phase to
    match them against."""
    subset = df[df["PanTS ID"].isin(ids)]
    return subset["ct phase"].value_counts(normalize=True)


def rounded_phase_counts(distribution: pd.Series, n: int) -> dict[str, int]:
    raw = distribution * n
    counts = raw.round().astype(int).to_dict()
    drift = n - sum(counts.values())
    if drift != 0:
        largest_phase = max(counts, key=counts.get)
        counts[largest_phase] += drift
    return counts


def sample_ids_matching_phase(
    pool_df: pd.DataFrame,
    phase_counts: dict[str, int],
    seed: int = 42,
) -> list[str]:
    sampled_ids: list[str] = []
    for phase, count in phase_counts.items():
        if count <= 0:
            continue
        phase_pool = pool_df[pool_df["ct phase"] == phase]
        sampled = phase_pool.sample(n=count, random_state=seed)
        sampled_ids.extend(sampled["PanTS ID"].tolist())
    return sampled_ids


def build_pants_lesion_splits(
    csv_path: str = DEFAULT_METADATA_PATH,
    output_dir: str = DEFAULT_SPLIT_DIR,
    seed: int = 42,
    train_neg_n: int = TRAIN_NEG_COUNT,
    test_neg_n: int = TEST_NEG_COUNT,
) -> dict[str, list[str]]:
    """Build PanTS train/test splits on top of the official ID-range split.

    All tumor-positive scans are kept (926 train, 151 test). Negatives are
    sampled separately from each split's own negative pool, matching that
    split's own positive ct-phase distribution -- the train and test
    positive phase distributions are not assumed to match each other, and
    negatives never cross the official train/test boundary.
    """
    df = load_pants_metadata(csv_path)
    train_df, test_df = get_official_split(df)

    train_pos_ids = get_tumor_positive_ids(train_df)
    test_pos_ids = get_tumor_positive_ids(test_df)

    train_pos_phase_dist = phase_distribution(train_df, train_pos_ids)
    test_pos_phase_dist = phase_distribution(test_df, test_pos_ids)

    train_neg_pool = get_tumor_negative_pool(train_df)
    test_neg_pool = get_tumor_negative_pool(test_df)

    train_neg_counts = rounded_phase_counts(train_pos_phase_dist, train_neg_n)
    test_neg_counts = rounded_phase_counts(test_pos_phase_dist, test_neg_n)

    train_neg_ids = sample_ids_matching_phase(train_neg_pool, train_neg_counts, seed)
    test_neg_ids = sample_ids_matching_phase(test_neg_pool, test_neg_counts, seed)

    os.makedirs(output_dir, exist_ok=True)
    write_pants_ids(train_pos_ids, os.path.join(output_dir, "train_pos.txt"))
    write_pants_ids(train_neg_ids, os.path.join(output_dir, "train_neg.txt"))
    write_pants_ids(test_pos_ids, os.path.join(output_dir, "test_pos.txt"))
    write_pants_ids(test_neg_ids, os.path.join(output_dir, "test_neg.txt"))

    return {
        "train_pos": train_pos_ids,
        "train_neg": train_neg_ids,
        "test_pos": test_pos_ids,
        "test_neg": test_neg_ids,
    }


def subsample_pos_neg(
    df: pd.DataFrame,
    pos_ids: list[str],
    neg_ids: list[str],
    n: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Draw an n-sized subsample from pos_ids + neg_ids, preserving their
    pos:neg ratio and each side's own ct-phase distribution."""
    pos_pool = df[df["PanTS ID"].isin(pos_ids)]
    neg_pool = df[df["PanTS ID"].isin(neg_ids)]

    pos_n = round(n * len(pos_ids) / (len(pos_ids) + len(neg_ids)))
    neg_n = n - pos_n

    pos_counts = rounded_phase_counts(phase_distribution(df, pos_ids), pos_n)
    neg_counts = rounded_phase_counts(phase_distribution(df, neg_ids), neg_n)

    pos_sample = sample_ids_matching_phase(pos_pool, pos_counts, seed)
    neg_sample = sample_ids_matching_phase(neg_pool, neg_counts, seed)
    return pos_sample, neg_sample


def build_train_subsamples(
    df: pd.DataFrame,
    train_pos_ids: list[str],
    train_neg_ids: list[str],
    output_dir: str = DEFAULT_SPLIT_DIR,
    sample_350_seed: int = 42,
    sample_50_seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
) -> dict[str, list[str]]:
    """Build the 350-scan train subsample and five independent 50-scan train
    subsamples, each drawn straight from the full train pool (not nested
    inside the 350 sample) and each preserving the full train set's pos:neg
    ratio and per-class phase distribution. Test is never touched."""
    results: dict[str, list[str]] = {}

    pos_350, neg_350 = subsample_pos_neg(df, train_pos_ids, train_neg_ids, 350, sample_350_seed)
    results["train_sample_350_pos"] = pos_350
    results["train_sample_350_neg"] = neg_350

    for seed in sample_50_seeds:
        pos_50, neg_50 = subsample_pos_neg(df, train_pos_ids, train_neg_ids, 50, seed)
        results[f"train_sample_50_seed{seed}_pos"] = pos_50
        results[f"train_sample_50_seed{seed}_neg"] = neg_50

    os.makedirs(output_dir, exist_ok=True)
    for name, ids in results.items():
        write_pants_ids(ids, os.path.join(output_dir, f"{name}.txt"))

    return results


def verify_split_integrity(df: pd.DataFrame, splits: dict[str, list[str]]) -> list[str]:
    """Check every id in a pos/neg-suffixed split: pos ids have tumor?>=1,
    neg ids have tumor?==0, train_* ids fall in TRAIN_ID_RANGE and test_*
    ids fall in TEST_ID_RANGE. Returns a list of problem descriptions (empty
    if everything checks out)."""
    problems: list[str] = []
    indexed = df.set_index("PanTS ID")

    for name, ids in splits.items():
        split_kind, pos_neg = name.split("_", 1)[0], name.rsplit("_", 1)[-1]
        rows = indexed.loc[ids]

        if pos_neg == "pos" and not (rows["tumor?"].astype(int) >= 1).all():
            problems.append(f"{name}: contains a row with tumor? < 1")
        if pos_neg == "neg" and not (rows["tumor?"].astype(int) == 0).all():
            problems.append(f"{name}: contains a row with tumor? != 0")

        if split_kind == "train" and not rows["id_num"].between(*TRAIN_ID_RANGE).all():
            problems.append(f"{name}: contains an id outside TRAIN_ID_RANGE")
        if split_kind == "test" and not rows["id_num"].between(*TEST_ID_RANGE).all():
            problems.append(f"{name}: contains an id outside TEST_ID_RANGE")

    return problems


def split_phase_crosstab(df: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """phase x (split, pos/neg) crosstab for verifying the sampled negatives
    track their split's own positive phase distribution."""
    rows = []
    for split_name, ids in splits.items():
        split_kind, pos_neg = split_name.rsplit("_", 1)
        subset = df[df["PanTS ID"].isin(ids)]
        for phase, count in subset["ct phase"].value_counts(dropna=False).items():
            rows.append(
                {"phase": phase, "split": split_kind, "label": pos_neg, "count": count}
            )
    long_df = pd.DataFrame(rows)
    return long_df.pivot_table(
        index="phase", columns=["split", "label"], values="count", fill_value=0
    )
