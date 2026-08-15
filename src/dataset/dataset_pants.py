from pathlib import Path
import itertools
import os
from dotenv import load_dotenv

load_dotenv()

dataset_dir = Path(os.getenv("DATASET_DIR"))

def get_folds() -> dict:
    folds_dir = Path(__file__).resolve().parent.parent / "data"

    files = sorted(folds_dir.glob("fold_[0-9]_ids.txt"))

    folds = dict()

    for file in files:
        with open(file, mode="r") as f:
            ids = f.read().split("\n")

        file_name = str(file.name).split("_id")[0]
        folds[file_name] = ids

    return folds

def get_dataset(dataset_dir: Path) -> dict:
    dataset = dict()

    dataset["images"] = [x for x in (dataset_dir / "images").iterdir() if x.is_dir()]
    dataset["masks"] = [x for x in (dataset_dir / "masks").iterdir() if x.is_dir()]

    return dataset


def preprocess_dataset(dataset: dict):
    for fold in dataset.keys():
        train_set = [f[1] for f in dataset.items() if f[0] != fold]
        train_set = list(itertools.chain.from_iterable(train_set))

        test_set = dataset[fold]