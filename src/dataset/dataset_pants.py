from pathlib import Path
import os
from dotenv import load_dotenv
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)

import logging
import gc
import ctypes
import json

TARGET_SPACING = (1.0, 1.0, 1.5)

FINGERPRINT_PATH = Path(__file__).resolve().parent.parent / "data" / "fingerprint.json"

logger = logging.getLogger(__name__)

def _trim_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

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

def index_by_id(paths: list[Path]) -> dict[str, Path]:
    return {p.name: p for p in paths}

def _collect_pancreas_hu(case_id: str, image_index: dict, mask_index: dict):
    try:
        ct_path = image_index[case_id] / "ct.nii.gz"
        seg_dir = mask_index[case_id] / "segmentations"
        pancreas_path = seg_dir / "pancreas.nii.gz"
        lesion_path = seg_dir / "pancreatic_lesion.nii.gz"

        if not (ct_path.exists() and pancreas_path.exists()):
            return None

        ct = nib.load(ct_path).dataobj[...]
        pancreas_mask = nib.load(pancreas_path).dataobj[...] > 0.0
        lesion_mask = nib.load(lesion_path).dataobj[...] > 0.0 if lesion_path.exists() else np.zeros_like(pancreas_mask)

        foreground = pancreas_mask | lesion_mask

        return ct[foreground].astype(np.float32)

    except Exception as e:
        logger.error(f"Error processing {case_id}: {e}")
        return None

def compute_global_fingerprint(case_ids, image_index, mask_index, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker = partial(_collect_pancreas_hu, image_index=image_index, mask_index=mask_index)
        chunks = list(executor.map(worker, case_ids))

    pooled = np.concatenate([c for c in chunks if c is not None and c.size > 0])

    lo, hi = np.percentile(pooled, [0.5, 99.5])

    return {
        "lo": float(lo),
        "hi": float(hi),
        "mean": float(pooled.mean()),
        "std": float(pooled.std()),
    }

def get_or_compute_fingerprint(case_ids, image_index, mask_index):
    if FINGERPRINT_PATH.exists():
        logger.info(f"loading cached fingerprint from {FINGERPRINT_PATH}")
        return json.loads(FINGERPRINT_PATH.read_text())

    logger.info(f"computing global fingerprint over {len(case_ids)} scans")
    fingerprint = compute_global_fingerprint(case_ids, image_index, mask_index)
    logger.info(f"fingerprint: {fingerprint}")

    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(json.dumps(fingerprint, indent=2))

    return fingerprint

def build_cache_transform() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=TARGET_SPACING,
                 mode=("bilinear", "nearest")),
    ])

def preprocess_scan(case_id: str, transform: Compose, image_index: dict, mask_index: dict, cache_dir: Path):
    scan_dir = cache_dir / case_id
    if (scan_dir / "image.npy").exists() and (scan_dir / "label.npy").exists():
        return True

    try:
        image_path = image_index[case_id] / "ct.nii.gz"
        label_path = mask_index[case_id] / "segmentations" / "pancreatic_lesion.nii.gz"

        data = transform({"image": image_path, "label": label_path})

        image = np.round(data["image"].numpy()).astype(np.int16)
        label = data["label"].numpy().astype(np.uint8)

        scan_dir.mkdir(parents=True, exist_ok=True)

        np.save(scan_dir / "image.npy", image)
        np.save(scan_dir / "label.npy", label)

        return True

    except Exception as e:
        logger.error(f"Error preprocessing {case_id}: {e}")
        return False


def _run_preprocess_pool(case_ids, transform, image_index, mask_index, cache_dir, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker = partial(preprocess_scan, transform=transform, image_index=image_index,
                          mask_index=mask_index, cache_dir=cache_dir)
        futures = {executor.submit(worker, case_id): case_id for case_id in case_ids}

        for i, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            ok = future.result()
            status = "ok" if ok else "FAILED"
            logger.info(f"{i}/{len(case_ids)}: {case_id} [{status}]")

            if i % 25 == 0:
                _trim_memory()


def preprocess_dataset():
    image_index = index_by_id([x for x in (dataset_dir / "images").iterdir() if x.is_dir()])
    mask_index = index_by_id([x for x in (dataset_dir / "masks").iterdir() if x.is_dir()])

    case_ids = sorted(image_index.keys())

    # geometry (orient/resample) and intensity normalization stats are both computed/cached
    # once, globally; normalization itself (clip + z-score) is applied later, at training time
    fingerprint = get_or_compute_fingerprint(case_ids, image_index, mask_index)

    transform = build_cache_transform()
    cache_dir = dataset_dir / "cache"

    _run_preprocess_pool(case_ids, transform, image_index, mask_index, cache_dir, max_workers=2)

    logger.info(f"done: {len(case_ids)} scans cached")

    return fingerprint


if __name__ == "__main__":
    log_path = Path(__file__).resolve().parent / "preprocess.log"
    setup_logging(log_path)

    logger.info("Starting dataset preprocessing")
    preprocess_dataset()
    logger.info("Finished dataset preprocessing")