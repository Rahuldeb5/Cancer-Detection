from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)

class PanTSDataset(Dataset):
    def __init__(self, filenames: list[str], cache_dir: Path, fingerprint: dict, train: bool,
                 spatial_size=(128, 128, 128), pos=2, neg=1, num_samples=4):
        self.filenames = filenames
        self.cache_dir = cache_dir
        self.fingerprint = fingerprint
        self.train = train

        self.train_transforms = get_train_transforms(spatial_size=tuple(spatial_size), pos=pos, neg=neg, num_samples=num_samples)
        self.val_transforms = get_val_transforms()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        case_id = self.filenames[idx]

        image_path = self.cache_dir / case_id / "image.npy"
        label_path = self.cache_dir / case_id / "label.npy"

        img = np.load(image_path).astype(np.float32)
        lab = np.load(label_path)

        img = np.clip(img, self.fingerprint["lo"], self.fingerprint["hi"])
        img = (img - self.fingerprint["mean"]) / self.fingerprint["std"]

        data = {
            "image": img,
            "label": lab
        }

        if self.train:
            data = self.train_transforms(data)
        else:
            data = self.val_transforms(data)

        return data


def get_train_transforms(spatial_size=(128, 128, 128), pos=2, neg=1, num_samples=4) -> Compose:
    return Compose([
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandGaussianNoised(keys="image", prob=0.15),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    return Compose([
        EnsureTyped(keys=["image", "label"]),
    ])