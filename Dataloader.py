import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold


class SkinLesionDataset(Dataset):
    def __init__(self, hdf5_file_path, metadata_df, target=None, transform=None):
        self.hdf5_file_path = hdf5_file_path
        self.metadata_df = metadata_df
        self.images_ids = metadata_df["isic_id"].astype(
            str
        )  # Ensure image_id is a string
        self.labels = target
        self.transform = transform

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        image_id = self.images_ids.iloc[idx]
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            image = (
                np.array(Image.open(BytesIO(hdf5_file[image_id][()])), dtype=np.float32)
                / 255
            )

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.labels is not None:
            label = self.labels.iloc[idx]
            return image, label
        else:
            return image, image_id


def get_transforms(image_dim=256):
    train_transform = A.Compose(
        [
            A.Resize(image_dim, image_dim),
            A.PadIfNeeded(min_height=image_dim, min_width=image_dim),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(image_dim, image_dim),
            A.PadIfNeeded(min_height=image_dim, min_width=image_dim),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(
    metadata_df, hdf5_file_path, batch_size=32, num_workers=4, image_dim=256, k_folds=5
):
    train_transform, val_transform = get_transforms(image_dim)

    skf = StratifiedKFold(n_splits=k_folds)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(metadata_df, metadata_df["target"])
    ):
        train_metadata_df = metadata_df.iloc[train_idx]
        val_metadata_df = metadata_df.iloc[val_idx]

        train_dataset = SkinLesionDataset(
            hdf5_file_path=hdf5_file_path,
            metadata_df=train_metadata_df,
            target=train_metadata_df["target"],
            transform=train_transform,
        )

        val_dataset = SkinLesionDataset(
            hdf5_file_path=hdf5_file_path,
            metadata_df=val_metadata_df,
            target=val_metadata_df["target"],
            transform=val_transform,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        yield fold, train_dataloader, val_dataloader
