import numpy as np
import pandas as pd
import h5py
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from io import BytesIO
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import time

import warnings

warnings.filterwarnings("ignore")


# Create Random Seed for Reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# for i in range(len(df)):
#     x.append(df.isic_id[i])
#     y.append(df.target[i])
#     patient_ids.append(df.patient_id[i])
#     with np.errstate(divide="ignore", invalid="ignore"):
#         color_uniformity = np.divide(
#             df.tbp_lv_color_std_mean[i], df.tbp_lv_radial_color_std_max[i]
#         )
#         if np.isscalar(color_uniformity):
#             if not np.isfinite(color_uniformity):
#                 color_uniformity = 0
#         else:
#             color_uniformity[~np.isfinite(color_uniformity)] = 0
#     metadata[x[-1]] = {
#         "color_uniformity": color_uniformity,
#         "tbp_lv_stdLExt": df.tbp_lv_stdLExt[i],
#         "tbp_lv_z": df.tbp_lv_z[i],
#         "hue_contrast": np.abs(df.tbp_lv_H[i] - df.tbp_lv_Hext[i]),
#         "tbp_lv_nevi_confidence": df.tbp_lv_nevi_confidence[i],
#         "tbp_lv_deltaB": df.tbp_lv_deltaB[i],
#         "tbp_lv_stdLExt": df.tbp_lv_stdLExt[i],
#     }


# ## dataloader


# color_uniformity = self.metadata[sample]["color_uniformity"]
# color_uniformity = torch.from_numpy(
#     np.array(color_uniformity, dtype=np.float32)
# ).unsqueeze(0)
# tbp_lv_stdLExt = self.metadata[sample]["tbp_lv_stdLExt"]
# tbp_lv_stdLExt = torch.from_numpy(np.array(tbp_lv_stdLExt, dtype=np.float32)).unsqueeze(
#     0
# )
# tbp_lv_z = self.metadata[sample]["tbp_lv_z"]
# tbp_lv_z = torch.from_numpy(np.array(tbp_lv_z, dtype=np.float32)).unsqueeze(0)
# hue_contrast = self.metadata[sample]["hue_contrast"]
# hue_contrast = torch.from_numpy(np.array(hue_contrast, dtype=np.float32)).unsqueeze(0)
# tbp_lv_nevi_confidence = self.metadata[sample]["tbp_lv_nevi_confidence"]
# tbp_lv_nevi_confidence = torch.from_numpy(
#     np.array(tbp_lv_nevi_confidence, dtype=np.float32)
# ).unsqueeze(0)
# tbp_lv_deltaB = self.metadata[sample]["tbp_lv_deltaB"]
# tbp_lv_deltaB = torch.from_numpy(np.array(tbp_lv_deltaB, dtype=np.float32)).unsqueeze(0)
# tbp_lv_stdLExt = self.metadata[sample]["tbp_lv_stdLExt"]
# tbp_lv_stdLExt = torch.from_numpy(np.array(tbp_lv_stdLExt, dtype=np.float32)).unsqueeze(
#     0
# )
class SkinLesionDataset(Dataset):
    def __init__(self, hdf5_file_path, metadata_df, target=None, transform=None):
        self.hdf5_file_path = hdf5_file_path
        self.metadata_df = metadata_df.fillna(0)  # Fill NaN values with 0
        self.image_ids = metadata_df["isic_id"].astype(str).tolist()
        self.labels = target.tolist() if target is not None else None
        self.transform = transform
        self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

        # Preprocess metadata and store in memory
        self.metadata = self._preprocess_metadata()

    def _preprocess_metadata(self):
        metadata = []
        for i in range(len(self.metadata_df)):
            row = self.metadata_df.iloc[i]
            color_uniformity = np.divide(
                row.tbp_lv_color_std_mean,
                row.tbp_lv_radial_color_std_max,
                out=np.zeros_like(row.tbp_lv_color_std_mean),
                where=row.tbp_lv_radial_color_std_max != 0,
            ).mean()  # Taking the mean to reduce to a single value

            # Replace NaN values with a default value (e.g., 0)
            processed_row = [
                color_uniformity,
                row.tbp_lv_stdLExt if not np.isnan(row.tbp_lv_stdLExt) else 0,
                row.tbp_lv_z if not np.isnan(row.tbp_lv_z) else 0,
                np.abs(row.tbp_lv_H - row.tbp_lv_Hext)
                if not np.isnan(row.tbp_lv_H - row.tbp_lv_Hext)
                else 0,
                row.tbp_lv_nevi_confidence
                if not np.isnan(row.tbp_lv_nevi_confidence)
                else 0,
                row.tbp_lv_deltaB if not np.isnan(row.tbp_lv_deltaB) else 0,
                row.tbp_lv_Lext if not np.isnan(row.tbp_lv_Lext) else 0,
            ]

            metadata.append(processed_row)

        return torch.tensor(metadata, dtype=torch.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = idx[0]

        image_id = self.image_ids[idx]
        image = (
            np.array(
                Image.open(BytesIO(self.hdf5_file[image_id][()])), dtype=np.float32
            )
            / 255.0
        )

        if self.transform:
            image = self.transform(image=image)["image"]

        metadata = self.metadata[idx]

        if self.labels is not None:
            label = self.labels[idx]
            return image, metadata, label
        else:
            return image, metadata, image_id

    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        self.num_batches = min(len(self.pos_indices), len(self.neg_indices)) // (
            batch_size // 2
        )

    def __iter__(self):
        pos_indices = np.random.permutation(self.pos_indices)
        neg_indices = np.random.permutation(self.neg_indices)
        pos_batches = np.array_split(pos_indices, self.num_batches)
        neg_batches = np.array_split(neg_indices, self.num_batches)

        for pos_batch, neg_batch in zip(pos_batches, neg_batches):
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


def get_transforms(image_dim):
    train_transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.75
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ],
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1),
                    A.ElasticTransform(alpha=3),
                ]
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7
            ),
            A.Rotate(limit=30),
            A.Resize(image_dim, image_dim),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(image_dim, image_dim),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def balance_dataset(metadata_df):
    # Randomly oversample the minority class
    ros = RandomOverSampler(sampling_strategy=0.01, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(
        metadata_df.drop(columns=["target"]), metadata_df["target"]
    )

    # Randomly undersample the majority class
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

    # Combine the resampled data into a DataFrame
    metadata_df_resampled = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=metadata_df.columns.drop("target")),
            pd.DataFrame(y_resampled, columns=["target"]),
        ],
        axis=1,
    )

    return metadata_df_resampled


def create_dataloaders(
    metadata_df,
    hdf5_file_path,
    batch_size=64,
    num_workers=1,
    image_dim=256,
    test_size=0.2,
):
    train_transform, val_transform = get_transforms(image_dim)

    # Use GroupShuffleSplit to create the train and validation sets based on patient_id
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(gss.split(metadata_df, groups=metadata_df["patient_id"]))
    train_metadata_df = metadata_df.iloc[train_idx]
    val_metadata_df = metadata_df.iloc[val_idx]

    print(f"Train samples: {len(train_metadata_df)}")
    print(f"Validation samples: {len(val_metadata_df)}")

    print(f"Samples with 0: {len(train_metadata_df[train_metadata_df['target'] == 0])}")
    print(f"Samples with 1: {len(train_metadata_df[train_metadata_df['target'] == 1])}")

    print("\n Balancing the training dataset... \n")

    time_before = time.time()
    # Balance the training dataset
    train_metadata_df_balanced = balance_dataset(train_metadata_df)
    time_after = time.time() - time_before

    print(f"Time taken to balance the dataset: {time_after} seconds \n")
    # Print the number of samples in each class
    print(
        f"Samples with 0: {len(train_metadata_df_balanced[train_metadata_df_balanced['target'] == 0])}"
    )
    print(
        f"Samples with 1: {len(train_metadata_df_balanced[train_metadata_df_balanced['target'] == 1])}"
    )

    train_dataset = SkinLesionDataset(
        hdf5_file_path=hdf5_file_path,
        metadata_df=train_metadata_df_balanced,
        target=train_metadata_df_balanced["target"],
        transform=train_transform,
    )

    val_dataset = SkinLesionDataset(
        hdf5_file_path=hdf5_file_path,
        metadata_df=val_metadata_df,
        target=val_metadata_df["target"],
        transform=val_transform,
    )

    # Create the custom balanced batch sampler for the training data
    train_labels = train_metadata_df_balanced["target"].values
    train_sampler = BalancedBatchSampler(train_labels, batch_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=2,
        pin_memory=False,
    )

    return train_dataloader, val_dataloader


def create__folds_dataloaders(
    metadata_df,
    hdf5_file_path,
    batch_size=64,
    num_workers=1,
    image_dim=156,
    num_folds=5,
):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    X = metadata_df.drop(columns=["target"]).values
    y = metadata_df["target"].values

    folds = []

    for train_index, val_index in skf.split(X, y):
        train_metadata_df = metadata_df.iloc[train_index]
        val_metadata_df = metadata_df.iloc[val_index]

        print(f"Train samples: {len(train_metadata_df)}")
        print(f"Validation samples: {len(val_metadata_df)}")
        print(f"Train samples: {len(train_metadata_df)}")
        print(f"Validation samples: {len(val_metadata_df)}")

        print(
            f"Samples with 0: {len(train_metadata_df[train_metadata_df['target'] == 0])}"
        )
        print(
            f"Samples with 1: {len(train_metadata_df[train_metadata_df['target'] == 1])}"
        )

        print("\n Balancing the training dataset... \n")

        time_before = time.time()
        train_metadata_df_balanced = balance_dataset(train_metadata_df)
        time_after = time.time() - time_before
        print(f"Time taken to balance the dataset: {time_after} seconds \n")

        # Print the number of samples in each class
        print(
            f"Samples with 0: {len(train_metadata_df_balanced[train_metadata_df_balanced['target'] == 0])}"
        )
        print(
            f"Samples with 1: {len(train_metadata_df_balanced[train_metadata_df_balanced['target'] == 1])} \n"
        )

        train_transform, val_transform = get_transforms(image_dim)

        train_dataset = SkinLesionDataset(
            hdf5_file_path=hdf5_file_path,
            metadata_df=train_metadata_df_balanced,
            target=train_metadata_df_balanced["target"],
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
            num_workers=2,
            pin_memory=False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        )

        folds.append((train_dataloader, val_dataloader))

    return folds
