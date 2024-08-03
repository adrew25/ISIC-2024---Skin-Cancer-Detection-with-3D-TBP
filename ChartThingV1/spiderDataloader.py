import numpy as np
import albumentations as A
from torch.utils.data import Dataset, Sampler
import h5py
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import torch
import time
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import os
import dotenv
import warnings

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

# Load and balance the dataset
dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")

train_metadata_path = os.path.join(dataset_dir, "train-metadata.csv")
train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class SkinLesionDataset(Dataset):
    def __init__(self, hdf5_file_path, metadata_df, target=None, transform=None):
        self.hdf5_file_path = hdf5_file_path
        self.metadata_df = metadata_df.fillna(0)
        self.image_ids = metadata_df["isic_id"].astype(str).tolist()
        self.labels = target.tolist() if target is not None else None
        self.transform = transform
        self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

        # Compute normalization parameters
        self.metadata = self._preprocess_metadata()
        self._compute_metadata_normalization_parameters()

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
                0,  # Placeholder for additional feature 8
                0,  # Placeholder for additional feature 9
            ]
            metadata.append(processed_row)
        return np.array(metadata, dtype=np.float32)

    def _compute_metadata_normalization_parameters(self):
        metadata_np = self.metadata
        self.log_metadata_min = np.min(metadata_np, axis=0)
        self.log_metadata_max = np.max(metadata_np, axis=0)

    def _log_normalize(self, value, min_val, max_val):
        return (
            (np.log1p(value - min_val) / np.log1p(max_val - min_val))
            if (max_val - min_val) != 0
            else 0
        )

    def _generate_radar_chart(self, metadata):
        if metadata.ndim != 1:
            raise ValueError("Metadata should be a 1D array or tensor")

        categories = [
            "tbp_lv_color_std_mean",
            "tbp_lv_radial_color_std_max",
            "tbp_lv_stdLExt",
            "tbp_lv_z",
            "tbp_lv_H",
            "tbp_lv_Hext",
            "tbp_lv_nevi_confidence",
            "tbp_lv_deltaB",
            "tbp_lv_Lext",
        ]

        if len(metadata) != len(categories):
            raise ValueError(
                f"Expected metadata length of {len(categories)}, but got {len(metadata)}"
            )

        values = [metadata[i] for i in range(len(categories))]

        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories, color="grey", size=8)
        ax.plot(angles, values, linewidth=1, linestyle="solid")
        ax.fill(angles, values, "b", alpha=0.1)
        ax.set_yticklabels([])

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)
        radar_image = np.array(Image.open(buf), dtype=np.float32) / 255.0

        if radar_image.ndim == 2:
            radar_image = np.stack([radar_image] * 3, axis=-1)
        elif radar_image.shape[2] == 4:
            radar_image = radar_image[:, :, :3]

        if radar_image.shape[2] != 3:
            raise ValueError(
                f"Expected 3 channels (RGB), but got {radar_image.shape[2]} channels"
            )

        return radar_image

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
        try:
            if self.hdf5_file is not None:
                self.hdf5_file.close()
        except AttributeError:
            pass


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
                ]
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

    radar_transform = Compose(
        [
            Resize(image_dim, image_dim),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    return train_transform, radar_transform, val_transform


def balance_dataset(metadata_df):
    target_column = "target"

    X = metadata_df.drop(columns=[target_column])
    y = metadata_df[target_column]

    ros = RandomOverSampler(sampling_strategy=0.01, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

    metadata_df_resampled = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.DataFrame(y_resampled, columns=[target_column]),
        ],
        axis=1,
    )

    return metadata_df_resampled


def create_folds_dataloaders(
    metadata_df,
    hdf5_file_path,
    batch_size=64,
    num_workers=1,
    image_dim=256,
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

        print(
            f"Samples with 0: {len(train_metadata_df_balanced[train_metadata_df_balanced['target'] == 0])}"
        )
        print(
            f"Samples with 1: {len(train_metadata_df_balanced[train_metadata_df_balanced['target'] == 1])} \n"
        )

        train_transform, val_transform, _ = get_transforms(image_dim)

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
            num_workers=num_workers,
            pin_memory=True,
            sampler=BalancedBatchSampler(
                train_metadata_df_balanced["target"], batch_size
            ),
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        folds.append((train_dataloader, val_dataloader))

    return folds


# batch_size = 2
# num_workers = 1
# image_dim = 256

# train_metadata_df = pd.read_csv(train_metadata_path, low_memory=False)

# folds = create_folds_dataloaders(
#     metadata_df=train_metadata_df,
#     hdf5_file_path=train_hdf5_path,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     image_dim=image_dim,
#     num_folds=2,
# )

# for i, (train_loader, val_loader) in enumerate(folds):
#     print(f"Fold {i+1}")

#     for images, metadata, labels in train_loader:
#         print("\nTrain Batch:")
#         print("Images:", images.shape, images.dtype)
#         print("Metadata:", metadata.shape, metadata.dtype)
#         print("Labels:", labels.shape, labels.dtype)

#         img_np = images[0].permute(1, 2, 0).numpy()
#         metadata_np = metadata[0].numpy()

#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#         axes[0].imshow(img_np)
#         axes[0].set_title("Image")
#         axes[0].axis("off")

#         radar_image = train_loader.dataset._generate_radar_chart(metadata_np)
#         axes[1].imshow(radar_image)
#         axes[1].set_title("Radar Chart")
#         axes[1].axis("off")

#         plt.savefig(f"fold_{i+1}_train_batch.png")
#         break

#     for images, metadata, labels in val_loader:
#         print("\nVal Batch:")
#         print("Images:", images.shape, images.dtype)
#         print("Metadata:", metadata.shape, metadata.dtype)
#         print("Labels:", labels.shape, labels.dtype)

#         img_np = images[0].permute(1, 2, 0).numpy()
#         metadata_np = metadata[0].numpy()

#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#         axes[0].imshow(img_np)
#         axes[0].set_title("Image")
#         axes[0].axis("off")

#         radar_image = val_loader.dataset._generate_radar_chart(metadata_np)
#         axes[1].imshow(radar_image)
#         axes[1].set_title("Radar Chart")
#         axes[1].axis("off")

#         plt.savefig(f"fold_{i+1}_val_batch.png")
#         break