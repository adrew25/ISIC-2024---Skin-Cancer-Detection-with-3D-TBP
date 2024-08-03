import os

import timm
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim

from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import dotenv
import warnings

import albumentations as A
from torch.utils.data import Dataset, Sampler

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import time
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")
dotenv.load_dotenv()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# Load and balance the dataset
dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")

train_metadata_path = os.path.join(dataset_dir, "train-metadata.csv")
train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")
train_metadata_df = pd.read_csv(train_metadata_path, low_memory=False)


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

    ros = RandomOverSampler(sampling_strategy=0.002, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    rus = RandomUnderSampler(sampling_strategy=0.01, random_state=42)
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


# Load and balance the dataset
dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")

train_metadata_path = os.path.join(dataset_dir, "train-metadata.csv")
train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")


def pauc(true, pred, min_tpr: float = 0.80) -> float:
    v_gt = abs(np.asarray(true) - 1)
    v_pred = -1.0 * np.asarray(pred)
    max_fpr = abs(1 - min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)

    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc


# Save the best model based on validation loss and partial AUC
def save_best_model(model, model_name, fold, epoch, val_loss, partial_auc, save_dir):
    model_file = f"{model_name}_fold{fold}_epoch{epoch}_valLoss{val_loss:.4f}_partialAUC{partial_auc:.4f}.pth"
    model_path = os.path.join(save_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return val_loss, partial_auc


# Model definition with ResNet18
class CombinedModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes=1):
        super(CombinedModel, self).__init__()
        self.resnet_image = timm.create_model("resnet18.a1_in1k", pretrained=True)
        self.resnet_image.fc = nn.Identity()  # Remove the final fully connected layer

        # Simple FC network for radar images (metadata)
        self.fc_radar = nn.Sequential(
            nn.Linear(num_metadata_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Combined fully connected layer
        self.fc = nn.Linear(self.resnet_image.num_features + 128, num_classes)

    def forward(self, skin_images, radar_data):
        image_features = self.resnet_image(skin_images)
        radar_features = self.fc_radar(radar_data)
        combined_features = torch.cat((image_features, radar_features), dim=1)
        output = self.fc(combined_features)
        return output


# Training loop
def train_model(
    folds, num_epochs, model, criterion, optimizer, device, save_dir, model_name
):
    for fold, (train_loader, val_loader) in enumerate(folds):
        print(f"Fold {fold+1}")
        best_val_loss = float("inf")
        best_partial_auc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Training phase
            model.train()
            running_loss = 0.0
            train_true, train_pred = [], []

            for skin_images, radar_data, labels in tqdm(train_loader, desc="Training"):
                skin_images = skin_images.to(device)
                radar_data = radar_data.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()
                outputs = model(skin_images, radar_data).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * skin_images.size(0)
                train_true.extend(labels.cpu().numpy())
                train_pred.extend(
                    torch.sigmoid(outputs).detach().cpu().numpy()
                )  # Detach before converting to numpy

            train_loss = running_loss / len(train_loader.dataset)
            train_accuracy = accuracy_score(
                train_true, (np.array(train_pred) > 0.5).astype(int)
            )
            train_pauc = pauc(train_true, train_pred)

            print(
                f"Train Loss: {train_loss:.4f} Accuracy: {train_accuracy:.4f} pAUC: {train_pauc:.4f}"
            )

            # Validation phase
            model.eval()
            running_loss = 0.0
            val_true, val_pred = [], []

            with torch.no_grad():
                for skin_images, radar_data, labels in tqdm(
                    val_loader, desc="Validation"
                ):
                    skin_images = skin_images.to(device)
                    radar_data = radar_data.to(device)
                    labels = labels.to(device).float()

                    outputs = model(skin_images, radar_data).squeeze()
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * skin_images.size(0)
                    val_true.extend(labels.cpu().numpy())
                    val_pred.extend(
                        torch.sigmoid(outputs).detach().cpu().numpy()
                    )  # Detach before converting to numpy

            val_loss = running_loss / len(val_loader.dataset)
            val_accuracy = accuracy_score(
                val_true, (np.array(val_pred) > 0.5).astype(int)
            )
            val_pauc = pauc(val_true, val_pred)

            print(
                f"Val Loss: {val_loss:.4f} Accuracy: {val_accuracy:.4f} pAUC: {val_pauc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss, best_partial_auc = save_best_model(
                    model, model_name, fold, epoch, val_loss, val_pauc, save_dir
                )


# Example usage
num_epochs = 10  # Set the number of epochs you want to train for
num_metadata_features = 9  # Set the number of metadata features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel(num_metadata_features=num_metadata_features, num_classes=1).to(
    device
)
criterion = nn.BCEWithLogitsLoss()  # Binary classification

optimizer = optim.Adam(model.parameters(), lr=0.0001)
folds = create_folds_dataloaders(
    train_metadata_df,
    train_hdf5_path,
    batch_size=64,
    num_workers=2,
    image_dim=256,
    num_folds=5,
)

save_dir = os.path.join(working_dir, "Models")
model_name = "combined_ResNet18"

train_model(
    folds, num_epochs, model, criterion, optimizer, device, save_dir, model_name
)
