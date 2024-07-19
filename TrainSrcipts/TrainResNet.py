import os
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import dotenv
import sklearn.model_selection as train_test_split
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
import gc

dotenv.load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")

train_metadata_path = os.path.join(dataset_dir, "train-metadata.csv")
test_metadata_path = os.path.join(dataset_dir, "test-metadata.csv")

train_metadata_df = pd.read_csv(train_metadata_path, low_memory=False)
test_metadata_df = pd.read_csv(test_metadata_path, low_memory=False)

print(len(train_metadata_df))

train_size = 0.8
# Splitting the train dataset into positive and negative samples and saving them in separate dataframes
positive_samples = train_metadata_df[train_metadata_df["target"] == 1]
negative_samples = train_metadata_df[train_metadata_df["target"] == 0]

# TODO: Taking Sample of 1% of the data
positive_samples = positive_samples.sample(frac=0.01)
negative_samples = negative_samples.sample(frac=0.01)

print(
    f"Positive samples: {positive_samples.shape} \n"
    f"Negative samples: {negative_samples.shape}"
)

# Splitting each type of samples into train and validation sets
train_positive_samples, val_positive_samples = train_test_split.train_test_split(
    positive_samples, test_size=1 - train_size
)

train_negative_samples, val_negative_samples = train_test_split.train_test_split(
    negative_samples, test_size=1 - train_size
)
print(f"Train positive samples: {train_positive_samples.shape}")
print(f"Train negative samples: {train_negative_samples.shape}")
print(f"Val positive samples: {val_positive_samples.shape}")
print(f"Val negative samples: {val_negative_samples.shape}")

# Concatenating the positive and negative samples to get the train and validation sets
train_metadata_df = pd.concat([train_positive_samples, train_negative_samples])
val_metadata_df = pd.concat([val_positive_samples, val_negative_samples])

print(f"Train samples: {train_metadata_df.shape}")
print(f"Validation samples: {val_metadata_df.shape}")
print(f"Test samples: {test_metadata_df.shape}")

train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")
test_hdf5_path = os.path.join(dataset_dir, "test-image.hdf5")
train_image_path = os.path.join(dataset_dir, "train-image")


class SkinLesionDataset(Dataset):
    def __init__(self, hdf5_file_path, metadata_df, target=None, transform=None):
        self.hdf5_file_path = hdf5_file_path
        self.metadata_df = metadata_df
        self.images_ids = metadata_df["isic_id"]
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


dim = 150

train_transform = A.Compose(
    [
        A.Resize(height=dim, width=dim),  # resize
        A.OneOf(
            [
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0
        ),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=dim, width=dim),  # resize
        A.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0
        ),
        ToTensorV2(),
    ]
)

train_dataset = SkinLesionDataset(
    train_hdf5_path,
    train_metadata_df,
    target=train_metadata_df["target"],
    transform=train_transform,
)

val_dataset = SkinLesionDataset(
    train_hdf5_path,
    val_metadata_df,
    target=val_metadata_df["target"],
    transform=train_transform,
)

test_dataset = SkinLesionDataset(
    test_hdf5_path, test_metadata_df, transform=test_transform
)

BATCH_SIZE = 128  # Increased batch size

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


def calculate_partial_auc_by_tpr(y_true, y_scores, max_tpr=0.8):
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the index where TPR is closest to max_tpr
    idx = np.where(tpr >= max_tpr)[0]

    # Check if the index is empty
    if len(idx) == 0:
        return 0.0

    # Calculate partial AUC up to the specified TPR
    idx = idx[0]
    partial_auc = auc(fpr[: idx + 1], tpr[: idx + 1])

    # Normalize the partial AUC to the range [0, 1]
    partial_auc /= max_tpr

    return partial_auc


model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # For mixed precision training


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    model.train()
    best_auc = 0.0  # Initialize the best AUC

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Wrap dataloader with tqdm for progress bar
        tqdm_loader = tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs - 1}", leave=False
        )

        for inputs, labels in tqdm_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            with autocast():  # Mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

            # Update tqdm progress bar
            tqdm_loader.set_postfix(loss=running_loss / len(train_loader.dataset))

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

        # Evaluate the model
        auc_score = evaluate_model(model, val_loader)
        print(f"Validation Partial AUC: {auc_score:.4f}")

        # Check if the current AUC is the best we have seen so far
        if auc_score > best_auc:
            best_auc = auc_score
            # Save the model with the best AUC
            torch.save(
                model.state_dict(),
                os.path.join(working_dir, "Models/ResNet50_best.pth"),
            )
            print(f"New best model saved with AUC: {best_auc:.4f}")

    return model


def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # Calculate Partial AUC
    partial_auc = calculate_partial_auc_by_tpr(
        np.array(all_labels), np.array(all_outputs)
    )
    print(f"Partial AUC (up to 80% TPR): {partial_auc:.4f}")

    return partial_auc


# Train the model
model = train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=25,
)
print("Model Trained Successfully")

# Clean up memory
del model
gc.collect()
torch.cuda.empty_cache()
