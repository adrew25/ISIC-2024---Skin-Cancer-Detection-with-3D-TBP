import os
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from albumentations.pytorch import ToTensorV2
import gc
import dotenv

dotenv.load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Kaggle Directories
# dataset_dir = "/kaggle/input/isic-2024-challenge/"
# working_dir = "/kaggle/input/train-isic-2024-skin-cancer-detection/"

dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")

# File paths
test_metadata_path = os.path.join(dataset_dir, "test-metadata.csv")
test_hdf5_path = os.path.join(dataset_dir, "test-image.hdf5")

train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")
train_image_path = os.path.join(dataset_dir, "train-image")


train_metadata_path = os.path.join(dataset_dir, "train-metadata.csv")

train_metadata_df = pd.read_csv(train_metadata_path, low_memory=False)


# Load test metadata
test_metadata_df = pd.read_csv(test_metadata_path, low_memory=False)


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


# Define image dimensions and transformations
dim = 50
test_transform = A.Compose(
    [
        A.Resize(height=dim, width=dim),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0
        ),
        ToTensorV2(),
    ]
)


train_dataset = SkinLesionDataset(
    test_hdf5_path, test_metadata_df, transform=test_transform
)
train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=False, num_workers=4
)  # Adjust batch size and num_workers


# Create test dataset and dataloader
test_dataset = SkinLesionDataset(
    test_hdf5_path, test_metadata_df, transform=test_transform
)
test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4
)  # Adjust batch size and num_workers

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)


def generate_predictions(model, dataloader, device):
    model.eval()
    sigmoid = nn.Sigmoid()

    with open("submission.csv", "w") as f:
        f.write("isic_id,target\n")
        with torch.no_grad():
            for inputs, batch_ids in tqdm(dataloader, desc="Generating Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = sigmoid(outputs).cpu().numpy().flatten()

                for id, output in zip(batch_ids, outputs):
                    output = float(output)
                    output = min(
                        max(output, 0), 1
                    )  # Ensure predictions are within [0, 1]
                    f.write(f"{id},{output}\n")

                # Clean up memory
                del inputs, outputs, batch_ids
                gc.collect()


# Load the pre-trained model
load_model_pth = os.path.join(working_dir, "Models/ResNet50_best.pth")
model.load_state_dict(torch.load(load_model_pth, map_location=device))
model.to(device)

# Generate predictions and save to submission.csv
print("Making predictions...")
# generate_predictions(model, test_loader, device)
generate_predictions(model, train_loader, device)

print("Predictions completed and saved to submission.csv")

# Clean up memory
del model
gc.collect()
