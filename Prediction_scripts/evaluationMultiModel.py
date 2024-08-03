import os
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import timm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
from albumentations.pytorch import ToTensorV2
import dotenv

dotenv.load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Kaggle Directories
# dataset_dir = "/kaggle/input/isic-2024-challenge/"
# working_dir = "/kaggle/input/train-isic-2024-skin-cancer-detection/"

# Lodal test metadata
dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")


# File paths
test_metadata_path = os.path.join(dataset_dir, "test-metadata.csv")
test_hdf5_path = os.path.join(dataset_dir, "test-image.hdf5")

# Load test metadata
test_metadata_df = pd.read_csv(test_metadata_path, low_memory=False)


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


# Define image dimensions and transformations
image_dim = 156
test_transform = A.Compose(
    [
        A.Resize(image_dim, image_dim),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

# Create test dataset and dataloader
test_dataset = SkinLesionDataset(
    test_hdf5_path, test_metadata_df, transform=test_transform
)
test_loader = DataLoader(
    test_dataset, batch_size=512, shuffle=False, num_workers=4
)  # Adjust batch size and num_workers


class MultiModel(nn.Module):
    def __init__(self, image_model, metadata_input_dim):
        super(MultiModel, self).__init__()
        self.image_model = image_model
        self.metadata_fc = nn.Linear(metadata_input_dim, 128)
        self.linear1 = nn.Linear(image_model.num_features + 128, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Increased Dropout to 0.5
        self.final_fc = nn.Linear(256, 1)

    def forward(self, image, metadata):
        image_features = self.image_model(image)
        metadata_features = F.relu(self.metadata_fc(metadata))
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        x = self.linear1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.final_fc(x)
        return output


# Define the model
image_model = timm.create_model("efficientnet_b0", pretrained=True)
num_features = image_model.classifier.in_features
image_model.reset_classifier(0)
image_model.num_features = num_features  # Add this line
model = MultiModel(image_model, metadata_input_dim=7)
model = model.to(device)


def evaluate_model(model, test_loader, device):
    preds = []
    ids = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, metadata, image_ids) in enumerate(test_loader):
            images = images.to(device)
            metadata = metadata.to(device)
            with torch.cuda.amp.autocast():
                model_outputs = model(images, metadata)
            sigm_model_outputs = torch.sigmoid(model_outputs.squeeze(-1))
            preds.extend(sigm_model_outputs.cpu().numpy())
            ids.extend(image_ids)

    return preds, ids


# Load the pre-trained model
load_model_pth = os.path.join(
    working_dir, "Models/multimodel_fold1_epoch9_valLoss0.2803_partialAUC0.1589.pth"
)
model.load_state_dict(torch.load(load_model_pth, map_location=device))
model.to(device)

# Generate predictions and save to submission.csv
print("Making predictions...")

preds, ids = evaluate_model(model, test_loader, device)

df_sub = pd.read_csv(os.path.join(dataset_dir, "sample_submission.csv"))
# df_sub = pd.read_csv("/kaggle/input/isic-2024-challenge/sample_submission.csv")
df_sub["target"] = preds
df_sub["isic_id"] = ids
df_sub.to_csv("submission.csv", index=False)
