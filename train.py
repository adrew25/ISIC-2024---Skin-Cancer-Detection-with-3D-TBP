import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import gc
import dotenv
from sklearn.utils.class_weight import compute_class_weight
import timm

from Dataloader import create_dataloaders
from utils.utils import calculate_partial_auc_by_tpr, save_best_model

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
train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")

train_metadata_df = pd.read_csv(train_metadata_path, low_memory=False)

# Define the model
model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model.to(device)

# Define criterion and optimizer
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_metadata_df["target"]),
    y=train_metadata_df["target"],
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

num_epochs = 10
batch_size = 32
k_folds = 5
best_auc = 0.0

for fold, train_dataloader, val_dataloader in create_dataloaders(
    train_metadata_df, train_hdf5_path, batch_size=batch_size, k_folds=k_folds
):
    print(f"Training fold {fold + 1}/{k_folds}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(
            train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        partial_auc = calculate_partial_auc_by_tpr(all_labels, all_outputs)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_dataloader)}, "
            f"Validation Loss: {val_loss / len(val_dataloader)}, Partial AUC: {partial_auc}"
        )

        best_auc = save_best_model(
            model,
            epoch,
            partial_auc,
            best_auc,
            f"{working_dir}/best_model_fold_{fold + 1}",
        )

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
