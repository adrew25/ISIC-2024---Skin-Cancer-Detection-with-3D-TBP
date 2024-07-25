import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import gc
import dotenv
from sklearn.utils.class_weight import compute_class_weight
import timm

from Dataloader import create_dataloaders
from utils.utils import pauc, save_best_model

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
batch_size = 128  # Reduce batch size to reduce memory usage
k_folds = 5
best_val_loss = float("inf")

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
                loss = criterion(outputs, labels.unsqueeze(1).float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataloader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_true = []
        val_pred = []
        with torch.no_grad():
            for images, labels in tqdm(
                val_dataloader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"
            ):
                images, labels = images.to(device), labels.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    val_loss += loss.item() * images.size(0)

                val_true.extend(labels.cpu().numpy())
                val_pred.extend(outputs.cpu().numpy())

        val_loss /= len(val_dataloader.dataset)
        partial_auc = pauc(val_true, val_pred, min_tpr=0.80)

        print(f"Epoch {epoch + 1}/{num_epochs}, Fold {fold + 1}/{k_folds}")
        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Partial AUC: {partial_auc:.4f}"
        )

        # Save best model based on validation loss
        best_val_loss = save_best_model(
            model, fold + 1, epoch + 1, val_loss, working_dir, best_val_loss
        )

        # Clear cache to free memory
        torch.cuda.empty_cache()
        gc.collect()
