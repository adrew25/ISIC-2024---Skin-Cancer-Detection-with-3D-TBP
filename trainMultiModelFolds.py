import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import dotenv
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataloader import create__folds_dataloaders
from utils.utils import pauc, save_best_model
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


# MultiModel class for combining image and metadata
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


# Define the loss function, optimizer, and scaler

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.00015, weight_decay=0.01
)  # Added weight decay

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.1, verbose=True)

scaler = GradScaler()

num_epochs = 10
batch_size = 64


def train_and_validate(folds, model, criterion, optimizer, device, num_epochs=10):
    fold_auc_scores = []

    for fold, (train_dataloader, val_dataloader) in enumerate(folds):
        print(f"Fold {fold+1}/{len(folds)}")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for images, metadata, labels in tqdm(
                train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"
            ):
                images, metadata, labels = (
                    images.to(device),
                    metadata.to(device),
                    labels.to(device),
                )
                optimizer.zero_grad()
                outputs = model(images, metadata)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_dataloader.dataset)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_true = []
            val_pred = []
            with torch.no_grad():
                for images, metadata, labels in tqdm(
                    val_dataloader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"
                ):
                    images, metadata, labels = (
                        images.to(device),
                        metadata.to(device),
                        labels.to(device),
                    )
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    sigmoid_outputs = torch.sigmoid(outputs)
                    val_loss += loss.item() * images.size(0)
                    val_true.extend(labels.cpu().numpy())
                    val_pred.extend(sigmoid_outputs.cpu().numpy())

            val_true = np.array(val_true)
            val_pred = np.array(val_pred)
            val_loss /= len(val_dataloader.dataset)
            partial_auc = pauc(val_true, val_pred, min_tpr=0.80)
            fold_auc_scores.append(partial_auc)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Partial AUC: {partial_auc:.4f}"
            )

            save_best_model(
                model, "MultiModel", fold + 1, epoch, val_loss, partial_auc, working_dir
            )

    print(f"Cross-Validation Partial AUC scores: {fold_auc_scores}")
    print(f"Mean Partial AUC: {np.mean(fold_auc_scores)}")


folds = create__folds_dataloaders(
    train_metadata_df, train_hdf5_path, batch_size=batch_size, num_folds=5
)

train_and_validate(folds, model, criterion, optimizer, device, num_epochs=num_epochs)
