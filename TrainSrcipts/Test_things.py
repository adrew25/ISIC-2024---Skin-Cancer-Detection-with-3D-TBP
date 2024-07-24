import os
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import dotenv
import sklearn.model_selection as train_test_split
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
import gc
import random
from torch.utils.data import Sampler
from matplotlib import pyplot as plt
import timm  # from efficientnet_pytorch import EfficientNet
from sklearn.utils.class_weight import compute_class_weight

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


# Splitting each type of samples into train and validation sets
train_positive_samples, val_positive_samples = train_test_split.train_test_split(
    positive_samples, test_size=1 - train_size
)

train_negative_samples, val_negative_samples = train_test_split.train_test_split(
    negative_samples, test_size=1 - train_size
)


# Concatenating the positive and negative samples to get the train and validation sets
train_metadata_df = pd.concat([train_positive_samples, train_negative_samples])
val_metadata_df = pd.concat([val_positive_samples, val_negative_samples])

# Show me how many train and validation samples we have
print(f"Train samples: {len(train_metadata_df)}")
print(f"Validation samples: {len(val_metadata_df)}")

# also print train positive and negative samples
print(f"Train positive samples: {len(train_positive_samples)}")
print(f"Train negative samples: {len(train_negative_samples)} \n")

print(f"Val positive samples: {len(val_positive_samples)}")
print(f"Val negative samples: {len(val_negative_samples)} \n")

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


image_size = 256

train_transform = A.Compose(
    [
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        A.RandomGamma(gamma_limit=(80, 120), p=0.75),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ],
            p=0.7,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.0),
                A.ElasticTransform(alpha=3),
            ],
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85
        ),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

transforms_val = A.Compose([A.Resize(image_size, image_size), A.Normalize()])

val_transform = A.Compose(
    [
        A.Resize(image_size, image_size),
        # A.PadIfNeeded(min_height=image_dim, min_width=image_dim),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


train_dataset = SkinLesionDataset(
    hdf5_file_path=train_hdf5_path,
    metadata_df=train_metadata_df,
    target=train_metadata_df["target"],
    transform=train_transform,
)

val_dataset = SkinLesionDataset(
    hdf5_file_path=train_hdf5_path,
    metadata_df=val_metadata_df,
    target=val_metadata_df["target"],
    transform=val_transform,
)


class BalancedBatchSampler(Sampler):
    def __init__(self, pos_samples, neg_samples, batch_size):
        self.batch_size = batch_size
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.num_batches = max(self.num_pos, self.num_neg) // (batch_size // 2)

    def __iter__(self):
        pos_indices = np.arange(self.num_pos)
        neg_indices = np.arange(self.num_neg)
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        for batch_idx in range(self.num_batches):
            start_idx_pos = batch_idx * (self.batch_size // 2)
            end_idx_pos = start_idx_pos + (self.batch_size // 2)
            start_idx_neg = batch_idx * (self.batch_size // 2)
            end_idx_neg = start_idx_neg + (self.batch_size // 2)

            batch_pos = pos_indices[start_idx_pos:end_idx_pos]
            batch_neg = neg_indices[start_idx_neg:end_idx_neg]

            batch_indices = np.concatenate([batch_pos, batch_neg + self.num_pos])
            np.random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches


# Create the train and validation dataloaders with the custom sampler
train_batch_size = 32  # This should be an even number

train_sampler = BalancedBatchSampler(
    train_positive_samples, train_negative_samples, train_batch_size
)
val_sampler = BalancedBatchSampler(
    val_positive_samples, val_negative_samples, train_batch_size
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    sampler=None,
    num_workers=4,
    pin_memory=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=train_batch_size,
    sampler=None,
    num_workers=4,
    pin_memory=True,
)


# show me from train_dataloader the first batch of targets
# how much is positive and how much is negative
# print the first batch of targets
for inputs, labels in train_dataloader:
    print(labels)
    break


# Model, loss, and optimizer
model = timm.create_model("efficientnet_b0", pretrained=True)
model_name = "efficientnet_b0"
model.classifier = nn.Linear(model.classifier.in_features, 1)  # Adjust the final layer
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
scaler = GradScaler()


def pauc(true, pred, min_tpr: float = 0.80) -> float:
    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(true) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(pred)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc


# Calculate class weights
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_metadata_df["target"]),
    y=train_metadata_df["target"],
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Update the criterion with class weights
criterion = nn.BCEWithLogitsLoss(
    pos_weight=class_weights[1]
)  # Weight the positive class more


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())  # Cast labels to float

            running_loss += loss.item() * inputs.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc_score = roc_auc_score(all_labels, all_preds)
    pauc_score = pauc(all_labels, all_preds)

    return running_loss / len(dataloader.dataset), auc_score, pauc_score


# Training the model
def train_model(
    model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs
):
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())  # Cast labels to float
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_dataloader.dataset)
        val_loss, val_auc, val_pauc = evaluate_model(model, val_dataloader, criterion)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val PAUC: {val_pauc:.4f}"
        )

        # if the Val AUC is better than the best Val AUC, save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(
                model.state_dict(),
                os.path.join(working_dir, f"{model_name}_{val_auc:.4f}.pth"),
            )

    return best_model


best_model = train_model(
    model, train_dataloader, val_dataloader, criterion, optimizer, 10
)

print("Training complete")
