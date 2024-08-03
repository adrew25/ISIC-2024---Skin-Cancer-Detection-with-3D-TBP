import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
import dotenv

dotenv.load_dotenv()


dataset_dir = os.getenv("ROOT_FOLDER")
if not dataset_dir:
    raise ValueError("ROOT_FOLDER environment variable is not set")

working_dir = os.getenv("WORKING_DIR")
if not working_dir:
    raise ValueError("WORKING_DIR environment variable is not set")

train_metadata_path = os.path.join(dataset_dir, "train-metadata.csv")
train_hdf5_path = os.path.join(dataset_dir, "train-image.hdf5")

train_metadata_df = pd.read_csv(train_metadata_path, low_memory=False)


def set_seed(seed):
    np.random.seed(seed)


set_seed(42)

metadata_columns = [
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


metadata_min = {}
metadata_max = {}

for column in metadata_columns:
    metadata_min[column] = train_metadata_df[column].min()
    metadata_max[column] = train_metadata_df[column].max()


def log_normalize(value, log_min_val, log_max_val):
    """Normalize a value to the range [0, 1] based on log min and max values."""
    log_value = np.log10(value if value > 0 else 1e-9)
    return (log_value - log_min_val) / (log_max_val - log_min_val)


def create_radar_chart(metadata, sample_id, log_metadata_min, log_metadata_max):
    """
    Create a radar chart for the given metadata using logarithmic scale.

    :param metadata: Dictionary containing metadata values.
    :param sample_id: ID of the sample.
    :param log_metadata_min: Dictionary containing log min values for metadata.
    :param log_metadata_max: Dictionary containing log max values for metadata.
    :return: Radar chart as a numpy array.
    """
    categories = list(metadata.keys())
    values = list(metadata.values())

    # Log normalize values
    normalized_values = [
        log_normalize(
            values[i], log_metadata_min[categories[i]], log_metadata_max[categories[i]]
        )
        for i in range(len(values))
    ]

    num_vars = len(categories)

    # Compute angle of each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    normalized_values += normalized_values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color="grey", size=8)
    ax.plot(angles, normalized_values, linewidth=1, linestyle="solid")
    ax.fill(angles, normalized_values, "b", alpha=0.1)
    ax.set_yticklabels([])

    plt.title(f"Sample {sample_id}", size=15, color="blue", y=1.1)
    plt.savefig(f"radar_charts/sample_{sample_id}.png")
    plt.close()


# Select a random sample ID
sample_id = 42

# Get metadata for the sample
metadata = train_metadata_df[metadata_columns].loc[sample_id].to_dict()

# Create a radar chart for the sample
create_radar_chart(metadata, sample_id, metadata_min, metadata_max)
