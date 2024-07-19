import pandas as pd
import dotenv
import os

# Load the environment variables
dotenv.load_dotenv()

root_dir = os.getenv("ROOT_FOLDER")

# Load the metadata
train_metadata = pd.read_csv(root_dir + "/train-metadata.csv")
test_metadata = pd.read_csv(root_dir + "/test-metadata.csv")

# Display basic info
print(train_metadata.info())
print(train_metadata.head())
