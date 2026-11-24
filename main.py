
import pandas as pd
from scipy.io import arff

# Load ARFF dataset
file_path = "dataset/Training Dataset.arff"

data, meta = arff.loadarff(file_path)

# Convert dataset into DataFrame
df = pd.DataFrame(data)

# Convert byte values into strings
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

# Dataset overview
print("\n========== DATASET INFORMATION ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Class Distribution:")
print(df["Result"].value_counts())

print("\n========== DATASET LOADED SUCCESSFULLY ==========")
