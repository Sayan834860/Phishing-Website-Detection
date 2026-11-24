import pandas as pd
from scipy.io import arff
import os

# Load Dataset
file_path = "dataset/Training Dataset.arff"

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convert byte values
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes) else x
        )

# Convert columns to numeric
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Original Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Remove duplicate records
df = df.drop_duplicates().copy()

print("\nDataset Shape After Removing Duplicates:", df.shape)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# Separate Features and Target
X = df.drop(columns=["Result"])
y = df["Result"]

print("\nInput Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Target distribution
print("\nTarget Distribution:")
print(y.value_counts())

# Save cleaned dataset
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/cleaned_dataset.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
print("PREPROCESSING COMPLETED!")
