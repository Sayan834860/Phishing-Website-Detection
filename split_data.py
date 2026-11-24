import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

# Load Cleaned Dataset
df = pd.read_csv("dataset/cleaned_dataset.csv")

# Separate Features and Target
X = df.drop(columns=["Result"])
y = df["Result"]

# Split Dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display Dataset Shapes
print("========== TRAIN TEST SPLIT ==========")

print("\nOriginal Dataset:", X.shape)
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target:", y_train.shape)
print("Testing Target:", y_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

# Save Split Data
os.makedirs("models", exist_ok=True)

split_data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}

joblib.dump(split_data, "models/split_data.pkl")

print("\nTrain-Test Data Saved Successfully!")
print("STEP 5 COMPLETED!")
