
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.io import arff

# Load Dataset
file_path = "dataset/Training Dataset.arff"
data, meta = arff.loadarff(file_path)

df = pd.DataFrame(data)

# Convert byte values into strings
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes) else x
        )

# Convert all columns into numeric format
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Create folder for graphs
os.makedirs("visualizations", exist_ok=True)

# Set graph style
sns.set_theme(style="whitegrid")

# 1. Target Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Result")
plt.title("Phishing vs Legitimate Websites")
plt.xlabel("Website Class")
plt.ylabel("Number of Websites")
plt.savefig("visualizations/class_distribution.png")
plt.show()
plt.close()

# 2. Feature Distribution
features = df.drop(columns=["Result"])

features.hist(
    figsize=(18, 14),
    bins=20
)

plt.suptitle("Feature Distribution Analysis")
plt.tight_layout()
plt.savefig("visualizations/feature_distribution.png")
plt.show()
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(18, 12))

correlation = df.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    linewidths=0.2
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("\nGraphs saved in visualizations folder.")
