import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Training and Testing Data
data = joblib.load("models/split_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Define Three Machine Learning Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# Store Results
results = []
trained_models = {}

print("========== MODEL TRAINING ==========")

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train Model
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)

    # -1 represents Phishing
    precision = precision_score(
        y_test, y_pred, pos_label=-1, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=-1, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=-1, zero_division=0
    )

    # Print Evaluation Results
    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision (Phishing):", round(precision, 4))
    print("Recall (Phishing):", round(recall, 4))
    print("F1-Score (Phishing):", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[-1, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))

    # Store Model and Results
    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

# Create Comparison Table
comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# Save Models
os.makedirs("models", exist_ok=True)

joblib.dump(
    trained_models,
    "models/trained_models.pkl"
)

comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nAll Models Saved Successfully!")
print("Model Comparison Saved Successfully!")
print("\nMODEL TRAINING COMPLETED!")
