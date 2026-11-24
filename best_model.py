import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Test Data
data = joblib.load("models/split_data.pkl")

X_test = data["X_test"]
y_test = data["y_test"]

# Load Tuned Models
tuned_models = joblib.load("models/tuned_models.pkl")

# Load Tuning Results
results = pd.read_csv("models/tuning_results.csv")

# Select Best Model based on Phishing Recall
best_row = results.loc[results["Recall"].idxmax()]
best_model_name = best_row["Model"]

best_model = tuned_models[best_model_name]

print("========== BEST MODEL SELECTION ==========")
print("\nSelected Model:", best_model_name)
print("Cross Validation Recall:", best_row["Recall"])

# Final Prediction on Test Data
y_pred = best_model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label=-1
)

recall = recall_score(
    y_test, y_pred, pos_label=-1
)

f1 = f1_score(
    y_test, y_pred, pos_label=-1
)

print("\n========== FINAL TEST RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[-1, 1],
    target_names=["Phishing", "Legitimate"]
))

# Save Best Model
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

# Save Final Results
final_results = pd.DataFrame([{
    "Best Model": best_model_name,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}])

final_results.to_csv(
    "models/final_evaluation.csv",
    index=False
)

print("\nBest Model Saved Successfully!")
print("Final Evaluation Saved Successfully!")
print("\nSTEP 9 COMPLETED!")
