import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Load Training Data
data = joblib.load("models/split_data.pkl")

X_train = data["X_train"]
y_train = data["y_train"]

# Define Models
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

# Define 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluation Metrics (Phishing class = -1)
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score, pos_label=-1, zero_division=0
    ),
    "recall": make_scorer(
        recall_score, pos_label=-1, zero_division=0
    ),
    "f1": make_scorer(
        f1_score, pos_label=-1, zero_division=0
    )
}

results = []

print("========== 5-FOLD CROSS VALIDATION ==========")

for name, model in models.items():

    print(f"\nModel: {name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    for metric in scoring:
        fold_scores = scores["test_" + metric]
        mean_score = fold_scores.mean()

        print(f"\n{metric.upper()} Fold Scores:")
        print(fold_scores)

        print(f"Average {metric.upper()}: {mean_score:.4f}")

    results.append({
        "Model": name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Precision Mean": scores["test_precision"].mean(),
        "Recall Mean": scores["test_recall"].mean(),
        "F1 Mean": scores["test_f1"].mean()
    })

# Comparison Table
comparison = pd.DataFrame(results)

print("\n========== CROSS VALIDATION SUMMARY ==========")
print(comparison.to_string(index=False))

# Save Results
comparison.to_csv(
    "models/cross_validation_results.csv",
    index=False
)

print("\nCross Validation Results Saved Successfully!")
print("STEP 7 COMPLETED!")
