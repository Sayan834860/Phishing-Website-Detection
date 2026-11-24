import pandas as pd
import joblib
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

# 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluation Metrics (Phishing = -1)
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

# Define Models and Parameters
models = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        {
            "model__C": [0.1, 1, 10]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    )
}

tuned_models = {}
results = []

print("========== HYPERPARAMETER TUNING ==========")

for name, (model, params) in models.items():

    print(f"\nTuning {name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scoring,
        refit="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Save Best Model
    tuned_models[name] = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("Best Cross Validation Recall:")
    print(round(grid_search.best_score_, 4))

    # Store Results
    best_index = grid_search.best_index_

    results.append({
        "Model": name,
        "Best Parameters": str(grid_search.best_params_),
        "Accuracy": grid_search.cv_results_["mean_test_accuracy"][best_index],
        "Precision": grid_search.cv_results_["mean_test_precision"][best_index],
        "Recall": grid_search.cv_results_["mean_test_recall"][best_index],
        "F1-Score": grid_search.cv_results_["mean_test_f1"][best_index]
    })

# Create Results Table
tuning_results = pd.DataFrame(results)

print("\n========== TUNING COMPARISON ==========")
print(tuning_results.to_string(index=False))

# Save Results
os.makedirs("models", exist_ok=True)

joblib.dump(
    tuned_models,
    "models/tuned_models.pkl"
)

tuning_results.to_csv(
    "models/tuning_results.csv",
    index=False
)

print("\nTuned Models Saved Successfully!")
print("Tuning Results Saved Successfully!")
print("\nSTEP 8 COMPLETED!")
