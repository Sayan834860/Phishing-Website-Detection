# Phishing Website Detection Using Machine Learning

## Project Overview
Phishing Website Detection is a Machine Learning project that identifies whether a website is potentially phishing or legitimate based on its URL-related and website characteristics.

The system uses three classification algorithms to train, evaluate, compare, and select the best-performing model.

## Project Objective
To develop a Machine Learning system that predicts whether a website is legitimate or potentially phishing.

## Dataset Information
- Dataset: UCI Phishing Websites Dataset
- Total Records: 11,055
- Input Features: 30
- Target Column: Result
- Classification Type: Binary Classification

### Target Labels
- -1: Phishing Website
- 1: Legitimate Website

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Joblib
- Streamlit

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Project Workflow
1. Dataset Loading and Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Train-Test Split (80:20)
5. Training Three Classification Models
6. Model Evaluation and Comparison
7. 5-Fold Cross Validation
8. Hyperparameter Tuning using GridSearchCV
9. Best Model Selection
10. Classification of New Website Data
11. Model Saving and Web Interface Development

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 91.28% |
| Decision Tree | 91.71% |
| Random Forest | 94.19% |

### Final Selected Model
Random Forest Classifier

- Accuracy: 94.19%
- Precision: 94.67%
- Recall: 94.04%
- F1-Score: 94.35%

## Project Structure

```text
Phishing-Website-Detection/
│
├── dataset/
│   ├── Training Dataset.arff
│   └── cleaned_dataset.csv
│
├── models/
│   ├── best_model.pkl
│   ├── trained_models.pkl
│   ├── tuned_models.pkl
│   ├── split_data.pkl
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   ├── tuning_results.csv
│   └── final_evaluation.csv
│
├── notebooks/
├── visualizations/
│
├── main.py
├── eda.py
├── preprocessing.py
├── split_data.py
├── train_models.py
├── cross_validation.py
├── hyperparameter_tuning.py
├── best_model.py
├── app.py
├── README.md
└── requirements.txt
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run the Project

### Step 1: Load Dataset

```bash
python main.py
```

### Step 2: Perform EDA

```bash
python eda.py
```

### Step 3: Data Preprocessing

```bash
python preprocessing.py
```

### Step 4: Train-Test Split

```bash
python split_data.py
```

### Step 5: Train Machine Learning Models

```bash
python train_models.py
```

### Step 6: Cross Validation

```bash
python cross_validation.py
```

### Step 7: Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

### Step 8: Best Model Evaluation

```bash
python best_model.py
```

### Step 9: Run Web Application

```bash
python -m streamlit run app.py
```

The application will open in a web browser.

## Expected Output

The system classifies website feature records into two categories:

- Potentially Phishing Website
- Likely Legitimate Website

## Conclusion

The project demonstrates how Machine Learning algorithms can be used for phishing website detection. Three classification models are trained and evaluated, and Random Forest achieves the best performance among the tested models.

The final trained model is saved using Joblib and integrated into a Streamlit web application for classifying new website feature data.

## Project Information

**Project Title:** Phishing Website Detection Using Machine Learning

**Department:** MCA

**Section:** E

**Project Type:** PBL Project