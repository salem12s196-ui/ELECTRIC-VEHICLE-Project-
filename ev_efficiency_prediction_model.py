"""
Electric Vehicle High-Efficiency Prediction Project
"""

from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent

# Find the folder containing the EV CSV dataset.
possible_data_folders = [
    SCRIPT_DIR,
    SCRIPT_DIR.parent
]

DATA_DIR = os.getcwd()
DATASET_FILE = "EV_dataset.csv"

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find the dataset file. Place Python script in the "
        "same folder as ( 'EV_dataset.csv'), "
        "or keep the dataset in the parent folder."
    )

os.chdir(DATA_DIR)
print("Dataset/output folder:", DATA_DIR)
print("Using dataset file:", DATASET_FILE)


# Part 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_validate
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)


# Part 2: Load the Dataset

df = pd.read_csv(DATASET_FILE, low_memory=False)

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()


# Part 3: Data Understanding

print("Statistical Summary:")
print(df.describe(include="all").to_string())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["target_high_efficiency"].value_counts())

print("\nTarget Percentages:")
print(df["target_high_efficiency"].value_counts(normalize=True) * 100)


# Part 4: Check Data Quality

print("Missing Values in Dataset:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print("Duplicates:", df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

quality = pd.DataFrame({
    "Column": df.columns,
    "Missing": df.isnull().sum().values,
    "Missing_Percent": (
        df.isnull().mean().values * 100
    ).round(2),
    "Unique_Values": df.nunique().values
})

print("\nData Quality Summary:")
print(quality)

quality.to_csv("data_quality.csv", index=False)


# Part 5: Data Cleaning

df_clean = df.copy()

# Remove duplicate records
df_clean = df_clean.drop_duplicates()

# Remove rows with missing target values
df_clean = df_clean.dropna(subset=["target_high_efficiency"])

# Clean text values
text_columns = df_clean.select_dtypes(
    include=["object"]
).columns

for col in text_columns:
    df_clean[col] = df_clean[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    df_clean[col] = df_clean[col].replace("", np.nan)

# Check invalid numeric values
positive_columns = [
    "battery_kwh", "range_km", "charging_time_hr",
    "price_usd", "top_speed_kmph", "acceleration_0_100_kmph"
]

for col in positive_columns:
    if col in df_clean.columns:
        df_clean.loc[df_clean[col] <= 0, col] = np.nan

# Check outliers using IQR
for col in ["price_usd", "range_km", "battery_kwh"]:
    if col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = (
            (df_clean[col] < lower) |
            (df_clean[col] > upper)
        ).sum()

        print(col, "Outliers:", outliers)

# Keep valid outliers because high-end luxury EVs and long-range models are real vehicles.

print("\nRows Before Cleaning:", len(df))
print("Rows After Cleaning:", len(df_clean))


# Part 6: Exploratory Data Analysis

# Chart 1: Target Class Distribution
plt.figure(figsize=(6, 4))
df_clean["target_high_efficiency"].value_counts().sort_index().plot(
    kind="bar", figsize=(6, 4)
)
plt.title("EV High-Efficiency Distribution")
plt.xlabel("Target (0 = Standard Efficiency, 1 = High Efficiency)")
plt.ylabel("Number of Vehicles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("efficiency_distribution.png")
plt.close()

# Chart 2: EV Price Distribution
plt.figure(figsize=(7, 4))
plt.hist(df_clean["price_usd"].dropna(), bins=40)
plt.title("EV Price Distribution (USD)")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

# Chart 3: High-Efficiency Rate by Top 10 Manufacturers
plt.figure(figsize=(9, 5))
maker_rate = df_clean.groupby("manufacturer")["target_high_efficiency"].agg(["mean", "count"])
maker_rate = maker_rate[maker_rate["count"] >= 200]
maker_rate = maker_rate.sort_values("mean", ascending=False).head(10)
maker_rate["mean"].plot(kind="bar", figsize=(9, 5))
plt.title("High-Efficiency Rate by Top 10 EV Manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("High-Efficiency Rate")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("manufacturer_efficiency_rate.png")
plt.close()

# Chart 4: Average Range (km) by Efficiency Target
plt.figure(figsize=(6, 4))
df_clean.groupby("target_high_efficiency")["range_km"].mean().plot(
    kind="bar", figsize=(6, 4)
)
plt.title("Average Range (km) by Efficiency Class")
plt.xlabel("Target (0 = Standard Efficiency, 1 = High Efficiency)")
plt.ylabel("Average Range (km)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average_range.png")
plt.close()

# Chart 5: High-Efficiency Rate by Release Year
plt.figure(figsize=(8, 4))
yearly_rate = df_clean.groupby("release_year")["target_high_efficiency"].mean().sort_index()
yearly_rate.plot(kind="line", marker="o", figsize=(8, 4))
plt.title("High-Efficiency Rate by Release Year")
plt.xlabel("Release Year")
plt.ylabel("High-Efficiency Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("yearly_efficiency_rate.png")
plt.close()


# Part 7: Feature Engineering

# Engineer domain-specific features
df_clean["vehicle_age"] = 2026 - df_clean["release_year"]
df_clean["price_per_kwh"] = df_clean["price_usd"] / df_clean["battery_kwh"]

# Remove target, leakage column (efficiency_score), and non-predictive columns
drop_columns = [
    "target_high_efficiency",
    "efficiency_score",
    "color",
    "city"
]

X = df_clean.drop(columns=drop_columns, errors="ignore")
y = df_clean["target_high_efficiency"].astype(int)

# Remove columns with only one unique value
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_columns)

# Treat boolean columns like fast_charging as categorical text
if "fast_charging" in X.columns:
    X["fast_charging"] = X["fast_charging"].apply(
        lambda x: str(x) if pd.notna(x) else np.nan
    )

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = [col for col in X.columns if col not in numeric_columns]

for col in categorical_columns:
    X[col] = X[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)

print("Numeric Columns:", numeric_columns)
print("\nCategorical Columns:", categorical_columns)
print("\nFinal Features:", X.shape[1])


# Part 8: Split Training and Validation Data

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Rows:", len(X_train))
print("Validation Rows:", len(X_valid))


# Part 9: Data Preprocessing

numeric_process = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_process = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_process, numeric_columns),
    ("cat", categorical_process, categorical_columns)
])


# Part 10: Build Machine Learning Models

models = {
    "Decision Tree": Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ("preprocess", preprocessor),
        ("model", BernoulliNB())
    ])
}

print("Models created:")
for name in models:
    print(name)


# Part 11: Hyperparameter Tuning

parameters = {
    "Decision Tree": {
        "model__max_depth": [5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 5, 10]
    },
    "Naive Bayes": {
        "model__alpha": [0.1, 0.5, 1, 2]
    }
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_models = {}
tuning_results = []

for name in parameters:
    print("\nTuning:", name)

    search = RandomizedSearchCV(
        estimator=models[name],
        param_distributions=parameters[name],
        n_iter=5,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_

    tuning_results.append({
        "Model": name,
        "Best F1": search.best_score_,
        "Best Parameters": str(search.best_params_)
    })

tuning_df = pd.DataFrame(tuning_results)

print("\nTuning Results:")
print(tuning_df.to_string(index=False))

tuning_df.to_csv("hyperparameter_tuning.csv", index=False)


# Part 12: Cross-Validation

cv_results = []

for name, model in best_models.items():
    print("Checking:", name)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring={
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
            "ROC_AUC": "roc_auc"
        },
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_Accuracy"].mean(),
        "Precision": scores["test_Precision"].mean(),
        "Recall": scores["test_Recall"].mean(),
        "F1": scores["test_F1"].mean(),
        "ROC_AUC": scores["test_ROC_AUC"].mean()
    })

cv_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(cv_df.to_string(index=False))

cv_df.to_csv("cross_validation.csv", index=False)


# Part 13: Final Model Evaluation

results = []

for name, model in best_models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(y_valid, predictions, zero_division=0)
    recall = recall_score(y_valid, predictions, zero_division=0)
    f1 = f1_score(y_valid, predictions, zero_division=0)
    auc = roc_auc_score(y_valid, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_valid, predictions)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Standard Efficiency", "High Efficiency"]
    ).plot()
    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    plt.savefig(name.replace(" ", "_") + "_confusion.png")
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_valid, probabilities)
    plt.title(name + " ROC Curve")
    plt.tight_layout()
    plt.savefig(name.replace(" ", "_") + "_roc.png")
    plt.close()


# Part 14: Compare Models

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

plt.figure(figsize=(11, 6))
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
].plot(kind="bar", figsize=(11, 6))
plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()


# Part 15: Select and Save Best Model

best_name = cv_df.sort_values("F1", ascending=False).iloc[0]["Model"]
print("Selected Model:", best_name)

best_model = best_models[best_name]

# Train the final model using all labelled data
best_model.fit(X, y)

joblib.dump(best_model, "ev_efficiency_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")


# Final Project Summary

print("\nEV HIGH-EFFICIENCY PREDICTION PROJECT")
print("-------------------------------------")
print("Total Records:", len(df))
print("Training Split (80%):", len(X_train))
print("Validation Split (20%):", len(X_valid))
print("Predictor Columns:", len(X.columns))
print("\nSelected Model:", best_name)
print("\nModel Performance:")
print(results_df.to_string(index=False))
print("\nProject completed successfully.")
print("Model file:", "ev_efficiency_model.pkl")
print("Feature columns:", "model_columns.pkl")