"""Train and save a Logistic Regression pipeline for the supplied CSV."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(
    r"C:\Users\battu\OneDrive\Documents\SEMESTER 3\Project-1"
    r"\Online_purchase_conversion_advisor\data\preprocessed_dataset.csv"
)
MODEL_PATH = Path(__file__).with_name("model.joblib")
TARGET = "Revenue"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
if TARGET not in data.columns:
    raise ValueError(f"Dataset must contain target column {TARGET!r}.")

# Normalize the preprocessed CSV's categorical text values and target labels.
for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].str.strip()

y_raw = data.pop(TARGET)
if y_raw.dtype == bool:
    y = y_raw.astype(int)
else:
    target_map = {
        True: 1, False: 0,
        "True": 1, "False": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0,
    }
    y = y_raw.map(target_map)
    if y.isna().any():
        raise ValueError(f"Unrecognized values in {TARGET}: {y_raw.unique()!r}")
    y = y.astype(int)

X = data
if X.isna().any().any():
    raise ValueError("The dataset contains missing feature values; decide how to handle them before training.")

categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
numeric_features = [column for column in X.columns if column not in categorical_features]
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ]
)

# Report a holdout evaluation, then fit the saved artifact on all available rows.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
print(f"Rows: {len(data)} | Features: {len(X.columns)}")
print(f"Feature order: {list(X.columns)}")
print(f"Holdout accuracy: {accuracy_score(y_test, predicted):.4f}")
print(f"Holdout ROC AUC: {roc_auc_score(y_test, probabilities):.4f}")
print(classification_report(y_test, predicted, target_names=["No purchase", "Purchase"]))

pipeline.fit(X, y)
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved fitted Logistic Regression pipeline to: {MODEL_PATH}")
