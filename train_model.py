
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

cleaned_path = PROJECT_DIR / "data" / "processed" / "wildlife_cleaned.csv"
raw_path = PROJECT_DIR / "data" / "raw" / "Sanders_et_al_datafile.csv"

DATA_PATH = cleaned_path if cleaned_path.exists() else raw_path
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "wildlife_model.joblib"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

print("\nDataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 3. CHECK TARGET
# --------------------------------------------------
TARGET = "Sign"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df[TARGET] = df[TARGET].astype("string").str.strip()
df[TARGET] = df[TARGET].replace("", pd.NA)
df = df.dropna(subset=[TARGET]).copy()

print("\nTarget distribution:")
print(df[TARGET].value_counts())

if df[TARGET].nunique() < 2:
    raise ValueError("Need at least two different target classes.")

# --------------------------------------------------
# 4. SELECT FEATURES
# Exclude identifiers and direct outcome measurements
# to reduce leakage and high-cardinality noise.
# --------------------------------------------------
excluded = [
    TARGET,
    "Study",
    "ID",
    "Title",
    "DOI",
    "Author",
    "Source",
    "Delete",
    "Control_Mean",
    "Control_SD",
    "Experimental_Mean",
    "Experimental_SD",
]

X = df.drop(
    columns=[c for c in excluded if c in df.columns],
    errors="ignore",
)

y = df[TARGET]

# Drop columns that have no useful variation
X = X.loc[:, X.nunique(dropna=False) > 1]

# Remove rows with missing values in every feature
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

if X.empty or X.shape[1] == 0:
    raise ValueError("No usable features remain after cleaning.")

# --------------------------------------------------
# 5. CHECK CLASS COUNTS
# --------------------------------------------------
class_counts = y.value_counts()
print("\nUsable rows:", len(y))
print("Feature count:", X.shape[1])
print("Class counts:\n", class_counts)

# Use stratified split only if every class has at least 2 rows
stratify_y = y if class_counts.min() >= 2 else None

if len(y) < 10:
    raise ValueError(
        "Too few usable rows for a meaningful train/test evaluation."
    )

# --------------------------------------------------
# 6. SPLIT DATA
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_y,
)

# --------------------------------------------------
# 7. PREPROCESS FEATURES
# --------------------------------------------------
numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in X_train.columns if c not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)

# --------------------------------------------------
# 9. EVALUATE MODEL
# --------------------------------------------------
predictions = model.predict(X_test)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 10. SAVE MODEL AND FEATURE NAMES
# --------------------------------------------------
joblib.dump({
    "model": model,
    "features": X.columns.tolist(),
    "target": TARGET,
    "classes": model.named_steps["classifier"].classes_.tolist(),
}, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Training complete.")
