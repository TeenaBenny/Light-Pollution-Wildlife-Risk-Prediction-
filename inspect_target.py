
from pathlib import Path
import pandas as pd

# Find the project folder
PROJECT_DIR = Path(__file__).resolve().parent

# Use cleaned data if available; otherwise use the original data
cleaned_path = (
    PROJECT_DIR / "data" / "processed" / "wildlife_cleaned.csv"
)
raw_path = (
    PROJECT_DIR / "data" / "raw" / "Sanders_et_al_datafile.csv"
)

if cleaned_path.exists():
    DATA_PATH = cleaned_path
else:
    DATA_PATH = raw_path

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

print("=" * 60)
print("WILDLIFE DATASET: TARGET COLUMN INSPECTION")
print("=" * 60)

print("\nDataset shape:", df.shape)

# Display all column names
print("\n1. Available columns:")
for column in df.columns:
    print("-", column)

# Inspect possible target columns
candidate_columns = [
    "Sign",
    "Category",
    "Activity",
    "Noct",
    "Field.Lab",
    "Measure.description",
]

for column in candidate_columns:
    if column in df.columns:
        print(f"\n2. Values in '{column}':")

        print(
            df[column]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

# Inspect light exposure measurements
if "Night_Lux" in df.columns:
    print("\n3. Night_Lux summary:")
    print(
        pd.to_numeric(df["Night_Lux"], errors="coerce")
        .describe()
    )

# Inspect experimental outcomes
outcome_columns = [
    "Control_Mean",
    "Experimental_Mean",
    "Control_SD",
    "Experimental_SD",
    "Control_N",
    "Experimental_N",
]

for column in outcome_columns:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        print(f"\n4. {column} summary:")
        print(values.describe())

print("\nInspection complete.")
print("Dataset used:", DATA_PATH)
