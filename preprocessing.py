
from pathlib import Path
import pandas as pd

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

RAW_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Sanders_et_al_datafile.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "wildlife_cleaned.csv"

# ==========================================
# 2. LOAD DATA
# ==========================================

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

print("=" * 55)
print("WILDLIFE DATA PREPROCESSING")
print("=" * 55)

print("\nOriginal dataset shape:", df.shape)

# ==========================================
# 3. REMOVE FULLY DUPLICATED ROWS
# ==========================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

df = df.drop_duplicates().copy()

print("Rows after duplicate removal:", len(df))

# ==========================================
# 4. CLEAN TEXT COLUMNS
# ==========================================

text_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in text_columns:
    df[column] = df[column].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

    # Treat blank strings as missing values
    df[column] = df[column].replace("", pd.NA)

# ==========================================
# 5. CONVERT KNOWN NUMERIC COLUMNS
# ==========================================

numeric_candidates = [
    "Publication.year",
    "Night_Lux",
    "Control_N",
    "Experimental_N",
    "Control_Mean",
    "Control_SD",
    "Experimental_Mean",
    "Experimental_SD",
    "Lat",
    "Long",
]

for column in numeric_candidates:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ==========================================
# 6. REPORT MISSING VALUES
# ==========================================

print("\nMissing values by column:")
print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

# Do not fill all missing values with zero.
# Missing scientific measurements need careful handling.

# ==========================================
# 7. SAVE CLEANED COPY
# ==========================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Saved cleaned dataset to:")
print(OUTPUT_PATH)

print("\nPreprocessing completed.")
