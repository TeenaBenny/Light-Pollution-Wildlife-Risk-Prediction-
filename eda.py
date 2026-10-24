
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Sanders_et_al_datafile.csv"
)

REPORT_DIR = PROJECT_DIR / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. LOAD DATASET
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, encoding="latin1")

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("=" * 60)
print("LIGHT POLLUTION WILDLIFE RISK - EDA")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head().to_string())

# ==========================================
# 3. DATASET INFORMATION
# ==========================================

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if not missing.empty else "No missing values")

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# ==========================================
# 4. MISSING VALUE VISUALIZATION
# ==========================================

missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]

if not missing_counts.empty:
    plt.figure(figsize=(12, 6))
    missing_counts.sort_values().plot(kind="barh")
    plt.title("Missing Values by Column")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "missing_values.png", dpi=150)
    plt.show()
else:
    print("\nMissing-value chart skipped: no missing values.")

# ==========================================
# 5. NUMERIC DATA DISTRIBUTIONS
# ==========================================

numeric_df = df.select_dtypes(include=np.number)

if not numeric_df.empty:
    print("\nNUMERIC SUMMARY")
    print(numeric_df.describe().T)

    # Histograms
    numeric_df.hist(
        figsize=(15, 12),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle("Distributions of Numeric Variables")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "numeric_distributions.png", dpi=150)
    plt.show()

    # Correlation heatmap
    if numeric_df.shape[1] >= 2:
        correlation = numeric_df.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            annot=False
        )
        plt.title("Correlation Between Numeric Variables")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
        plt.show()
else:
    print("\nNo numeric columns available for numeric analysis.")

# ==========================================
# 6. CATEGORICAL DATA ANALYSIS
# ==========================================

categorical_df = df.select_dtypes(
    include=["object", "category", "bool"]
)

print("\nCATEGORICAL COLUMN SUMMARY")

for column in categorical_df.columns:
    print(f"\n{column}")
    print("Unique values:", df[column].nunique(dropna=True))
    print(df[column].value_counts(dropna=False).head(10))

    # Avoid plotting identifiers and very high-cardinality columns
    if 2 <= df[column].nunique(dropna=True) <= 15:
        counts = df[column].value_counts().head(15)

        plt.figure(figsize=(10, 5))
        counts.plot(kind="bar")
        plt.title(f"Top Categories: {column}")
        plt.xlabel(column)
        plt.ylabel("Record Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        safe_name = "".join(
            char if char.isalnum() else "_"
            for char in column
        )

        plt.savefig(
            REPORT_DIR / f"category_{safe_name}.png",
            dpi=150
        )
        plt.show()

# ==========================================
# 7. EXPORT BASIC EDA REPORTS
# ==========================================

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(dtype) for dtype in df.dtypes],
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2),
    "Unique_Values": df.nunique(dropna=True).values
})

missing_report.to_csv(
    REPORT_DIR / "column_quality_report.csv",
    index=False
)

numeric_df.describe().T.to_csv(
    REPORT_DIR / "numeric_summary.csv"
)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("Reports saved to:", REPORT_DIR)
print("=" * 60)
