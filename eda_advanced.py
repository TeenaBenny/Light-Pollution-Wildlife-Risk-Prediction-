
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================

project_dir = Path(__file__).resolve().parent
file_path = (
    project_dir / "data" / "raw"
    / "Sanders_et_al_datafile.csv"
)

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

print("Dataset loaded:", df.shape)

# ==========================================
# QUESTION 1
# What reported effect signs occur most often?
# ==========================================

if "Sign" in df.columns:
    print("\nQUESTION 1: Distribution of reported effect signs")

    sign_counts = df["Sign"].fillna(
        "Missing"
    ).astype(str).value_counts()

    print(sign_counts)

    sign_counts.plot(kind="bar", figsize=(7, 4))
    plt.title("Distribution of Reported Effect Signs")
    plt.xlabel("Recorded sign")
    plt.ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ==========================================
# QUESTION 2
# How do reported signs vary by activity?
# ==========================================

if "Sign" in df.columns and "Activity" in df.columns:
    print("\nQUESTION 2: Reported signs by activity")

    activity_data = df[["Activity", "Sign"]].dropna()

    if not activity_data.empty:
        table = pd.crosstab(
            activity_data["Activity"],
            activity_data["Sign"]
        )

        print(table)

        table.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 5)
        )

        plt.title("Reported Effect Signs by Wildlife Activity")
        plt.xlabel("Activity")
        plt.ylabel("Number of records")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Recorded sign")
        plt.tight_layout()
        plt.show()

# ==========================================
# QUESTION 3
# How many observations were in each group?
# ==========================================

sample_columns = [
    col for col in ["Control_N", "Experimental_N"]
    if col in df.columns
]

if sample_columns:
    print("\nQUESTION 3: Sample size summary")

    for col in sample_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(df[sample_columns].describe())

    df[sample_columns].plot(
        kind="box",
        figsize=(7, 4)
    )

    plt.title("Control and Experimental Sample Sizes")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.show()

print("\nAdvanced EDA completed.")
