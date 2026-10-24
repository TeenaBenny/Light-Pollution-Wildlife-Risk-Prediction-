
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
project_dir = Path(__file__).resolve().parent

file_path = (
    project_dir
    / "data"
    / "raw"
    / "Sanders_et_al_datafile.csv"
)

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

print("=" * 60)
print("LIGHT POLLUTION WILDLIFE RISK - EDA QUESTIONS")
print("=" * 60)

# QUESTION 1: How many records and columns exist?
print("\nQ1. How large is the dataset?")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))

# QUESTION 2: Which species are studied most often?
species_col = "Common.name"

if species_col in df.columns:
    print("\nQ2. Which wildlife species appear most often?")

    species_counts = (
        df[species_col]
        .dropna()
        .value_counts()
        .head(10)
    )

    print(species_counts)

    if not species_counts.empty:
        species_counts.sort_values().plot(
            kind="barh",
            figsize=(9, 5)
        )
        plt.title("Top 10 Wildlife Common Names by Record Count")
        plt.xlabel("Number of records")
        plt.ylabel("Wildlife common name")
        plt.tight_layout()
        plt.show()

# QUESTION 3: Field versus laboratory studies
field_col = "Field.Lab"

if field_col in df.columns:
    print("\nQ3. How are studies divided by field/lab setting?")

    field_counts = df[field_col].fillna(
        "Missing"
    ).value_counts()

    print(field_counts)

    field_counts.plot(kind="bar", figsize=(7, 4))
    plt.title("Research Setting Distribution")
    plt.xlabel("Research setting")
    plt.ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# QUESTION 4: What types of measurements are recorded?
measure_col = "Measure.description"

if measure_col in df.columns:
    print("\nQ4. Which measurements are most common?")

    measure_counts = (
        df[measure_col]
        .dropna()
        .value_counts()
        .head(10)
    )

    print(measure_counts)

    if not measure_counts.empty:
        measure_counts.sort_values().plot(
            kind="barh",
            figsize=(10, 6)
        )
        plt.title("Top 10 Recorded Measurement Descriptions")
        plt.xlabel("Number of records")
        plt.ylabel("Measurement description")
        plt.tight_layout()
        plt.show()

# QUESTION 5: What nighttime light values are available?
light_col = "Night_Lux"

if light_col in df.columns:
    print("\nQ5. What are the nighttime light measurements?")

    light_values = pd.to_numeric(
        df[light_col], errors="coerce"
    ).dropna()

    print(light_values.describe())

    if not light_values.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(light_values, bins=20, edgecolor="black")
        plt.title("Distribution of Nighttime Light Measurements")
        plt.xlabel("Nighttime light (lux, if this is the recorded unit)")
        plt.ylabel("Number of records")
        plt.tight_layout()
        plt.show()

print("\nEDA questions completed.")
