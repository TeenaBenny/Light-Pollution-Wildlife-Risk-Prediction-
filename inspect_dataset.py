
from pathlib import Path
import pandas as pd

# Correct project folder
project_folder = Path(r"C:\teena_ML_CC\Light_Pollution_Wildlife_Risk")

# Dataset location
file_path = (
    project_folder
    / "data"
    / "raw"
    / "Sanders_et_al_datafile.csv"
)

print("Checking dataset location:", file_path)

if not file_path.exists():
    print("ERROR: Dataset file not found!")
    print("Files inside the raw folder:")

    raw_folder = project_folder / "data" / "raw"

    if raw_folder.exists():
        for file in raw_folder.iterdir():
            print(file.name)
    else:
        print("The data/raw folder does not exist.")

else:
    df = pd.read_csv(file_path, encoding="latin1")

    print("\nDATASET LOADED SUCCESSFULLY!")
    print("\n1. DATASET SIZE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. FIRST 5 ROWS")
    print(df.head().to_string())

    print("\n4. DATA TYPES")
    print(df.dtypes)

    print("\n5. MISSING VALUES")
    print(df.isnull().sum())

    print("\n6. DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\n7. SUMMARY STATISTICS")
    print(df.describe(include="all").T)
