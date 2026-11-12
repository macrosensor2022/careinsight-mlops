
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

FILES = {
    "Demographics": "Block_3_Level _2.csv",
    "Household": "Block 4 Level 3.csv",
    "Disability": "Block 5 Level 4.csv",
    "Caregiver": "Block_6_Level _12.csv",
    "Activity": "Block 8 Level 14.csv",
    "Assistance": "Block 5_1 Level 5.csv",
}

print("\nCAREINSIGHT — DATASET INSPECTION")
print("=" * 60)

for name, filename in FILES.items():
    path = DATA_DIR / filename

    # Inspect only column names and count data rows.
    # Do not print individual survey records.
    columns = pd.read_csv(
        path,
        nrows=0,
        low_memory=False
    ).columns.tolist()

    row_count = sum(
        1 for chunk in pd.read_csv(
            path,
            chunksize=50000,
            low_memory=False
        )
        for _ in range(len(chunk))
    )

    print(f"\nDATASET: {name}")
    print(f"File: {filename}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(columns)}")
    print(f"First 15 column names: {columns[:15]}")

print("\nInspection completed.")
