
from pathlib import Path

DATA_DIR = Path("data/raw")

REQUIRED_FILES = {
    "Demographics": "Block_3_Level _2.csv",
    "Household": "Block 4 Level 3.csv",
    "Disability": "Block 5 Level 4.csv",
    "Caregiver": "Block_6_Level _12.csv",
    "Activity": "Block 8 Level 14.csv",
    "Assistance": "Block 5_1 Level 5.csv",
}

print("\nCareInsight ML - Dataset Verification")
print("------------------------------------")

missing_files = []

for description, filename in REQUIRED_FILES.items():
    file_path = DATA_DIR / filename

    if file_path.is_file():
        print(f"[OK] {description}: {filename}")
    else:
        print(f"[MISSING] {description}: {filename}")
        missing_files.append(filename)

print("------------------------------------")

if missing_files:
    print(f"{len(missing_files)} dataset files missing.")
    print("Place the original CSV files in data/raw.")
else:
    print("All six required dataset files are available.")
