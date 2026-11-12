
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

input_path = DATA_DIR / "person_level.csv"

df = pd.read_csv(input_path, low_memory=False)
# Match the original notebook's age cleaning.
df = df.loc[df["age"].between(18, 110)].copy()
df = df.reset_index(drop=True)

print(f"Records after age validation: {len(df):,}")

print("\nCareInsight — Train/Test Split")
print("-" * 45)

# Validate identifiers and original target codes
assert not df["HHID"].isna().any()
assert not df["person_id"].isna().any()
assert not df["caregiver_code"].isna().any()
assert not df.duplicated(["HHID", "person_id"]).any()

# Reproduce original notebook's binary label.
# Survey-code meanings still require validation.
df["needs_care"] = (
    df["caregiver_code"] != 16
).astype(int)

print("Target distribution:")
print(df["needs_care"].value_counts())
print()

# Keep members of each household together.
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X=df,
        y=df["needs_care"],
        groups=df["HHID"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# Check for household leakage
train_households = set(train_df["HHID"])
test_households = set(test_df["HHID"])

overlap = train_households.intersection(
    test_households
)

assert len(overlap) == 0
assert len(train_df) + len(test_df) == len(df)

print(f"Training records: {len(train_df):,}")
print(f"Testing records: {len(test_df):,}")
print(f"Overlapping households: {len(overlap)}")

print("\nTraining target proportions:")
print(train_df["needs_care"].value_counts(normalize=True))

print("\nTesting target proportions:")
print(test_df["needs_care"].value_counts(normalize=True))

train_df.to_csv(DATA_DIR / "train.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("\nSaved train.csv and test.csv")
print("Household-grouped split completed.")
