
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Use the same feature list as the deployed model.
SELECTED = joblib.load(
    ROOT / "models" / "selected_features.joblib"
)

# Original disability indicators from the notebook.
DISABILITY_FLAGS = [
    "dis_limbs_movement",
    "dis_loss_sensation",
    "dis_deformity",
    "dis_seeing",
    "dis_hearing",
    "dis_speech",
    "dis_understanding",
    "dis_reasoning",
    "dis_anxiety_mood",
    "dis_unusual_experiences",
    "dis_social_interaction",
    "dis_neuro_blood",
]

# The notebook removed the constant reasoning feature.
DISABILITY_FLAGS.remove("dis_reasoning")

BINARY = [
    "sector",
    *DISABILITY_FLAGS,
    "uses_public_transport",
    "accesses_public_building",
    "working_before_onset",
    "has_certificate",
    "subsidiary_work",
]

NOMINAL = [
    "relation_to_head",
    "parents_blood_related",
    "gender",
    "marital_status",
    "education_level",
    "technical_education",
    "religion",
    "social_group",
    "land_possessed",
    "aid_help_received",
    "principal_activity_status",
]

NUMERIC = [
    "age",
    "household_size",
    "log_monthly_exp",
]


def prepare_features(df):
    df = df.copy()

    # 1. Validate survey-coded binary values.
    for column in BINARY:
        valid = df[column].dropna().isin([1, 2])

        if not valid.all():
            raise ValueError(
                f"Unexpected survey code in {column}"
            )

    # 2. Recode binary survey variables.
    # Sector: 2 = urban, 1 = rural.
    df["sector"] = (
        df["sector"] == 2
    ).astype(int)

    # Other binary values: 1 = yes, 2 = no.
    for column in BINARY:
        if column != "sector":
            df[column] = (
                df[column] == 1
            ).astype(int)

    # 3. Convert nominal categories to strings.
    # This matches the original training pipeline.
    for column in NOMINAL:
        df[column] = df[column].map(
            lambda value: (
                np.nan
                if pd.isna(value)
                else str(int(value))
            )
        )

    # 4. Log-transform household expenditure.
    expenditure = pd.to_numeric(
        df["monthly_consumer_exp"],
        errors="coerce"
    )

    if (expenditure.dropna() < 0).any():
        raise ValueError(
            "Negative monthly expenditure detected"
        )

    df["log_monthly_exp"] = np.log1p(
        expenditure
    )

    # 5. Engineer multiple-disability indicator.
    disability_count = df[
        DISABILITY_FLAGS
    ].sum(axis=1)

    df["multiple_disability"] = (
        disability_count >= 2
    ).astype(int)

    # 6. Validate selected feature schema.
    missing = [
        column
        for column in SELECTED
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing selected features: {missing}"
        )

    # Target and household IDs are not model inputs.
    X = df[SELECTED].copy()
    y = df["needs_care"].astype(int).copy()

    return X, y


def main():
    for split in ["train", "test"]:
        path = DATA_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(
            path,
            low_memory=False
        )

        X, y = prepare_features(df)

        print(f"\n{split.upper()} DATA")
        print("-" * 40)
        print(f"Records: {len(X):,}")
        print(f"Selected features: {len(X.columns)}")
        print(f"Positive-class proportion: {y.mean():.4f}")

        print("\nFeature data types:")
        print(X.dtypes.to_string())

        output_X = DATA_DIR / f"X_{split}.pkl"
        output_y = DATA_DIR / f"y_{split}.pkl"

        X.to_pickle(output_X)
        y.to_pickle(output_y)

        print(f"\nSaved: {output_X.name}")
        print(f"Saved: {output_y.name}")

    print("\nFeature engineering completed.")


if __name__ == "__main__":
    main()
