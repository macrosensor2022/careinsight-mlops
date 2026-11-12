
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

FILES = {
    "L02": "Block_3_Level _2.csv",
    "L03": "Block 4 Level 3.csv",
    "L04": "Block 5 Level 4.csv",
    "L12": "Block_6_Level _12.csv",
    "L14": "Block 8 Level 14.csv",
}

FLAG_COLS = [
    "Difficulty_handsFingersToes_move",
    "Loss_sensation_paralysisleprosy",
    "Deformity_bodyparts_leprosy_acid",
    "Difficulty_seeing_10feet",
    "Difficulty_hearing_daytoday_spee",
    "Difficulty_in_speech",
    "Difficulty_understand_dailyactiv",
    "Difficulty_understand_reasoning",
    "Unnecessary_worry_anxiety",
    "Unusual_voices_seeing_visions_sm",
    "Unusual_behaviour_social_interac",
    "Parkinson_multiple_sclerosis_oth",
]

RENAME = {
    "Sector": "sector",
    "Arrangement_regular_caregiver": "caregiver_code",
    "Receipt_any_aid_help": "aid_help_received",
    "Uses_public_transport": "uses_public_transport",
    "Accesses_public_building": "accesses_public_building",
    "Working_before_onset_disability": "working_before_onset",
    "Having_a_certificate_of_disabili": "has_certificate",
    "Relation_to_head": "relation_to_head",
    "Whether_the_parents_are_blood_re": "parents_blood_related",
    "Gender": "gender",
    "Age": "age",
    "Marital_status": "marital_status",
    "Highest_level_education": "education_level",
    "Highest_level_technical_educatio": "technical_education",
    "Household_size": "household_size",
    "Religion": "religion",
    "Social_group": "social_group",
    "Land_possessed": "land_possessed",
    "Usual_monthly_consumer_expenditu": "monthly_consumer_exp",
    "Difficulty_handsFingersToes_move": "dis_limbs_movement",
    "Loss_sensation_paralysisleprosy": "dis_loss_sensation",
    "Deformity_bodyparts_leprosy_acid": "dis_deformity",
    "Difficulty_seeing_10feet": "dis_seeing",
    "Difficulty_hearing_daytoday_spee": "dis_hearing",
    "Difficulty_in_speech": "dis_speech",
    "Difficulty_understand_dailyactiv": "dis_understanding",
    "Difficulty_understand_reasoning": "dis_reasoning",
    "Unnecessary_worry_anxiety": "dis_anxiety_mood",
    "Unusual_voices_seeing_visions_sm": "dis_unusual_experiences",
    "Unusual_behaviour_social_interac": "dis_social_interaction",
    "Parkinson_multiple_sclerosis_oth": "dis_neuro_blood",
    "Usual_principal_activity_status": "principal_activity_status",
    "Engaged_anywork_Subsidiary_capac": "subsidiary_work",
}


def load_datasets():
    raw = {}

    for level, filename in FILES.items():
        path = DATA_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

        df = pd.read_csv(
            path,
            low_memory=False
        )

        # Match the notebook's household-key conversion.
        df["HHID"] = df["HHID"].astype("int64")

        raw[level] = df

        print(
            f"{level}: {len(df):,} rows, "
            f"{len(df.columns)} columns"
        )

    return raw


def merge_datasets(raw):
    l12 = raw["L12"].rename(
        columns={"Srl_col1block_5": "person_id"}
    )[[
        "HHID", "person_id", "Sector", "MULT",
        "Arrangement_regular_caregiver",
        "Receipt_any_aid_help",
        "Uses_public_transport",
        "Accesses_public_building",
        "Working_before_onset_disability",
        "Having_a_certificate_of_disabili"
    ]]

    l02 = raw["L02"].rename(
        columns={"Person_Srl_No": "person_id"}
    )[[
        "HHID", "person_id",
        "Relation_to_head",
        "Whether_the_parents_are_blood_re",
        "Gender", "Age", "Marital_status",
        "Highest_level_education",
        "Highest_level_technical_educatio"
    ]]

    l03 = raw["L03"][[
        "HHID", "Household_size",
        "Religion", "Social_group",
        "Land_possessed",
        "Usual_monthly_consumer_expenditu"
    ]]

    l04 = raw["L04"].rename(
        columns={"Srl_col1_bl_3": "person_id"}
    )[["HHID", "person_id"] + FLAG_COLS]

    l14 = raw["L14"].rename(
        columns={"Srl_col1block_5": "person_id"}
    )[[
        "HHID", "person_id",
        "Usual_principal_activity_status",
        "Engaged_anywork_Subsidiary_capac"
    ]]

    merged = (
        l12
        .merge(
            l02,
            on=["HHID", "person_id"],
            how="left",
            validate="1:1"
        )
        .merge(
            l03,
            on="HHID",
            how="left",
            validate="m:1"
        )
        .merge(
            l04,
            on=["HHID", "person_id"],
            how="left",
            validate="1:1"
        )
        .merge(
            l14,
            on=["HHID", "person_id"],
            how="left",
            validate="1:1"
        )
    )

    return merged.rename(columns=RENAME)


def main():
    print("\nLoading NSS survey tables...")
    raw = load_datasets()

    print("\nMerging survey records...")
    df = merge_datasets(raw)

    print(f"Rows after merge: {len(df):,}")
    print(f"Columns after merge: {len(df.columns)}")

    print("\nUnmatched source indicators:")
    for column in [
        "gender",
        "household_size",
        "dis_seeing",
        "principal_activity_status"
    ]:
        print(
            f"{column}: {df[column].isna().sum():,} missing"
        )

    # The original research kept adults aged 18+.
    df = df[df["age"] >= 18].reset_index(drop=True)

    print(f"\nAdult records (18+): {len(df):,}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "person_level.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved local dataset: {output_path}")
    print("\nData preparation completed.")


if __name__ == "__main__":
    main()
