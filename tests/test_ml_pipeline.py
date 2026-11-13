
import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import prepare_features


def make_sample():
    return pd.DataFrame({
        "dis_limbs_movement": [2, 1],
        "dis_loss_sensation": [2, 2],
        "dis_deformity": [2, 2],
        "dis_seeing": [1, 2],
        "dis_hearing": [2, 2],
        "dis_speech": [2, 2],
        "dis_understanding": [2, 2],
        "dis_anxiety_mood": [2, 2],
        "dis_unusual_experiences": [2, 2],
        "dis_social_interaction": [2, 2],
        "dis_neuro_blood": [2, 2],
        "sector": [1, 2],
        "uses_public_transport": [1, 2],
        "accesses_public_building": [2, 1],
        "working_before_onset": [1, 2],
        "has_certificate": [2, 1],
        "subsidiary_work": [2, 2],
        "relation_to_head": [1, 5],
        "parents_blood_related": [2, 2],
        "gender": [1, 2],
        "marital_status": [2, 1],
        "education_level": [1, 2],
        "technical_education": [2, 2],
        "religion": [1, 1],
        "social_group": [2, 2],
        "land_possessed": [1, 2],
        "aid_help_received": [2, 2],
        "principal_activity_status": [95, 92],
        "age": [51, 35],
        "household_size": [4, 3],
        "monthly_consumer_exp": [8900, 6000],
        "needs_care": [1, 0],
    })


def test_feature_count():
    X, y = prepare_features(make_sample())

    assert X.shape == (2, 14)
    assert len(y) == 2


def test_nominal_values_are_strings():
    X, _ = prepare_features(make_sample())

    assert X["education_level"].iloc[0] == "1"
    assert X["relation_to_head"].iloc[1] == "5"
    assert X["principal_activity_status"].iloc[0] == "95"


def test_binary_encoding():
    X, _ = prepare_features(make_sample())

    assert X["dis_seeing"].tolist() == [1, 0]
    assert X["uses_public_transport"].tolist() == [1, 0]
    assert X["accesses_public_building"].tolist() == [0, 1]


def test_multiple_disability():
    X, _ = prepare_features(make_sample())

    assert X["multiple_disability"].tolist() == [0, 0]


def test_expenditure_transformation():
    X, _ = prepare_features(make_sample())

    assert X["log_monthly_exp"].iloc[0] == pytest.approx(
        np.log1p(8900)
    )


def test_invalid_binary_code_rejected():
    df = make_sample()
    df.loc[0, "dis_seeing"] = 7

    with pytest.raises(ValueError):
        prepare_features(df)
