import pandas as pd

PARTICIPANT_COLUMN = "RECORDING_SESSION_LABEL"
TRIAL_COLUMN = "TRIAL_INDEX"
INTEREST_AREA_COLUMN = "IA_ID"
KEY_COLUMNS = (PARTICIPANT_COLUMN, TRIAL_COLUMN, INTEREST_AREA_COLUMN)
REQUIRED_COLUMNS = KEY_COLUMNS


def validate_required_columns(df, required_columns=REQUIRED_COLUMNS):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def check_duplicate_keys(df, key_columns=KEY_COLUMNS):
    validate_required_columns(df, key_columns)
    duplicate_rows = df[df.duplicated(subset=list(key_columns), keep=False)]
    if not duplicate_rows.empty:
        duplicate_keys = duplicate_rows.loc[:, list(key_columns)].drop_duplicates()
        raise ValueError(
            f"Duplicate keys found for {list(key_columns)}: "
            f"{duplicate_keys.to_dict(orient='records')}"
        )


def summarize_data(df):
    summary = {
        "interest_area_records": len(df),
        "participants": df[PARTICIPANT_COLUMN].nunique(),
        "participant_trial_combinations": (
            df[[PARTICIPANT_COLUMN, TRIAL_COLUMN]].drop_duplicates().shape[0]
        ),
        "distinct_interest_area_ids": df[INTEREST_AREA_COLUMN].nunique(),
    }
    print(
        "Summary: "
        f"{summary['interest_area_records']} interest-area records, "
        f"{summary['participants']} participants, "
        f"{summary['participant_trial_combinations']} participant-trial combinations, "
        f"{summary['distinct_interest_area_ids']} distinct interest-area IDs"
    )
    return summary


def load_data(file_path):
    df = pd.read_excel(file_path)
    validate_required_columns(df)
    check_duplicate_keys(df)
    print("Initial shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    summarize_data(df)
    return df