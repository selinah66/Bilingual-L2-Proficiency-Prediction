"""Participant-level aggregation, derived features, and model preparation."""

import numpy as np
import pandas as pd

from config import TARGET_COLUMN


def create_temporal_features(df):
    """Create derived ratios and transforms from aggregated eye-tracking columns.

    The formulas operate on whatever scale is supplied. In the conference
    workflow, ``df`` comes from the robust-scaled processed data, so these
    derived values are ratios/transforms of scaled inputs rather than physical
    milliseconds or amplitudes. Existing column names are retained for
    compatibility:

    * ``regression_dwell_ratio`` = regression-path duration / dwell time.
    * ``fixation_density`` = fixation count / dwell time.
    * ``log_first_fixation`` = ``log1p(first fixation duration)``.
    * ``saccade_speed`` = first-saccade amplitude / first-fixation duration;
      it is an amplitude-to-duration ratio, not a direct measurement of
      physical saccade speed.
    """
    df = df.copy()

    df['regression_dwell_ratio'] = \
        df['IA_REGRESSION_PATH_DURATION'] / (df['IA_DWELL_TIME'] + 1e-6)

    df['fixation_density'] = \
        df['IA_FIXATION_COUNT'] / (df['IA_DWELL_TIME'] + 1e-6)

    df['log_first_fixation'] = np.log1p(df['IA_FIRST_FIXATION_DURATION'])

    df['saccade_speed'] = \
        df['IA_FIRST_SACCADE_AMPLITUDE'] / (df['IA_FIRST_FIXATION_DURATION'] + 1e-6)

    return df


def aggregate_features(
    df,
    available_features,
    group_col="RECORDING_SESSION_LABEL",
):
    """Aggregate the fixed five-feature set by participant/session.

    ``available_features`` is the broader list supplied by the workflow for
    schema validation. The historical aggregation scope remains fixed to the
    five eye-tracking measurements listed in ``agg_dict`` below, each reduced
    with ``mean``.
    """
    missing = [feature for feature in available_features if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing features for aggregation: {missing}")

    agg_dict = {
        "IA_FIRST_FIXATION_DURATION": ["mean"],
        "IA_REGRESSION_PATH_DURATION": ["mean"],
        "IA_DWELL_TIME": ["mean"],
        "IA_FIXATION_COUNT": ["mean"],
        "IA_FIRST_SACCADE_AMPLITUDE": ["mean"],
    }
    fixed_scope_missing = [
        feature for feature in agg_dict if feature not in df.columns
    ]
    if fixed_scope_missing:
        raise KeyError(
            f"Missing fixed aggregation features: {fixed_scope_missing}"
        )

    if group_col not in df.columns:
        raise KeyError(f"Grouping column is missing from data: {group_col}")

    if "CONDITION" in df.columns:
        # Attach the first condition observed for each participant/session;
        # CONDITION is not part of the grouping key.
        condition_map = df.groupby(group_col)["CONDITION"].first()
        grouped = df.groupby(group_col).agg(agg_dict)
        grouped.columns = [col for col, _ in grouped.columns]
        grouped = grouped.reset_index()
        grouped = grouped.merge(condition_map.reset_index(), on=group_col)
    else:
        grouped = df.groupby(group_col).agg(agg_dict)
        grouped.columns = [col for col, _ in grouped.columns]
        grouped = grouped.reset_index()

    return grouped


def prepare_features_and_target(
    grouped_df,
    cleaned_df,
    group_col="RECORDING_SESSION_LABEL",
    target_col=TARGET_COLUMN,
):
    """Merge participant features with one target label per participant/session."""

    target = cleaned_df.drop_duplicates(group_col).set_index(group_col)[target_col]
    data = grouped_df.merge(target, left_on=group_col, right_index=True)

    # Historical weighted row reordering: sample every row without
    # replacement, rather than creating or balancing class counts.
    class_counts = data[target_col].value_counts()
    if class_counts.max() / class_counts.min() > 2:
        print(f"Applying weighted row reordering (ratio: {class_counts.max() / class_counts.min():.1f}:1)")
        data = data.sample(frac=1, weights=data[target_col].map({k: 1 / v for k, v in class_counts.items()}))

    X = data.drop([group_col, target_col], axis=1)
    y = data[target_col]

    return X, y
