"""Historical preprocessing workflow and reusable transformation utilities.

The conference workflow in function `preprocess_pipeline` preserves the original
order: categorical values are encoded first, rows are filtered, missing
numeric values are imputed, outliers are clipped using fitted bounds, and
numeric columns are robust-scaled.  The workflow fits those transformations
on the data passed to it.  :func:`apply_transformations` is an additional
utility for applying a previously fitted transformation state to new data;
it is not currently called by the conference workflow.
"""

import numpy as np
from scipy.stats import iqr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from config import MISSINGNESS_FILTER_GROUPS, PROTECTED_COLUMNS

def filter_rows(df, critical_cols=None, regex_pattern=r'^[a-zA-Z]', verbose=True):
    """Apply deterministic row filters without learning dataset parameters."""
    # Language-based filtering
    df = df[df['IA_LABEL'].str.match(regex_pattern, na=False)]
    if verbose:
        print(f"Filtered English rows: {df.shape[0]}")

    # Null value filtering (skip protected columns)
    for cols in MISSINGNESS_FILTER_GROUPS:
        filtered_cols = [c for c in cols if c not in PROTECTED_COLUMNS]
        if not filtered_cols:
            continue

        before = len(df)
        df = df[~df[filtered_cols].isnull().all(axis=1)]
        if verbose:
            print(f"Removed {before - len(df)} rows with nulls in: {filtered_cols}")

    # Critical columns handling
    if critical_cols:
        initial = len(df)
        df = df.dropna(subset=critical_cols)
        if verbose:
            print(f"Removed {initial - len(df)} rows with missing critical cols")

    return df

def fit_transformations(df, numeric_cols, cat_cols=('L2 PROFICIENCY', 'CONDITION'),
                        outlier_threshold=3):
    """Fit learned transformations and apply them to ``df``."""
    transformed = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in transformed.columns:
            encoder = LabelEncoder()
            transformed[col] = encoder.fit_transform(transformed[col])
            encoders[col] = encoder

    numeric_columns = [
        col for col in numeric_cols
        if col in transformed.select_dtypes(include=np.number).columns
    ]
    protected = [col for col in numeric_columns if col in PROTECTED_COLUMNS]
    regular = [col for col in numeric_columns if col not in PROTECTED_COLUMNS]
    imputers = {}
    for columns, strategy in ((protected, 'median'), (regular, 'mean')):
        if columns:
            imputer = SimpleImputer(strategy=strategy)
            transformed[columns] = imputer.fit_transform(transformed[columns])
            imputers[strategy] = imputer

    clip_bounds = {}
    for col in numeric_columns:
        q1 = transformed[col].quantile(0.25)
        q3 = transformed[col].quantile(0.75)
        spread = iqr(transformed[col])
        clip_bounds[col] = (
            q1 - (outlier_threshold * spread),
            q3 + (outlier_threshold * spread),
        )
        transformed[col] = transformed[col].clip(*clip_bounds[col])

    scaler = RobustScaler() if numeric_columns else None
    if numeric_columns:
        transformed[numeric_columns] = scaler.fit_transform(
            transformed[numeric_columns]
        )
    state = {
        'encoders': encoders,
        'imputers': imputers,
        'numeric_columns': numeric_columns,
        'clip_bounds': clip_bounds,
        'scaler': scaler,
    }
    return transformed, state


def apply_transformations(df, state):
    """Apply fitted transformations without learning parameters from ``df``."""
    transformed = df.copy()
    for col, encoder in state['encoders'].items():
        if col in transformed.columns:
            transformed[col] = encoder.transform(transformed[col])

    for strategy, imputer in state['imputers'].items():
        columns = [
            col for col in state['numeric_columns']
            if col in transformed.columns and (
                (strategy == 'median' and col in PROTECTED_COLUMNS)
                or (strategy == 'mean' and col not in PROTECTED_COLUMNS)
            )
        ]
        if columns:
            transformed[columns] = imputer.transform(transformed[columns])

    present_numeric = [
        col for col in state['numeric_columns'] if col in transformed.columns
    ]
    for col in present_numeric:
        transformed[col] = transformed[col].clip(*state['clip_bounds'][col])
    if present_numeric:
        transformed[present_numeric] = state['scaler'].transform(
            transformed[present_numeric]
        )
    return transformed


def preprocess_pipeline(df, numeric_cols, critical_cols, return_steps=False):
    """Encode, filter, impute, clip, and robust-scale in historical order.

    The returned ``"imputed"`` and ``"processed"`` entries intentionally point
    to the same fully transformed DataFrame. They are retained as historical
    stage names, not distinct snapshots of the imputation and scaling steps.
    """
    encoded, _ = fit_transformations(
        df.copy(),
        numeric_cols=[],
    )
    df_filtered = filter_rows(encoded, critical_cols=critical_cols)
    df_processed, _ = fit_transformations(
        df_filtered,
        numeric_cols=numeric_cols,
        cat_cols=(),
    )

    if return_steps:
        return {
            'filtered': df_filtered,
            'imputed': df_processed,
            'processed': df_processed
        }
    else:
        return df_processed
    
