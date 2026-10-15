
def print_preprocessing_metrics(df_before, df_after, numeric_cols, steps):
    """Print null, class-distribution, and outlier preprocessing metrics."""
    print("\nPreprocessing Metrics")

    nulls_before = df_before.isnull().sum().sum()
    nulls_after = df_after.isnull().sum().sum()
    before_rate = nulls_before / df_before.size if df_before.size else 0
    after_rate = nulls_after / df_after.size if df_after.size else 0
    reduction_rate = (
        (nulls_before - nulls_after) / nulls_before
        if nulls_before
        else 0
    )
    print("\nNull Values:")
    print(f"Initial: {nulls_before} ({before_rate:.1%})")
    print(f"Final: {nulls_after} ({after_rate:.1%})")
    print(f"Reduction: {reduction_rate:.1%}")

    if "L2 PROFICIENCY" in df_after.columns:
        print("\nClass Distribution:")
        print("Before preprocessing:", df_before["L2 PROFICIENCY"].value_counts(normalize=True))
        print("After preprocessing:", df_after["L2 PROFICIENCY"].value_counts(normalize=True))

    if "scaling" in steps:
        print("\nOutlier Handling:")
        total_outliers = 0
        for col in numeric_cols:
            if col in df_before.columns:
                q1 = df_before[col].quantile(0.25)
                q3 = df_before[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3*iqr
                upper = q3 + 3*iqr
                outliers = ((df_before[col] < lower) | (df_before[col] > upper)).sum()
                total_outliers += outliers
        outlier_rate = total_outliers / len(df_before) if len(df_before) else 0
        print(
            "Flagged feature values across all categories: "
            f"{total_outliers} ({outlier_rate:.1%} of rows; "
            "one row may contribute multiple flags)"
        )


def print_preprocessing_stage_statistics(processing_steps, column):
    """Print descriptive statistics for a column at each processing stage."""
    stages = [
        ("filtered", "After Filtering/Cleaning"),
        ("imputed", "After Imputation"),
        ("processed", "After Processing"),
    ]
    print("\nPreprocessing Stage Summary Statistics:")
    for stage_name, label in stages:
        if column not in processing_steps[stage_name].columns:
            continue
        data = processing_steps[stage_name][column].dropna()
        print(f"\n{label}:")
        print(f"Mean: {data.mean():.2f}")
        print(f"Std: {data.std():.2f}")
        print(f"Min: {data.min():.2f}")
        print(f"Max: {data.max():.2f}")
        print(f"Count: {len(data)}")


def print_feature_analysis(X, y, selected_features, importances, description):
    """Print feature importance, selection, and target-correlation details."""
    correlations = X[selected_features].apply(lambda column: column.corr(y))
    print("\nFeature Analysis")
    print("\nAll Available Features:")
    for index, (feature, importance) in enumerate(
        importances.sort_values(ascending=False).items(), 1
    ):
        status = "SELECTED" if feature in selected_features else ""
        print(f"{index}. {feature}: {importance:.3f} {status}")

    print("\nSelected Features for Training:")
    for index, feature in enumerate(selected_features, 1):
        print(f"{index}. {feature}")
        print(f"   Importance: {importances[feature]:.3f}")
        print(f"   Correlation with target: {correlations[feature]:.3f}")
        print(f"   Description: {description(feature)}")


def get_feature_description(feature):
    descriptions = {
        "IA_FIRST_FIXATION_DURATION": "Duration of the first fixation on a word",
        "IA_REGRESSION_PATH_DURATION": "Time spent in regression movements",
        "IA_DWELL_TIME": "Total time spent looking at a word",
        "IA_FIXATION_COUNT": "Number of fixations on a word",
        "IA_FIRST_SACCADE_AMPLITUDE": "Distance of the first eye movement",
        "regression_dwell_ratio": "Ratio of regression time to total dwell time",
        "fixation_density": "Number of fixations per unit time",
        "log_first_fixation": "Log-transformed first fixation duration",
        "saccade_speed": "First-saccade amplitude divided by first-fixation duration",
    }
    return descriptions.get(feature, "No description available")


def print_tree_statistics(model):
    """Print node, depth, and leaf counts for each tree in a forest."""
    for index, tree in enumerate(model.estimators_, 1):
        n_leaves = sum(
            1
            for node in range(tree.tree_.node_count)
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]
        )
        print(f"\nTree {index} Statistics:")
        print(f"Number of nodes: {tree.tree_.node_count}")
        print(f"Max depth: {tree.tree_.max_depth}")
        print(f"Number of leaves: {n_leaves}")


def print_regression_path_statistics(processing_steps):
    """Print stage statistics, noting that processed values are robust-scaled."""
    stages = [
        ("filtered", "After Filtering/Cleaning"),
        ("imputed", "After Imputation"),
        ("processed", "After Processing"),
    ]
    print("\nRegression Path Duration Statistics")
    for stage_name, label in stages:
        data = processing_steps[stage_name]["IA_REGRESSION_PATH_DURATION"].dropna()
        print(f"\n{label}:")
        unit = " (robust-scaled units)" if stage_name == "processed" else " ms"
        print(f"Mean: {data.mean():.2f}{unit}")
        print(f"Median: {data.median():.2f}{unit}")
        print(f"Std: {data.std():.2f}{unit}")
        print(f"Min: {data.min():.2f}{unit}")
        print(f"Max: {data.max():.2f}{unit}")
        print(f"Q1: {data.quantile(0.25):.2f}{unit}")
        print(f"Q3: {data.quantile(0.75):.2f}{unit}")
        print(f"Count: {len(data)}")
        iqr = data.quantile(0.75) - data.quantile(0.25)
        print(f"IQR: {iqr:.2f}{unit}")
        print(f"Range: {data.max() - data.min():.2f}{unit}")