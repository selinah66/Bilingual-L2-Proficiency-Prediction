"""Run the historical conference workflow; see README for evaluation limits."""

import argparse

from sklearn.model_selection import train_test_split

from config import (
    CRITICAL_COLUMNS,
    FILEPATH,
    MAX_DEPTH,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)
from data import load_data
from cross_validate_model import evaluate_model
from feature import (
    aggregate_features,
    create_temporal_features,
    prepare_features_and_target,
)
from models import (
    get_feature_importances,
    select_top_features,
    train_random_forest,
    tune_random_forest,
)
from preprocessing import preprocess_pipeline
from reporting import (
    print_feature_analysis,
    print_preprocessing_metrics,
    print_preprocessing_stage_statistics,
    print_regression_path_statistics,
    print_tree_statistics,
    get_feature_description,
)


def main(data_path=FILEPATH, no_plots=False, skip_tuning=False):
    if not no_plots:
        import plots

    def show_plot(name, *args, **kwargs):
        if not no_plots:
            getattr(plots, name)(*args, **kwargs)

    print("Historical exploratory workflow: full-dataset preprocessing and feature")
    print("selection limit held-out and cross-validation performance estimates.")
    # 1. Load and inspect data.
    data = load_data(data_path)
    show_plot("plot_feature_distributions", data)
    show_plot("plot_proficiency_radar", data)

    # 2. Run the original preprocessing.
    preprocessing_steps = preprocess_pipeline(
        data,
        NUMERIC_COLUMNS,
        CRITICAL_COLUMNS,
        return_steps=True,
    )
    processed_data = preprocessing_steps["processed"]
    print_preprocessing_metrics(
        data,
        processed_data,
        NUMERIC_COLUMNS,
        steps=["encoding", "filtering", "imputation", "scaling"],
    )
    show_plot("plot_preprocessing_stages", preprocessing_steps, "IA_FIXATION_COUNT")
    show_plot("plot_descriptive_stats_comparison", preprocessing_steps)
    print_preprocessing_stage_statistics(preprocessing_steps, "IA_FIXATION_COUNT")
    print_regression_path_statistics(preprocessing_steps)

    # 3. Aggregate participant features.
    aggregated_features = aggregate_features(
        processed_data,
        NUMERIC_COLUMNS,
    )
    print(
        "\nFeatures after participant aggregation:",
        aggregated_features.columns.tolist(),
    )

    # 4. Create derived features.
    engineered_features = create_temporal_features(aggregated_features)
    print(
        "\nFeatures after derived feature creation:",
        engineered_features.columns.tolist(),
    )

    # 5. Prepare predictors and labels.
    X, y = prepare_features_and_target(engineered_features, processed_data)
    print("\nPredictors:", X.columns.tolist())
    print("Labels:", y.name)

    # 6. Run the original selection, splitting, training, and evaluation sequence.
    X_selected, selected_features = select_top_features(X, y)
    # get_feature_importances() returns a DataFrame; convert it to a
    # feature-indexed Series for reporting and plotting.
    importances = get_feature_importances(X, y).set_index("Feature")["Importance"]
    print("\nSelected Features (Top 5):")
    for index, feature in enumerate(selected_features, 1):
        print(f"{index}. {feature}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # This score belongs to the separately tuned forest.
    if not skip_tuning:
        tuned_model = tune_random_forest(X_selected, y)
        print(
            "Tuned random forest nominal test score (full-data refit): "
            f"{tuned_model.score(X_test, y_test):.2f}"
        )
    else:
        print("Skipping optional full-dataset hyperparameter search.")

    # These results belong to the configured primary forest, not tuned_model.
    configured_model = train_random_forest(X_train, y_train)
    test_predictions = configured_model.predict(X_test)
    print(
        "Configured random forest train accuracy: "
        f"{configured_model.score(X_train, y_train):.2f}"
    )
    print(
        "Configured random forest test accuracy: "
        f"{configured_model.score(X_test, y_test):.2f}"
    )
    evaluate_model(configured_model, X_selected, y)

    # 7. Generate the relevant reports and figures.
    show_plot("plot_feature_correlation", X, X.columns)
    show_plot("plot_feature_importance_analysis", 
        X,
        y,
        selected_features,
        importances,
    )
    print_feature_analysis(
        X,
        y,
        selected_features,
        importances,
        get_feature_description,
    )
    show_plot("plot_confusion_matrix", y_test, test_predictions)
    show_plot("visualize_decision_trees", 
        configured_model,
        selected_features,
        max_depth=MAX_DEPTH,
        n_trees=3,
    )
    print_tree_statistics(configured_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=FILEPATH,
                        help="Input Excel workbook (default: data/IA_data.xlsx).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Run without opening figures or loading plotting dependencies.")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip the optional 50-candidate search; retain configured-model evaluation.")
    options = parser.parse_args()
    main(data_path=options.data, no_plots=options.no_plots,
         skip_tuning=options.skip_tuning)
