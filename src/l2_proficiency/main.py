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
from plots import (
    plot_confusion_matrix,
    plot_descriptive_stats_comparison,
    plot_feature_correlation,
    plot_feature_distributions,
    plot_feature_importance_analysis,
    plot_preprocessing_stages,
    plot_proficiency_radar,
    visualize_decision_trees,
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


def main():
    # 1. Load and inspect data.
    data = load_data(FILEPATH)
    plot_feature_distributions(data)
    plot_proficiency_radar(data)

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
    plot_preprocessing_stages(preprocessing_steps, "IA_FIXATION_COUNT")
    plot_descriptive_stats_comparison(preprocessing_steps)
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
    tuned_model = tune_random_forest(X_selected, y)
    print(
        "Tuned random forest test score: "
        f"{tuned_model.score(X_test, y_test):.2f}"
    )

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
    plot_feature_correlation(X, X.columns)
    plot_feature_importance_analysis(
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
    plot_confusion_matrix(y_test, test_predictions)
    visualize_decision_trees(
        configured_model,
        selected_features,
        max_depth=MAX_DEPTH,
        n_trees=3,
    )
    print_tree_statistics(configured_model)


if __name__ == "__main__":
    main()
