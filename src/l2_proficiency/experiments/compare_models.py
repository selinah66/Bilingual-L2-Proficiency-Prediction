from config import CRITICAL_COLUMNS, FILEPATH, NUMERIC_COLUMNS
from data import load_data
from cross_validate_model import analyze_model_stability, compare_classifiers
from feature import (
    aggregate_features,
    create_temporal_features,
    prepare_features_and_target,
)
from plots import plot_classifier_comparison, plot_model_stability
from preprocessing import preprocess_pipeline


def load_features_and_target():
    """Build experiment features without importing application state."""
    raw_data = load_data(FILEPATH)
    steps = preprocess_pipeline(
        raw_data,
        NUMERIC_COLUMNS,
        CRITICAL_COLUMNS,
        return_steps=True,
    )
    grouped = aggregate_features(steps["processed"], NUMERIC_COLUMNS)
    engineered = create_temporal_features(grouped)
    return prepare_features_and_target(engineered, steps["processed"])


def main():
    X, y = load_features_and_target()

    comparison = compare_classifiers(X, y)
    print("Classifier Comparison")
    for name, metrics in comparison.items():
        print(
            f"{name}: {metrics['mean_accuracy']:.3f} "
            f"+/- {metrics['std_accuracy']:.3f}"
        )
    plot_classifier_comparison(comparison)

    stability = analyze_model_stability(X, y)
    print("\nModel Stability Analysis")
    print(
        f"Mean Accuracy: {stability['mean_accuracy']:.3f} "
        f"+/- {stability['std_accuracy']:.3f}"
    )
    plot_model_stability(X.columns, stability)


if __name__ == "__main__":
    main()
