import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.model_selection import cross_val_score
from config import CV_FOLDS, N_REPEATS, N_SPLITS
from models import build_stability_random_forest, get_comparison_classifiers

# Evaluate model performance using cross-validation
def evaluate_model(model, X, y):
    """Report cross-validation results for the supplied fitted model.

    The conference workflow passes the configured primary random forest here,
    not the separately tuned forest.
    """
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
    )
    metrics = ['accuracy', 'f1_weighted']

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=metrics,
        return_train_score=True,
        n_jobs=-1
    )

    print("\nOverfitting Analysis")
    print(f"Train Accuracy: {results['train_accuracy'].mean():.2f} (±{results['train_accuracy'].std():.2f})")
    print(f"CV validation accuracy:  {results['test_accuracy'].mean():.2f} (±{results['test_accuracy'].std():.2f})")
    print(f"\nTrain F1: {results['train_f1_weighted'].mean():.2f} (±{results['train_f1_weighted'].std():.2f})")
    print(f"Test F1:  {results['test_f1_weighted'].mean():.2f} (±{results['test_f1_weighted'].std():.2f})")

    return results


def compare_classifiers(X, y, cv=CV_FOLDS):
    """Return cross-validated accuracy statistics for each classifier."""
    results = {}
    for name, classifier in get_comparison_classifiers().items():
        scores = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")
        results[name] = {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
        }
    return results


def analyze_model_stability(X, y, n_iterations=10, cv=CV_FOLDS):
    """Return accuracy and feature-importance variability across runs."""
    accuracies = []
    feature_importances = []
    for iteration in range(n_iterations):
        model = build_stability_random_forest(iteration)
        accuracies.append(cross_val_score(model, X, y, cv=cv).mean())
        model.fit(X, y)
        feature_importances.append(model.feature_importances_)

    return {
        "accuracies": accuracies,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "importance_std": np.std(feature_importances, axis=0),
    }
