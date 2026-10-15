import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import (
    MAX_FEATURES,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_ESTIMATORS,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
)

def select_top_features(X, y, n_features=5):
    """Select features using label-dependent model importance."""
    importance = get_feature_importances(X, y)
    selected_features = importance.head(n_features)["Feature"].tolist()
    return X[selected_features], selected_features


def get_feature_importances(X, y):
    """Fit the selector model and return feature importances."""
    selector = RandomForestClassifier()
    selector.fit(X, y)
    return pd.DataFrame({
        "Feature": X.columns,
        "Importance": selector.feature_importances_,
    }).sort_values("Importance", ascending=False)


def _configured_random_forest():
    """Build the primary forest from the fixed configuration settings."""
    return RandomForestClassifier(
        class_weight="balanced",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def train_random_forest(X_train, y_train):
    """Fit the configured forest used for the primary reported result."""
    return _configured_random_forest().fit(X_train, y_train)


def tune_random_forest(X, y):
    """Fit a separately tuned forest selected by repeated cross-validation."""
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [4, 8],
        "max_features": ["sqrt", 0.3, 0.5],
        "bootstrap": [True],
        "max_leaf_nodes": [50, 100],
    }
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


def get_comparison_classifiers():
    """Build the classifiers used by the model comparison experiment."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    }


def build_stability_random_forest(iteration):
    """Build a separate forest for the stability-analysis runs."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE + iteration,
    )
