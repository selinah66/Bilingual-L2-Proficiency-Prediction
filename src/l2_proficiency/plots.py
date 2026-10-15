"""Plotting helpers grouped by the stage of analysis they explain.

The plotting functions expect the historical workflow's data conventions:

* Exploratory DataFrames use string ``L2 PROFICIENCY`` labels ``"L"`` and
  ``"H"``. Numeric feature columns are plotted on their existing scale unless
  a function explicitly says otherwise.
* Preprocessing-stage DataFrames are supplied in the ``processing_steps``
  mapping under the keys ``"filtered"``, ``"imputed"``, and ``"processed"``.
  ``"imputed"`` and ``"processed"`` currently reference the same fully
  transformed, robust-scaled DataFrame; they are not distinct snapshots.
* Model-result plots accept the labels or predictions produced by the model.
  Confusion-matrix labels may therefore be strings or numeric encodings, but
  ``y_true`` and ``y_pred`` must use the same label space.
* Feature-importance inputs are aligned by feature name when supplied as a
  pandas Series or mapping. A sequence is interpreted in ``X.columns`` order.
  The importance values should be numeric and represent one importance value
  per feature; the plotting function does not normalize them.

This module only creates figures. Descriptive statistics and textual summaries
are in ``reporting.py``.
"""

import io
import shutil
import textwrap

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz, plot_tree

from config import MAX_DEPTH


# Data exploration
def split_by_proficiency(df: pd.DataFrame):
    """Return low/high subsets using string labels ``"L"`` and ``"H"``.

    ``df`` must contain ``L2 PROFICIENCY`` with the original string labels;
    encoded target values such as ``0`` and ``1`` are not converted here.
    """
    low_prof = df[df["L2 PROFICIENCY"] == "L"]
    high_prof = df[df["L2 PROFICIENCY"] == "H"]
    return low_prof, high_prof


def _available_comparison_features(df):
    candidates = [
        "IA_DWELL_TIME",
        "IA_FIXATION_COUNT",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIRST_SACCADE_AMPLITUDE",
        "fixation_density",
    ]
    return [feature for feature in candidates if feature in df.columns]


def plot_feature_distributions(df, features=None):
    """Plot feature distributions by proficiency group.

    ``df`` must contain string ``L2 PROFICIENCY`` labels (``"L"``/``"H"``).
    Feature columns should be numeric and remain on their input scale.
    """
    features = features or _available_comparison_features(df)[:2]
    if not features:
        raise ValueError("No supported features are available to plot.")

    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 6))
    axes = np.atleast_1d(axes)
    for axis, feature in zip(axes, features):
        sns.violinplot(
            x="L2 PROFICIENCY",
            y=feature,
            data=df,
            hue="L2 PROFICIENCY",
            legend=False,
            ax=axis,
        )
        axis.set_title(feature.replace("IA_", "").replace("_", " ").title())
        axis.set_xlabel("L2 Proficiency")
    fig.suptitle("Feature Distributions by L2 Proficiency Group")
    fig.tight_layout()
    plt.show()


def plot_proficiency_radar(df, features=None):
    """Plot min-max-normalized feature means for ``"L"`` and ``"H"`` groups.

    ``df`` must contain string proficiency labels and numeric feature columns.
    Each feature is normalized independently to the inclusive ``[0, 1]``
    range using the complete DataFrame before group means are calculated.
    """
    from matplotlib.projections import register_projection
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.spines import Spine
    from matplotlib.path import Path
    from matplotlib.transforms import Affine2D
    from matplotlib.patches import RegularPolygon

    features = features or _available_comparison_features(df)
    if not features:
        raise ValueError("No supported features are available to plot.")

    theta = np.linspace(0, 2 * np.pi, len(features), endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    line.set_data(np.append(x, x[0]), np.append(y, y[0]))
            return lines

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), len(features), radius=0.5, edgecolor="k")

        def _gen_axes_spines(self):
            spine = Spine(
                axes=self,
                spine_type="circle",
                path=Path.unit_regular_polygon(len(features)),
            )
            spine.set_transform(
                Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
            )
            return {"polar": spine}

    register_projection(RadarAxes)
    low, high = split_by_proficiency(df)
    normalized = {
        feature: (df[feature] - df[feature].min())
        / (df[feature].max() - df[feature].min() or 1)
        for feature in features
    }
    low_means = [normalized[feature].loc[low.index].mean() for feature in features]
    high_means = [normalized[feature].loc[high.index].mean() for feature in features]
    labels = [feature.replace("IA_", "").replace("_", " ").title() for feature in features]

    fig, axis = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "radar"})
    axis.plot(theta, low_means, color="blue", label="Low L2 Proficiency")
    axis.fill(theta, low_means, color="blue", alpha=0.25)
    axis.plot(theta, high_means, color="red", label="High L2 Proficiency")
    axis.fill(theta, high_means, color="red", alpha=0.25)
    axis.set_varlabels(labels)
    axis.set_theta_direction(-1)
    axis.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    plt.show()

# Preprocessing diagnostics
def plot_preprocessing_stages(processing_steps, column_to_plot):
    """Plot KDEs for a column across filtered, imputed, and processed stages.

    ``processing_steps`` must contain the three historical stage keys and
    DataFrames. ``column_to_plot`` is numeric. Filtered values retain their
    source scale; imputed and processed values are robust-scaled in the
    current workflow, so no single physical unit applies to every curve.
    """
    plt.figure(figsize=(12, 7))
    stages_to_plot = [
        ("filtered", "green", "-", "After Filtering/Cleaning"),
        ("imputed", "orange", "-", "After Imputation"),
        ("processed", "purple", "-", "After Processing"),
    ]

    for stage_name, color, style, label in stages_to_plot:
        if column_to_plot in processing_steps[stage_name].columns:
            data = processing_steps[stage_name][column_to_plot].dropna()
            sns.kdeplot(data, label=label, color=color, linestyle=style)

    plt.title(f"{column_to_plot} Distribution Through Processing Stages")
    plt.xlabel(f"{column_to_plot} (stage-dependent scale)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_descriptive_stats_comparison(processing_steps):
    """Compare regression-path-duration stages on raw and log axes.

    ``processing_steps`` must provide the historical stage keys and numeric
    ``IA_REGRESSION_PATH_DURATION`` columns where a stage is plotted. The
    filtered stage is in source units; imputed and processed entries may be
    robust-scaled and therefore are not necessarily milliseconds. The log
    panel requires positive values.
    """
    stages = [
        ("filtered", "After Filtering/Cleaning"),
        ("imputed", "After Imputation"),
        ("processed", "After Processing"),
    ]
    box_data = []
    labels = []
    for stage_name, stage_label in stages:
        df = processing_steps[stage_name]
        if "IA_REGRESSION_PATH_DURATION" in df.columns:
            data = df["IA_REGRESSION_PATH_DURATION"].dropna()
            box_data.append(data)
            labels.append(stage_label)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    box1 = ax1.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    ax1.set_title("Regression Path Duration (Stage Values)")
    ax1.set_ylabel("Stage-dependent value")
    ax1.grid(True, alpha=0.3)
    box2 = ax2.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    ax2.set_yscale("log")
    ax2.set_title("Regression Path Duration (Log Scale)")
    ax2.set_ylabel("Stage-dependent value - Log Scale")
    ax2.grid(True, alpha=0.3)

    colors = ["lightblue", "lightgreen", "lightcoral"]
    for box in [box1, box2]:
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    plt.tight_layout()
    plt.show()

# Model results
def plot_confusion_matrix(y_true, y_pred):
    """Plot counts for matching true/predicted labels.

    ``y_true`` and ``y_pred`` must use the same labels and encoding. The
    heatmap displays raw counts, not normalized proportions.
    """
    matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()


def plot_classifier_comparison(results):
    """Plot cross-validation mean accuracy with its standard deviation.

    ``results`` maps each classifier name to numeric ``mean_accuracy`` and
    ``std_accuracy`` values, on the accuracy scale ``[0, 1]``.
    """
    names = list(results)
    means = [results[name]["mean_accuracy"] for name in names]
    stds = [results[name]["std_accuracy"] for name in names]
    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.title("Classifier Performance Comparison")
    plt.ylabel("Mean Accuracy (Cross-Validation)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_model_stability(feature_names, stability):
    """Plot per-feature importance standard deviations across stability runs.

    ``feature_names`` must be in the same order as the numeric
    ``stability["importance_std"]`` sequence. The values are standard
    deviations of importance scores, not normalized feature values.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, stability["importance_std"])
    plt.title("Feature Importance Stability")
    plt.xlabel("Features")
    plt.ylabel("Standard Deviation of Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Interpretation
def plot_feature_correlation(X, selected_features):
    """Plot Pearson correlations among selected numeric feature columns.

    ``X`` must contain numeric columns for every name in ``selected_features``.
    Correlations are on the standard ``[-1, 1]`` scale.
    """
    missing_features = [feature for feature in selected_features if feature not in X.columns]
    if missing_features:
        raise ValueError(f"Features missing from X: {missing_features}")

    non_numeric = X[selected_features].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features selected: {non_numeric}")

    corr_matrix = X[selected_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        square=True,
        mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
    )
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_feature_importance_analysis(X, y, selected_features, importances):
    """Compare feature importances with selected features and target correlation.

    ``X`` must contain numeric predictor columns and ``y`` must be numeric or
    coercible to numeric for Pearson correlation. ``selected_features`` must
    be a subset of ``X.columns``. ``importances`` must provide one numeric
    value per ``X`` column: a Series/mapping is aligned by feature name, while
    a sequence is interpreted in ``X.columns`` order. Values are plotted as
    supplied and are not normalized.
    """
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3)

    all_importances = pd.Series(importances, index=X.columns if not hasattr(importances, "items") else None)
    if hasattr(importances, "items"):
        all_importances = pd.Series(importances).reindex(X.columns)
    if all_importances.isna().any():
        raise ValueError("Importances must contain every feature in X.")

    ax1 = fig.add_subplot(gs[0, 0])
    all_importances.sort_values().plot(kind="barh", ax=ax1, color="lightblue")
    ax1.set_title("Available Features by\nImportance Score")
    ax1.set_xlabel("Importance Score")
    ax1.set_ylabel("Features")
    ax1.tick_params(axis="y", rotation=45)

    ax2 = fig.add_subplot(gs[0, 1])
    selected_importances = all_importances[selected_features]
    selected_importances.sort_values().plot(kind="barh", ax=ax2, color="green")
    ax2.set_title("Selected Features by\nImportance Score")
    ax2.set_xlabel("Importance Score")
    ax2.set_ylabel("Features")
    ax2.tick_params(axis="y", rotation=50)

    ax3 = fig.add_subplot(gs[0, 2])
    correlations = X[selected_features].apply(lambda column: column.corr(y))
    correlations.sort_values().plot(kind="barh", ax=ax3, color="purple")
    ax3.set_title("Selected Features by\nCorrelation with Target")
    ax3.set_xlabel("Correlation Coefficient")
    ax3.set_ylabel("Features")
    ax3.tick_params(axis="y", rotation=50)

    def process_label(label):
        cleaned = label.replace("IA_", "").replace("_", " ")
        wrapped = textwrap.wrap(cleaned, width=15)
        return "\n".join(wrapped)

    for ax in [ax1, ax2, ax3]:
        labels = [label.get_text() for label in ax.get_yticklabels()]
        new_labels = [process_label(label) for label in labels]
        ax.set_yticklabels(new_labels, rotation=50)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.5, wspace=1.2)
    plt.suptitle(
        "Feature Selection: Comparison of Feature Importance, Target Correlations",
        fontsize=16,
        y=1.05,
    )
    plt.show()


def visualize_decision_trees(model, feature_names, max_depth=3, n_trees=3):
    """Render selected trees from a fitted random forest.

    ``feature_names`` must match the model's training-column order.
    ``n_trees`` limits the number rendered. ``max_depth`` is retained for the
    public API, while the historical Graphviz export uses ``MAX_DEPTH``.
    Current preprocessing uses ``LabelEncoder`` with ``H=0`` and ``L=1``.
    The display labels therefore follow that encoded order as ``High``, then
    ``Low``. This changes annotations only; it does not alter the fitted model
    or predictions.
    """
    for i in range(min(n_trees, len(model.estimators_))):
        tree = model.estimators_[i]
        if shutil.which("dot") is None:
            plt.figure(figsize=(15, 15))
            plot_tree(
                tree,
                feature_names=feature_names,
                class_names=["High", "Low"],
                filled=True,
                rounded=True,
                max_depth=MAX_DEPTH,
            )
            plt.title(f"Decision Tree {i + 1}")
            plt.tight_layout()
            plt.show()
            continue

        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=["High", "Low"],
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=MAX_DEPTH,
        )
        graph = graphviz.Source(dot_data)
        png_bytes = graph.pipe(format="png")
        image = Image.open(io.BytesIO(png_bytes))
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Decision Tree {i + 1}")
        plt.show()
