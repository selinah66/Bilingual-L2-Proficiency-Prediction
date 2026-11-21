# Predicting English L2 Proficiency from Eye-Tracking

An exploratory machine-learning study of reading behavior in Mandarin Chinese–English bilinguals, using a published eye-tracking dataset.

**Conference:** California NeuroTechnology Conference, April 26, 2025  
**Team:** Selina Hui (Project Lead), Lily Meihui Wu, Asha Dukkipati, and Sienna Jackson

[Conference poster](docs/BPET_Poster.pdf) · [Supplemental figures](docs/supplementalFig.pdf) · [Methods](docs/methods.md) · [Research background](docs/literatureReview.md)

## At a glance

| | Summary |
| --- | --- |
| Research question | Can eye-movement features distinguish the source study’s low- and high-English-proficiency groups? |
| Data | Published recordings from 80 Mandarin Chinese–English bilinguals; 40 participants per group |
| Modeling unit | One aggregated feature vector per participant |
| Approach | Data inspection, preprocessing, participant-level feature engineering, and random-forest classification |
| Conference-reported result | 69% accuracy and 0.69 macro F1 on an evaluation subset of 32 participants |
| Baseline | 50% majority-class accuracy for the balanced evaluation subset |
| Interpretation | Exploratory secondary analysis; historical data leakage limits the reported performance estimate |

**Evaluation caveat:** Although participants were separated between training and evaluation sets, preprocessing and supervised feature selection used the full dataset, and one model was tuned on the full dataset before evaluation. The reported scores therefore do not provide an unbiased estimate of performance on unseen participants.

## Motivation

Reading in a second language draws on language experience and processing resources. Eye-tracking measures provide a way to examine reading behavior, including how long readers fixate and whether they revisit text. This project explores whether these measurements can distinguish proficiency groups in an existing bilingual reading dataset.

The task predicts the source study’s binary proficiency labels; it does not estimate a continuous proficiency score. Eye movements provide indirect measures of processing demands and do not directly measure comprehension or establish a cognitive mechanism.

## Project leadership

As project lead, **Selina Hui** initiated the project and contributed to data inspection, feature engineering, model experimentation, visualization, interpretation, and poster preparation. The analysis was collaborative; individual contributions are listed below. Our team analyzed published data and did not collect the original eye-tracking recordings.

## Approach

1. **Inspect the data.** Review participant and trial identifiers, proficiency labels, missing values, and eye-movement measurements.
2. **Prepare reading records.** Retain English interest-area records and apply the historical missing-value, imputation, clipping, and scaling rules.
3. **Build participant features.** Aggregate measurements into one row per participant and construct derived predictors.
4. **Explore and classify.** Examine group differences, select features, and train random-forest models.
5. **Evaluate and communicate.** Summarize classification results and document the analysis in a conference poster and supplemental figures.

Features include participant means of first-fixation duration, regression-path duration, total dwell time, fixation count, and first-saccade amplitude, plus derived ratios and a logarithmic first-fixation feature. Exact formulas and their order relative to scaling matter for interpretation; see the [methods](docs/methods.md).

## Results

![Dwell time and fixation count by proficiency group](figures/readingMeasuresbyProf.png)

*Historical figure: distributions of repeated interest-area measurements, rather than independent participant samples. L = low proficiency; H = high proficiency. Dwell time is shown in milliseconds.*

The conference poster reports the following results for an evaluation subset containing 16 high- and 16 low-proficiency participants:

| Metric | Reported value |
| --- | ---: |
| Accuracy | 0.69 |
| Macro F1 | 0.69 |
| Weighted F1 | 0.69 |
| Majority-class baseline accuracy | 0.50 |

These values reproduce the conference report; they are not newly recomputed results. The historical workflow used a stratified 60/40 participant split with `random_state=42`. Full-dataset preprocessing and model-development steps limit interpretation of that split, as described above.

Exploratory comparisons associated lower proficiency with longer dwell times, more fixations, and greater regression-related activity. These are sample-specific associations and do not establish causation. Repeated reading records are not independent participants.

For per-class metrics, preprocessing counts, experimental configurations, and figure descriptions, see [methods](docs/methods.md).

## Repository guide

| Location | Contents |
| --- | --- |
| `src/l2_proficiency/` | Python analysis workflow: data loading, preprocessing, feature engineering, modeling, evaluation, and plots |
| `data/` | Published source workbooks and source-study R scripts; see [data guide](data/README.md) |
| `figures/` | Saved historical figures |
| `scripts/check_data.py` | Data-schema, sample-count, finite-feature, and participant-split audit |
| `docs/` | Conference materials and research documentation |
| `archive/` | Earlier code-switching/proficiency experiments and learning notes |

The project initially explored code-switching prediction before shifting to English L2 proficiency classification. Archived work records that development; the proficiency workflow is the main analysis presented here.

## Run the analysis

From the repository root, create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_data.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Run the configured model and cross-validation without figures or the optional hyperparameter search:

```bash
python src/l2_proficiency/main.py --no-plots --skip-tuning
```

Run the full historical workflow, including the 50-candidate search and interactive figures:

```bash
python src/l2_proficiency/main.py
```

Use `--no-plots` for a run without interactive windows, or `--data /path/to/workbook.xlsx` to override the default bundled `data/IA_data.xlsx`. The default full search performs 750 cross-validation fits plus a refit and may take longer. Figures display interactively; the workflow does not automatically overwrite saved conference figures.

Graphviz tree rendering is optional. If the Python `graphviz` package or system `dot` executable is absent, trees use Matplotlib instead. To enable Graphviz, install its Python package and the Graphviz system application.

**Validation:** The supplied dataset, configured-model run, plotting path, and full search were checked with Python 3.12.14 during this repository cleanup. See [validation details](docs/VALIDATION.md) and [tested package versions](requirements-tested.txt). These are current checks, not a reconstruction of the 2025 environment. Historical package versions were not recorded, and unseeded feature selection and validation splits mean rerun scores can vary.

For supplementary classifier comparisons and stability analysis:

```bash
# macOS/Linux, from repository root
PYTHONPATH=src/l2_proficiency python src/l2_proficiency/experiments/compare_models.py
```

This supplementary utility shares the historical full-data preprocessing limitations. It was not included in the execution checks for this cleanup.

## Team contributions

| Contributor | Contributions |
| --- | --- |
| **Selina Hui** | Project initiation, data inspection, feature engineering, model experimentation, visualization, interpretation, and poster preparation |
| **Lily Meihui Wu** | Data analysis, preprocessing, and machine-learning model development across the analysis pipeline |
| **Sienna Jackson** | Literature review and guidance on feature engineering informed by bilingual language-processing and eye-movement research |
| **Asha Dukkipati** | Initial project design, refinement of the prediction objective, poster layout, presentation of model results, and abstract writing |

## Limitations and next steps

- **Small independent sample:** There are 80 participants; repeated reading records do not increase that number.
- **Historical data leakage:** Learned preprocessing, feature selection, and tuning were not fully confined to training data.
- **Restricted task:** The model distinguishes two source-study groups rather than measuring proficiency across populations or languages.
- **Feature interpretation:** Filtering, transformations, and derived-feature formulas affect what model predictors represent.
- **No external validation:** Reported performance does not establish generalization to a separate cohort.

A methodological extension would fit every learned preprocessing and feature-selection step within participant-level training folds, tune within those folds, and evaluate on held-out participants or an independent cohort. Any new evaluation should be reported separately from the historical conference analysis.

## Data source and acknowledgment

Wang, T., Wang, Y., Hu, H., et al. (2025). *An eye-movement database of bilingual language control for Chinese-English bilinguals*. **Scientific Data, 12**, 332. [https://doi.org/10.1038/s41597-025-04628-2](https://doi.org/10.1038/s41597-025-04628-2).

We thank the source-study authors for making the data available. Source data and third-party materials remain subject to their applicable terms.

**Conference poster title:** *Prediction of Bilingual Proficiency through Eye-Tracking Analysis of Mandarin Chinese-English Code-Switching: A Machine Learning Approach.*
