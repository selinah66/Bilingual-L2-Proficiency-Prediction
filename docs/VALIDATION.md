# Execution checks

These checks describe the cleaned repository in the current runtime. They do not reconstruct the 2025 conference environment or provide a corrected generalization estimate.

## Environment

Python 3.12.14 with the direct dependency versions in [requirements-tested.txt](../requirements-tested.txt). The runtime already had these packages installed, there was no fresh installation from the requirements files performed. The optional Graphviz Python package was absent, and tree plotting used the Matplotlib fallback.

## Checks completed

| Check | Outcome |
| --- | --- |
| Workbook schema and duplicate participant/trial/interest-area keys | Passed |
| One nonmissing proficiency label per participant | Passed |
| Historical preprocessing and feature engineering | Passed |
| One modeling row per participant and finite numeric predictors | Passed |
| Stratified 60/40 participant split | 48 training, 32 evaluation; no participant overlap |
| `python src/l2_proficiency/main.py --no-plots --skip-tuning` | Completed |
| `MPLBACKEND=Agg python src/l2_proficiency/main.py --skip-tuning` | Completed; exercised plot generation with a noninteractive backend, including tree fallback |
| `python src/l2_proficiency/main.py --no-plots` | Completed, including the 50-candidate × 15-fold search and subsequent configured-model evaluation |
| Main README and documentation local Markdown links | Checked against included files |
| Python syntax | Compiled without executing archived experiments |

The model runs used `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` to limit numeric-library threads. These do not change the configured statistical workflow.

## Verified sample accounting

| Quantity | Input | After historical preprocessing |
| --- | ---: | ---: |
| Interest-area records | 25,600 | 11,290 |
| Participant–trial combinations | 12,800 | 10,091 |
| Participants | 80 | 80 |
| Low-proficiency records | 12,800 | 5,729 |
| High-proficiency records | 12,800 | 5,561 |

Participant aggregation produced 80 rows and 10 candidate predictors, including the historical first-retained `CONDITION` field. Feature selection then retains five predictors for the main model. The participant groups each contain 40 people.

## Rerun scores are separate from conference results

The first configured-model check returned 0.69 evaluation accuracy, while the later full-workflow check returned 0.72. The full-data tuned estimator returned a nominal evaluation accuracy of 0.81. These are execution observations, not new validated performance claims. In particular, the tuned estimator is refitted on all participants before scoring the nominal evaluation subset, so that subset is not held out from its training.

The original workflow leaves selector forests and repeated cross-validation splits unseeded. Feature selection also fits a separate forest from the forest used to display feature importances. Rerun variation and possible discrepancies between selected features and displayed rankings are therefore expected. No seeds, transformations, selected features, hyperparameters, or scientific evaluation procedures were changed during cleanup.

## Limits of these checks

- Full-dataset preprocessing and supervised feature selection still influence evaluation.
- Plot generation was checked with Agg.
- Optional Graphviz rendering itself was not checked, while its Matplotlib fallback was.
- Archived scripts, R scripts, and the supplementary classifier-comparison utility were not executed.
- Historical conference commit, fitted estimator, and environment remain unverified.
- A successful run does not establish source-data redistribution permissions, external validity, or publication status.
