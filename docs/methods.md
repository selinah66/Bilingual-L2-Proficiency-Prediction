# Methods: English L2 Proficiency Prediction from Eye-Tracking Data

## 1. Objective and scope

The project explored whether eye-movement measurements collected during reading could distinguish low- and high-proficiency English L2 readers among Mandarin Chinese–English bilinguals. The prediction target was the source dataset's participant-level proficiency category, not the occurrence of a language switch, reading comprehension, or a continuous proficiency score.

Earlier work investigated code-switching prediction. Those experiments, together with proposed new data collection and alternative feature representations, are historical exploration rather than components of the proficiency pipeline described below.

## 2. Data provenance and observation units

The project reused the dataset described by [Wang et al. (2025)](https://doi.org/10.1038/s41597-025-04628-2). The team did not recruit the source-study participants or collect the original recordings. The study includes Mandarin Chinese–English bilinguals reading monolingual and code-switched sentences.

The supplied `IA_data.xlsx` is the workbook used for the verified sample accounting below. The supplied Python configuration references `Data/RawEyeMovement.xlsx`; the correspondence between these filenames should be preserved explicitly rather than assumed. `Sentences.xlsx` contains stimulus information. `Descriptions.R` and `Technical_validation.R` provide source-study documentation and validation analyses and are separate from the Python classifier.

| Unit | Identification | Interpretation |
| --- | --- | --- |
| Participant | `RECORDING_SESSION_LABEL` | Independent person and final prediction unit |
| Participant–trial | Participant plus `TRIAL_INDEX` | A reading trial completed by a particular person |
| Interest-area record | Participant, trial, and `IA_ID` | Measurements for one word/interest area within that trial |

`TRIAL_LABEL` identifies the sentence/item; `IA_LABEL` contains the interest-area text. `LANGUAGE` and `CONDITION` describe experimental properties. A repeated `IA_ID` is not a globally unique word or observation identifier.

### 2.1 Verified sample accounting

| Quantity | Input workbook | After supplied preprocessing |
| --- | ---: | ---: |
| Participants | 80 | 80 |
| Participant–trial combinations | 12,800 | 10,091 |
| Interest-area records | 25,600 | 11,290 |
| Low / high participants | 40 / 40 | 40 / 40 |
| Low / high interest-area records | 12,800 / 12,800 | 5,729 / 5,561 |

These sample-flow counts have been verified by the project owner for the documented preprocessing. The input workbook contains 160 critical trials per participant and two interest-area records per participant–trial. Filtering can retain one or both records from a trial. The source study also presented 40 filler sentences per participant; these are distinct from the 160 critical trials represented here.

The final modeling table has **one row per participant**, giving 80 samples. The 11,290 retained interest-area records contribute to those features and are not treated as 11,290 independent people.

## 3. Measurements and initial inspection

The input is an interest-area report containing calculated eye-movement measures. It does not contain continuous timestamped gaze coordinates; this project did not perform raw-gaze event detection.

| Measure | Input field | Interpretation |
| --- | --- | --- |
| First fixation duration | `IA_FIRST_FIXATION_DURATION` | Duration of the first fixation on an interest area |
| Second fixation duration | `IA_SECOND_FIXATION_DURATION` | Duration of its second fixation, when available |
| Gaze duration | `IA_FIRST_RUN_DWELL_TIME` | Fixation time during the first run through an interest area |
| Regression-path duration | `IA_REGRESSION_PATH_DURATION` | Go-past time, including regressions before progressing beyond an interest area |
| Total dwell time | `IA_DWELL_TIME` | Sum of fixation durations on an interest area |
| First saccade amplitude | `IA_FIRST_SACCADE_AMPLITUDE` | Amplitude of the first saccade associated with an interest area |
| Fixation count | `IA_FIXATION_COUNT` | Number of fixations, not fixation duration |
| Skipping indicator | `IA_SKIP` | Whether the interest area was skipped during first-pass reading |
| First-run fixation count | `IA_FIRST_RUN_FIXATION_COUNT` | Number of fixations in the first run |
| Regression counts | `IA_REGRESSION_IN_COUNT`, `IA_REGRESSION_OUT_COUNT` | Regressions into or out of an interest area |

Raw duration measurements are recorded in milliseconds; fixation and regression measures are counts, and `IA_SKIP` is a binary indicator. Scaled duration and count inputs no longer retain those original units. The physical unit of `IA_FIRST_SACCADE_AMPLITUDE` is not established in the repository documentation and is therefore left unspecified; captions should not assign degrees, pixels, or another physical unit without verified metadata. Derived features calculated after scaling must be labeled as transformed features rather than raw physical measurements.

Initial inspection checks column names, types, missingness, participant labels, and duplicate participant–trial–interest-area keys. Participant IDs and item IDs are identifiers.

## 4. Preprocessing

### 4.1 Record filtering

`filter_rows()` retains records whose `IA_LABEL` begins with an ASCII alphabetic character, using `^[a-zA-Z]`. This is the implemented operational rule for retaining English records; it is not a language-classification model.

The configured missingness-filter groups are processed after excluding protected columns. With the supplied configuration, this removes records missing `IA_FIRST_RUN_DWELL_TIME`, followed by those missing `IA_FIRST_SACCADE_AMPLITUDE`. Records missing any configured critical field are then excluded:

- `IA_REGRESSION_PATH_DURATION`
- `IA_FIRST_FIXATION_DURATION`
- `IA_FIRST_RUN_DWELL_TIME`
- `IA_FIRST_SACCADE_AMPLITUDE`

These exclusions can remove skipped or otherwise unmeasured interest areas.

### 4.2 Encoding and imputation

The supplied `fit_transformations()` encodes proficiency and condition with `LabelEncoder`. The confirmed proficiency encoding for the reported analysis is:

| Encoded value | Source label | Proficiency group |
| --- | --- | --- |
| 0 | `H` | High |
| 1 | `L` | Low |

Class-specific metrics and confusion-matrix captions should use this mapping. A numeric class order of `[0, 1]` therefore means `[High, Low]`. Alternate configuration files must not be used to reinterpret this confirmed encoding.

The numeric preprocessing list contains first-fixation duration, gaze duration, regression-path duration, dwell time, first-saccade amplitude, fixation count, first-run fixation count, and regression-in count. Remaining missing values are imputed as follows:

| Column group | Strategy |
| --- | --- |
| Protected: first-fixation duration, regression-path duration, dwell time | Median |
| Other configured numeric columns | Mean |

This is a configured column-group rule, not an automatic decision based on measured skewness. Critical-field exclusions occur before imputation; protecting a column does not rescue a record removed for missing critical measurements.

### 4.3 Clipping and scaling

For each configured numeric column, clipping bounds are calculated after imputation as:

\[
[Q_1 - 3\,IQR,\; Q_3 + 3\,IQR], \qquad IQR=Q_3-Q_1.
\]

Values outside the bounds are clipped to the nearest bound. This retains the record and differs from deleting every outlier row. `RobustScaler` then centers and scales the numeric columns using their median and interquartile range.

The earlier working documentation also mentions percentile clipping, `StandardScaler`, skew-triggered log transformations, and low-variance filtering. Their inclusion and order in the conference run have not been established. They are not asserted here as additional steps of the confirmed preprocessing sequence. The explicit `log_first_fixation` calculation described in Section 5.1 is a separate derived feature.

### 4.4 Returned stage labels

In the documented stage dictionary, `imputed` and `processed` refer to the **same fully processed DataFrame**. They are not separate snapshots immediately after imputation and after scaling. 

## 5. Participant features

`aggregate_features()` computes participant means of five transformed columns: first-fixation duration, regression-path duration, dwell time, fixation count, and first-saccade amplitude. Although the function receives a longer feature list, its implemented aggregation dictionary contains these five measures.

It also attaches each participant's first retained `CONDITION` value. This does not represent their complete experimental-condition history or a participant-by-condition aggregation.

### 5.1 Derived predictors

Let \(\bar{x}\) denote a participant mean of the already transformed input column and let \(\epsilon=10^{-6}\). The historical feature-construction step uses:

| Code name | Implemented calculation |
| --- | --- |
| `regression_dwell_ratio` | Mean regression-path duration / (mean dwell time + epsilon) |
| `fixation_density` | Mean fixation count / (mean dwell time + epsilon) |
| `log_first_fixation` | `log1p(mean first-fixation duration)` |
| `saccade_speed` | Mean first-saccade amplitude / (mean first-fixation duration + epsilon) |

Because scaling precedes these calculations, the ratios are mathematical transformations of scaled participant features, not literal ratios of raw times. `saccade_speed` is the column name: dividing amplitude by fixation duration does not measure actual saccade velocity. 

Together with the five means and retained condition value, this provides ten candidate predictors before selecting five. `prepare_features_and_target()` joins one proficiency label per participant and removes the participant identifier from the predictor matrix.

## 6. Feature selection and model fitting

Random-forest impurity-based importance is used to rank candidates and retain five predictors. The poster identifies the following five predictors; this list identifies the selected set and does not claim an independently verified importance ranking:

| Poster name | Model field | Representation |
| --- | --- | --- |
| Dwell Time | `IA_DWELL_TIME` | Participant mean of transformed dwell time |
| Regression Path Duration | `IA_REGRESSION_PATH_DURATION` | Participant mean of transformed go-past time |
| Fixation Count | `IA_FIXATION_COUNT` | Participant mean of transformed fixation count |
| First Fixation Duration | `IA_FIRST_FIXATION_DURATION` | Participant mean of transformed first-fixation duration |
| Fixation Density | `fixation_density` | Ratio defined in Section 5.1, calculated from transformed participant means |

Correlation plots describe associations and redundancy; they are not causal analyses.

### 6.1 Initial configured forest and optimized selection

The primary reported metrics in the current script and the metrics in **poster Table 3** correspond to the **initial configured random forest**. Hyperparameter optimization is a separate analysis branch.

| Parameter | Initial configured forest: Table 3 | Poster-reported optimized selection |
| --- | ---: | ---: |
| Trees (`n_estimators`) | 50 | 50 |
| Maximum depth (`max_depth`) | 10 | 5 |
| Minimum samples to split (`min_samples_split`) | 10 | Not confirmed for the saved optimized run |
| Minimum samples per leaf (`min_samples_leaf`) | 4 | 8 |
| Features per split (`max_features`) | 0.3 | 0.5 |
| Class weights (`class_weight`) | `balanced` | `balanced` in the supplied tuning code |

The poster's optimized selection does not identify every estimator setting. In particular, the selected `max_leaf_nodes` value has not been recovered. The supplied tuning code fixes `bootstrap=True`; no missing saved-run parameters are inferred from defaults.

### 6.2 Hyperparameter-search configuration

The supplied tuning code uses `RandomizedSearchCV` to evaluate **50 candidate configurations**, scored by **weighted F1** (`f1_weighted`), with **5-fold cross-validation repeated 3 times**. The estimator and randomized search use `random_state=42`. The supplied `RepeatedKFold` construction does not set its own random state; do not assume all fold assignments were fixed by the search seed.

| Search parameter | Candidate values in the supplied code |
| --- | --- |
| `n_estimators` | 50, 100, 200 |
| `max_depth` | `None`, 5, 10 |
| `min_samples_split` | 10, 20 |
| `min_samples_leaf` | 4, 8 |
| `max_features` | `sqrt`, 0.3, 0.5 |
| `bootstrap` | `True` |
| `max_leaf_nodes` | 50, 100 |

Weighted F1 is the search objective; the macro F1 reported in Table 3 is a different metric. Neither the best cross-validation score nor the separate optimized-model score should be presented as the Table 3 result.

The participant groups are already balanced. The feature-assembly function's optional weighted sampling branch, if triggered, reorders all rows without replacement rather than changing class counts. Class weighting in the forest is a separate mechanism.

## 7. Evaluation and interpretation

The model used a stratified **60/40 participant-level split** with `random_state=42`: **48 training participants (24 per group)** and **32 evaluation participants (16 per group)**. There is one modeling row per participant, and no participant is present on both sides of this split.

`RepeatedKFold` was used for the repeated validation procedure, with 5 folds repeated 3 times. This cross-validation splitter is not stratified, unlike the 60/40 training/evaluation split. Randomized search involves 15 fold evaluations per candidate. The separate classifier-comparison utility uses 5-fold cross-validation.

**Historical evaluation limitation:** preprocessing and supervised feature selection are fitted on the full dataset before the participant split. The initial configured forest is then fitted on the training partition and evaluated on the nominal evaluation partition. This earlier exposure to evaluation information means the Table 3 result is not independent of all model development. In a separate optimization branch, a forest is also tuned on the full dataset before scoring the nominal evaluation subset; that score must not be substituted for Table 3. The historical workflow is not leakage-free, and its scores are exploratory rather than an unbiased estimate of unseen-participant performance.

**Poster Table 3 reports the initial configured, non-optimized random forest**, with accuracy **0.69** and macro F1 **0.69** for **32 evaluation participants**. These metrics must be associated with the configured settings in Section 6.1, not the poster-reported optimized selection. The majority-class baseline is **50%**, because the evaluation supports are 16 per group.

Feature distributions use repeated interest-area measurements, while prediction uses participant samples. These observation levels must be distinguished in captions and sample-size statements. Train/validation gaps describe apparent fitting differences; they do not demonstrate external validity.

## 8. Execution and reproducibility status

### 8.1 Entry point

From the repository root, run the proficiency pipeline with the project owner's specified command:

```bash
python3 src/l2_proficiency/main.py
```

The supplied configuration references `Data/RawEyeMovement.xlsx`. The verified sample table was supplied under the filename `IA_data.xlsx`; the relationship between those filenames remains an explicit documentation uncertainty. The repository command is recorded here as provided; an end-to-end execution of that repository layout was not performed during this documentation revision.

### 8.2 Environment record

| Item | Available record |
| --- | --- |
| Python version | **3.14.8**, reported by the project owner |
| Historical package versions | Not recorded in the supplied materials |
| Historical environment lockfile | Not available in the supplied materials |
| Exact conference commit | Not established |
| Model associated with poster Table 3 | Initial configured random forest; see Sections 6.1 and 7 |

The reported Python version has not been independently tied to a saved conference-run environment. It should not be interpreted as proof of the original interpreter or package versions. A current environment record, if added later, should be identified separately from the historical environment.

### 8.3 Remaining documentation limitations

The following details remain unresolved and are retained explicitly:

- The exact conference script revision or commit, and its correspondence to the reorganized repository entry point.
- The correspondence between `IA_data.xlsx` and the configured `Data/RawEyeMovement.xlsx` input.
- Historical dependency versions and a complete saved execution environment.
- The complete saved optimized estimator, including its selected `min_samples_split` and `max_leaf_nodes` values.
- Whether the additional transformations mentioned in the earlier working documentation were used in the conference run, and in what order.
- The physical unit of `IA_FIRST_SACCADE_AMPLITUDE` in the repository's input documentation.

Archived and reorganized code versions differ in some seed settings, label mappings, and preprocessing details. The label mapping and result attribution documented here follow the project owner's confirmations; alternate code versions are not evidence that the reported run used their settings. In particular, one supplied configuration declares the reverse target mapping (`L=0`, `H=1`), while the confirmed reported analysis uses **`H=0`, `L=1`**. This discrepancy is documented rather than silently reconciled.

The preprocessing-stage labels described in Section 4.4 are an established implementation detail, not an unresolved question. This revision clarifies documentation and does not rerun models or alter the conference results.

## 9. Related documentation and source

See [literatureReview.md](literatureReview.md) for the research background. The underlying source-study reference is:

Wang, T., Wang, Y., Hu, H., Wang, X., Chen, S., & Yang, Y. (2025). An eye-movement database of bilingual language control for Chinese-English bilinguals. *Scientific Data, 12*, 332. https://doi.org/10.1038/s41597-025-04628-2
