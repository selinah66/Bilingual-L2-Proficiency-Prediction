# Predicting English L2 Proficiency Using Eye-Tracking in Mandarin Chinese–English Bilinguals

An exploratory machine-learning study connecting bilingual reading behavior with English second-language proficiency.

**Reported conference result:** 69% accuracy and macro F1 of 0.69 on 32 participant samples, compared with a 50% majority-class baseline.

**Authors:** Selina Hui (Project Lead), Lily Meihui Wu, Asha Dukkipati, and Sienna Jackson  
**Repository:** [NeurotechUSC Bilingual Code-Switching](https://github.com/selinah66/NeurotechUSC-Bilingual-Code-Switching)  
**Conference Date:** California NeuroTechnology Conference 2025 (https://caneurotech.vercel.app/), Saturday, April 26th, 2025
**Conference materials:** [Poster](docs/BPET_Poster.pdf) · [Supplemental information and figures](docs/supplementalFig.pdf)

*Poster title: Prediction of Bilingual Proficiency through Eye-Tracking Analysis of Mandarin Chinese-English Code-Switching: A Machine Learning Approach.*

## Research question

How do eye-movement patterns differ between English proficiency groups, and how well can these features classify proficiency in this dataset?

The analysis examines measures such as fixation counts, dwell time, and regression-path duration. These describe aspects of reading behavior and provide indirect indicators of processing demands; they do not directly measure comprehension or establish a cognitive mechanism.

## Project overview

This project combines data inspection, preprocessing, participant-level
feature engineering, random-forest classification, and exploratory
visualization using published eye-tracking data. The repository documents
the completed conference analysis and preserves its reported results.

## Data source

The data come from [Wang et al. (2025), *An eye-movement database of bilingual language control for Chinese-English bilinguals*](https://doi.org/10.1038/s41597-025-04628-2), published in *Scientific Data*.

The source study includes 80 participants, with 40 in each proficiency group, reading monolingual and code-switched sentences. This project reuses the published data; our team did not collect the original recordings.

| Item | Description |
| --- | --- |
| `IA_data.xlsx` | Participant, trial, interest-area, and proficiency identifiers, together with calculated eye-movement measurements; its second sheet describes the variables |
| `Sentences.xlsx` | Experimental sentence materials, trial labels, conditions, and experimental lists |
| `Descriptions.R` | Source-study descriptive-statistics and t-test code |
| `Technical_validation.R` | Source-study paired t-test and effect-size code |

### Source and analyzed samples

| Quantity | Source workbook | After supplied preprocessing |
| --- | ---: | ---: |
| Unique participants | 80 | 80 |
| Distinct participant–trial combinations | 12,800 | 10,091 |
| Interest-area records | 25,600 | 11,290 |
| Low / high participants | 40 / 40 | 40 / 40 |
| Low / high interest-area records | 12,800 / 12,800 | 5,729 / 5,561 |

The source workbook contains 160 trials per participant and two interest-area records per participant–trial. The analyzed record counts above were obtained by running the preprocessing implementation, which retains English interest-area records and applies its missing-value rules. A retained trial may contribute one or two records.

**Final modeling unit: participant.** Measurements are aggregated into one feature vector per participant, producing 80 modeling samples: 40 low-proficiency and 40 high-proficiency readers. Multiple eye-movement records from a person contribute features to that person's sample; they do not increase the number of independent participants.

## Analysis workflow

1. Load the source data and inspect identifiers, proficiency labels, data types, duplicate keys, and missing values.
2. Apply the documented English-record and missing-value inclusion rules.
3. Impute remaining selected numeric measurements, clip outliers, and apply robust scaling using the historical preprocessing sequence.
4. Aggregate selected eye-movement measurements by participant and construct the derived features used by the analysis.
5. Describe feature distributions and proficiency-group counts, then select predictors and fit the configured or tuned random-forest model.
6. Summarize evaluation metrics and visualize exploratory reading patterns and model behavior.

### Features

The implemented participant aggregation computes means of first-fixation duration, regression-path duration, total dwell time, fixation count, and first-saccade amplitude. Derived predictors include ratios involving regression duration, fixation count, dwell time, and first-fixation duration, along with a logarithmic first-fixation feature.

The historical aggregation also attaches the first retained `CONDITION` value for each participant; this is not a summary of every condition that participant read. Derived-feature formulas and their position relative to scaling are preserved in the source code. Interpret these predictors according to their implemented formulas rather than treating their names as direct physiological measurements.

### Evaluation design

We used a **stratified participant-level split**, allocating 60% of participants to training and 40% to evaluation, with `random_state=42`. The 80 participants were evenly divided between proficiency groups, resulting in 48 training participants (24 per group) and 32 evaluation participants (16 per group). Each modeling row represented one participant, so no participant appeared in both partitions.

Random-forest tuning and cross-validation summaries used **5-fold cross-validation repeated 3 times**. Randomized hyperparameter search evaluated **50 candidate configurations**, scored by weighted F1, with 15 fold evaluations per candidate, as documented in supplemental Figure S9. The separate classifier-comparison utility used 5-fold cross-validation. Participants could appear in different roles across folds or repeats, but never on both sides of an individual fold split.

**Historical evaluation limitation**: Preprocessing and supervised feature selection were fitted using the full dataset, and one model was tuned on the full dataset before scoring the nominal evaluation subset. Although the participant partitions were disjoint, evaluation information therefore influenced model development. The reported scores should be interpreted as **exploratory results** rather than an unbiased estimate of performance on unseen participants. This repository preserves the historical workflow for transparency.

## Findings and evaluation

The project initially explored **code-switching prediction**. Following exploratory analysis, we shifted to predicting English L2 proficiency because the available data were better suited to this objective. Earlier code is retained in `archive/codeSwitching/` for reference.

Exploratory comparisons associated lower proficiency with longer dwell times, more fixations, and greater regression-related activity, consistent with possible differences in reading effort. These observations are limited to the analyzed sample and measurements; they do not establish a causal relationship or independently validate proficiency prediction.

### Primary reported result

These results are transcribed from **poster Table 3** similar to the classification-report portion of supplemental Figure S9.

| Metric | Reported result |
| --- | --- |
| Task | Binary English L2 proficiency classification |
| Evaluation sample | 32 participants: 16 high, 16 low |
| Majority-class baseline | 50% accuracy, calculated from equal class supports |
| Random forest accuracy | 0.69 |
| Balanced accuracy | Approximately 0.69, from Table 3's reported macro recall |
| Macro F1 | 0.69 |
| Weighted F1 | 0.69 |

| Proficiency group | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| High | 0.67 | 0.75 | 0.71 | 16 |
| Low | 0.71 | 0.62 | 0.67 | 16 |

Reported accuracy is approximately **19 percentage points above the majority-class baseline**. Recall is higher for high-proficiency readers (0.75 versus 0.62), while precision is higher for low-proficiency predictions (0.71 versus 0.67). These are descriptive comparisons; the small sample and historical evaluation limitations constrain generalization claims.


<details>
<summary>Original conference classification table</summary>

![Original classification table from the conference poster](figures/resultTable.png)

Poster Table 3, extracted without recomputing results. Support values count participants; metrics are rounded to two decimal places. Balanced accuracy above summarizes the reported macro recall and is not a separately retained model output.

</details>

### Selected figures

![Dwell-time and fixation-count distributions by proficiency group](figures/readingMeasuresbyProf.png)

**Figure 1. Reading behavior by proficiency group.** Distributions of dwell
time (milliseconds) and fixation count for low- and high-proficiency readers.
The observations are interest-area records, with repeated measurements from
participants; they are not independent participant-level samples.

![Exploratory feature importance and target correlations](figures/featureImportancelogFixVariant.png)

**Figure 2. Exploratory feature-importance analysis.** Random-forest
importance scores and correlations with the encoded proficiency label for
one historical feature-selection variant. This export includes
log-transformed first-fixation duration among its selected predictors.
Importance scores describe model behavior and do not establish causal effects.


### Poster-reported optimized configuration

| Hyperparameter | Reported value |
| --- | ---: |
| Number of trees | 50 |
| Maximum depth | 5 |
| Minimum samples to split | 10 |
| Minimum samples per leaf | 8 |
| Maximum features per split | 0.5 |
| Class weighting | Balanced |

These optimized settings differ from the defaults in the supplied `config.py`, such as depth 10 and minimum leaf size 4. Figure S9 displays outputs from more than one model stage; the materials do not establish that every score belongs to this same fitted configuration.

## Code organization

| Module | Responsibility |
| --- | --- |
| `config.py` | Input paths, feature lists, filtering settings, and model parameters |
| `cross_validate_model.py` | Cross-validation evaluation, classifier comparisons, and model-stability analysis |
| `data.py` | Workbook loading, identifier checks, and sample summaries |
| `preprocessing.py` | Record filtering, encoding, imputation, clipping, and scaling |
| `feature.py` | Implemented participant aggregation, derived predictors, and feature/target assembly |
| `models.py` | Feature selection, random-forest training and tuning, and comparison-model constructors |
| `main.py` | Orchestrates the end-to-end workflow from data loading, preprocessing through feature engineering, model training, evaluation, reporting, and visualization. |
| `evaluation.py` | Cross-validation, classifier comparison, and stability summaries |
| `reporting.py` | Printed preprocessing diagnostics |
| `plots.py` | Data distributions, preprocessing diagnostics, model comparisons, and interpretation figures |
| `archive/codeSwitching/` | Earlier exploration of code-switching prediction |

### Supplementary functions

`compare_classifiers()` and `analyze_model_stability()` support supplementary model comparisons and variability analyses. `apply_transformations()` applies previously fitted preprocessing state. Plotting and reporting functions help inspect the analysis and are not separate prediction models.

### Execution

Place the input workbook at the location configured in `config.py`, then run the confirmed entry-point script from the repository root:

```bash
python3 src/l2_proficiency/main.py
```

### Configured settings

| Setting | Value in supplied configuration |
| --- | --- |
| Test fraction | 0.40 |
| Shared random-state setting | 42 |
| Configured random forest | 50 trees; maximum depth 10 |
| Minimum samples to split / per leaf | 10 / 4 |
| Maximum features per split | 0.30 |
| Primary random-forest class weights | `balanced` |
| Feature selection | Top 5 features |
| Repeated cross-validation | 5 folds × 3 repeats |
| Hyperparameter search | 50 candidates; weighted F1 |

These describe the supplied configuration, not the fitted parameters of every experimental model. Use the reference version to confirm the settings associated with the reported conference run.

### Dependencies

The Python modules use `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`, `Pillow`, and the Python `graphviz` package. Tree rendering also requires the Graphviz system application. The R files are source-study supporting scripts and are separate from the Python classification workflow.

**Python version:** Python 3.14.8
Exact package versions used for the conference analysis were not recorded.

Some historical operations did not specify random seeds. Reorganization alone does not guarantee identical rerun scores; original saved metrics, figures, and fitted models, where available, are the record of the conference results.

## Team and contributions

This project was completed collaboratively by Selina Hui, Lily Meihui Wu, Sienna Jackson, and Asha Dukkipati. Our team reused published eye-tracking data; we did not collect the original recordings. Contributions to the secondary analysis and conference materials are summarized below.

| Contributor | Contributions |
| --- | --- |
| **Selina Hui** | Project initiation, data inspection, feature engineering, model experimentation, visualization, interpretation of findings, and poster preparation. |
| **Lily Meihui Wu** | Data analysis, preprocessing, and machine-learning model development, contributing across the analysis pipeline. |
| **Sienna Jackson** | Literature review and guidance on feature engineering based on research into bilingual language processing and eye-movement measures. |
| **Asha Dukkipati** | Initial project design, refinement of the prediction objective, poster layout, presentation of model results, and abstract writing. |

## Limitations and future directions

- The independent sample comprises 80 participants; repeated interest-area records do not increase that number.
- The task classifies the source study's low/high groups rather than estimating a continuous proficiency score.
- English-record filtering and missing-value exclusions limit which reading behavior contributes to the analysis.
- The historical preprocessing and model-development sequence limits conclusions about generalization to new participants.
- Exploratory group differences do not establish causation or isolate every possible explanation for reading behavior.

A future study could fit all learned preprocessing and feature selection within participant-level training folds and evaluate the resulting pipeline on an independent cohort. Such an analysis would be a separate methodological extension, not a replacement for the historical conference results.

## Citation and acknowledgments

Wang, T., Wang, Y., Hu, H., et al. (2025). *An eye-movement database of bilingual language control for Chinese-English bilinguals*. Scientific Data, 12, 332. [https://doi.org/10.1038/s41597-025-04628-2](https://doi.org/10.1038/s41597-025-04628-2).

We thank the source-study authors for making the data available. Reuse of source data and third-party materials remains subject to their applicable terms.
