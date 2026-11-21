# Cleanup notes

## Included changes

- Rewrote the README around the research question, sample, project leadership, findings, and reproducible entry points. Retained team attribution and data provenance.
- Linked the existing detailed methods and literature review rather than creating a duplicate methods document.
- Corrected the default dataset path from absent `Data/RawEyeMovement.xlsx` to bundled `data/IA_data.xlsx`.
- Added `--data`, `--no-plots`, and `--skip-tuning` to the historical main entry point. Default model logic remains intact.
- Made the Graphviz Python import optional, enabling the existing Matplotlib fallback when the Python package is missing.
- Corrected printed units for the historical `imputed` stage, which already contains robust-scaled values.
- Added the workbook and participant-feature audit in `scripts/check_data.py`.
- Added root dependency files and retained `src/requirements.txt` as a forwarding file for compatibility.
- Moved workshop notes from `src/learn_notebooks/` into `archive/learning/workshop_notes/`.
- Added data, archive, and documentation guides and a current validation record.
- Removed the unverified Python 3.14.8 environment claim from active documentation.
- Preserved the actual lowercase `figures/` folder and verified its main README image path. The uploaded folder, rather than the earlier screenshot, determines the spelling.
- Omitted local environments, Git internals, IDE settings, operating-system files, caches, duplicate nested project content, and the third-party reading PDF collection from this streamlined copy. The original ZIP is unchanged.

The conference poster, supplemental PDF, source workbooks, existing figures, historical archives, and statistical preprocessing/modeling functions are preserved. No code license was added and no GitHub setting was changed.

## How to apply

Use the cleaned files to update your existing checkout so its Git history is retained. The ZIP does not include `.git/`. Review the diff before committing. Copying a folder over an existing checkout does not remove old tracked clutter: separately review tracked IDE files, environments, obsolete source-note paths, and nested duplicate content before untracking them.

Do not use a broad recursive deletion command. Keep any personal environments and reading PDFs outside the public project folder if you still use them locally. Add or merge the included `.gitignore`; ignored files that are already tracked still need to be untracked separately.

## Suggested GitHub presentation

**About description:**

> Exploratory machine learning on published eye-tracking data from 80 Mandarin–English bilinguals to classify English L2 proficiency. Presented at California NeuroTechnology Conference 2025.

**Topics:** `eye-tracking`, `bilingualism`, `psycholinguistics`, `machine-learning`, `random-forest`, `python`, `scikit-learn`.

If you intend this as a completed research portfolio project, disable Template repository in GitHub settings unless template reuse is intentional. Keep the repository pinned if it is among your strongest projects. Renaming the repository is optional; preserving a stable application link is more useful than renaming it solely for presentation.

## Remaining substantive work

A separate methodological extension should confine learned preprocessing, feature selection, and tuning to participant-level training folds and evaluate on held-out participants. Keep its code, metrics, and interpretation separate from the historical conference analysis. An independent cohort would provide stronger evidence of generalization.

The poster labels Table 3 as post-optimization results, while the existing methods document associates it with the initially configured forest. The cleanup retains the reported values and documents the conflicting attribution rather than selecting an unsupported interpretation.
