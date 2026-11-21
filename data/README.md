# Data guide

This project analyzes published data rather than original team recordings.

**Source:** Wang, T., Wang, Y., Hu, H., et al. (2025). *An eye-movement database of bilingual language control for Chinese-English bilinguals*. Scientific Data, 12, 332. https://doi.org/10.1038/s41597-025-04628-2

| File | Role |
| --- | --- |
| `IA_data.xlsx` | Bundled input used by the Python workflow; first sheet contains reading records and the second describes variables |
| `Sentences.xlsx` | Source-study sentence materials |
| `Descriptions.R` | Source-study descriptive analysis; separate from the Python model |
| `Technical_validation.R` | Source-study validation analysis; separate from the Python model |

The first worksheet of `IA_data.xlsx` contains 25,600 interest-area records for 80 participants. Records are identified by participant (`RECORDING_SESSION_LABEL`), trial (`TRIAL_INDEX`), and interest area (`IA_ID`). The prediction target is `L2 PROFICIENCY`: `H` or `L`. The historical LabelEncoder maps H to 0 and L to 1.

The default path is resolved relative to the repository rather than the shell's working directory. Override it with:

```bash
python src/l2_proficiency/main.py --data /path/to/IA_data.xlsx
```

Use the dataset links provided in the source publication if you need to obtain the workbooks again. The repository does not establish that its workbook is byte-identical to the original conference-run input. Preserve the source citation and follow the source materials' reuse terms; no new license is assigned to them by this cleanup.

Run `python scripts/check_data.py` from the repository root to inspect the bundled workbook's current counts and participant-level features.
