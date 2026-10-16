# Literature Review: Research Background & Selected References 

This selective review explains the research background for our conference project on predicting English L2 proficiency from published eye-movement measurements. It draws on collaboratively prepared literature notes and is not a systematic review. 

The original literature notes were prepared collaboratively by the project team, with Sienna Jackson contributing primarily to the literature review and its connection to feature engineering.

## Research question

Our project asks whether participant-level summaries of reading behavior distinguish the source dataset's lower- and higher-proficiency groups. Three comparisons must remain separate:

- **L1 versus L2 reading:** how bilingual readers read in their first and second languages.
- **Bilingual versus monolingual reading:** how different reader groups perform under specified language and task conditions.
- **Proficiency-group prediction:** whether measurements distinguish proficiency groups within a bilingual sample.

Evidence for one comparison provides context for the others; it does not establish that they have the same explanation or predictive performance.

## Evidence from bilingual reading

[Cop, Drieghe, and Duyck (2015)](https://doi.org/10.1371/journal.pone.0134008) compared eye movements during natural novel reading. Within their bilingual sample, L2 reading involved longer sentence reading times, more fixations, shorter rightward saccades, and less word skipping than L1 reading. There was no overall main effect of reading language on regression rate, although regression patterns interacted with word length. Bilingual L1 reading did not show major overall differences from monolingual reading in the comparison studied.

These findings support considering multiple eye-movement measures. They do not establish a universal bilingual disadvantage or imply that every regression measure increases as proficiency decreases.

[Quiñonez-Beltran et al. (2024)](https://doi.org/10.3390/educsci14040375) synthesized L2 eye-tracking research in a scoping review. Their discussion emphasizes proficiency and the characteristics of both languages, including orthographic differences. Their limitations identify inconsistent participant descriptions and variation in tasks and measures across studies.

**Relevance to this project:** interpret patterns within the Mandarin Chinese–English language pair and the source reading task. Findings from other settings motivate feature exploration but do not validate this project's classifier.

## Task demands change the interpretation of gaze

[Tywoniw (2023)](https://doi.org/10.3389/fcomm.2023.1176986) examined L2 reading assessment using cloze, multiple-choice, and summary tasks. Relationships among comprehension, proficiency, individual differences, and eye movements varied across tasks. For example, more fixations per word predicted better summary scores in the fitted model, while shorter average fixation durations were associated with better performance. Proficiency and motivation also interacted with fixation duration in the summary model.

A simple interpretation such as “more fixations always mean poorer reading” is therefore unsuitable. The author also cautions that broad eye-tracking measures do not directly identify particular cognitive processes or prove conscious use of a reading strategy.

A related but different task appears in [Wang and Pellicer-Sánchez (2022)](https://doi.org/10.1111/lang.12495), which studied incidental vocabulary learning during subtitled documentary viewing. Outcomes depended on what was tested: bilingual subtitles outperformed L2 captions for meaning recognition, whereas L2 captions outperformed bilingual subtitles for form recognition. In the bilingual-subtitle condition, total and first-pass reading times on unfamiliar L2 words predicted form-recognition and meaning-recall gains; this did not extend to every reading measure or learning outcome.

This study provides context for interpreting attention and learning, rather than directly testing sentence-reading proficiency classification. Its immediate posttests also do not establish long-term vocabulary retention.

## The source dataset and proficiency labels

[Wang et al. (2025)](https://doi.org/10.1038/s41597-025-04628-2) published the Chinese–English bilingual eye-movement database reused in our project. The source study included **80 participants, with 40 in each proficiency group**, reading monolingual and code-switched sentences. Our team did not collect the original recordings.

The full stimulus pool contained **1,280 experimental sentences across eight conditions**. Each participant received **160 critical sentences plus 40 fillers**, or 200 sentences in total. The pool size is therefore different from the number of trials presented to each participant.

The higher-proficiency group comprised English majors scoring at least 50 on the Oxford Placement Test; the lower-proficiency group comprised non-English majors scoring 20–35. Self-rated language skills were also considered. The Bilingual Switching Questionnaire measured switching frequency; it was not the objective proficiency test.

Because group membership also differed by academic major, interpretation should remain tied to these source-defined groups. The classifier does not isolate proficiency from all related educational or language-experience differences.

## Implications for feature interpretation

The source study provides measurements for two target interest areas within reading trials. These are repeated observations, not independent participants. Our final modeling unit is the **participant**; preprocessing and aggregation are described in [methods.md](methods.md).

The following definitions summarize the source study's measures. Interpretation depends on the text, task, and analysis.

| Measure | What it records | Interpretation boundary |
| --- | --- | --- |
| First fixation duration | Duration of the first fixation in an interest area | Does not measure all processing of that region |
| Gaze duration | Fixation time before first leaving the region | Describes the first visit rather than all later rereading |
| Regression-path duration | Time from first entry until moving beyond the region's right boundary, including intervening regressions | Is not simply time spent looking backward |
| Total fixation duration / dwell time | Sum of fixation durations in the region | Can include both initial reading and later visits |
| Fixation count | Number of fixations in the region | Its relationship with comprehension or proficiency is task dependent |

The literature supports exploring several measures and interpreting them cautiously. It does not establish a causal effect of proficiency on gaze, a one-to-one mapping between features and cognitive processes, or the validity of an assessment tool. Published associations also do not establish the predictive accuracy of our historical model.

## References

- Cop, U., Drieghe, D., & Duyck, W. (2015). Eye Movement Patterns in Natural Reading: A Comparison of Monolingual and Bilingual Reading of a Novel. *PLOS ONE, 10*(8), e0134008. https://doi.org/10.1371/journal.pone.0134008
- Quiñonez-Beltran, J. F., Seymour, T. M., Robbins, R. A. J., Xu, Y., & Joshi, R. M. (2024). What Can Eye Movements Tell Us about Reading in a Second Language: A Scoping Review of the Literature. *Education Sciences, 14*(4), 375. https://doi.org/10.3390/educsci14040375
- Tywoniw, R. (2023). Compensatory effects of individual differences, language proficiency, and reading behavior: an eye-tracking study of second language reading assessment. *Frontiers in Communication, 8*, 1176986. https://doi.org/10.3389/fcomm.2023.1176986
- Wang, A., & Pellicer-Sánchez, A. (2022). Incidental Vocabulary Learning From Bilingual Subtitled Viewing: An Eye-Tracking Study. *Language Learning, 72*(3), 765–805. https://doi.org/10.1111/lang.12495
- Wang, T., Wang, Y., Hu, H., Wang, X., Chen, S., & Yang, Y. (2025). An eye-movement database of bilingual language control for Chinese-English bilinguals. *Scientific Data, 12*, 332. https://doi.org/10.1038/s41597-025-04628-2

## Contributions and documentation

This synthesis builds on the team's original literature notes. Sienna Jackson contributed primarily to the literature review and feature-engineering rationale. See the repository README for the full team contributions.
