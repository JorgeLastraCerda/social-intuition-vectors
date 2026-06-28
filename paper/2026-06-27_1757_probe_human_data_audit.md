# Probe-vs-Human Alignment Data Audit

**Produced:** 2026-06-27 17:57 (Europe/Berlin)
**Model(s):** Gemma-3-12B-it · Gemma-3-27B-it · Llama-3.1-8B-Instruct · Qwen3-14B
**Scope:** Data-quality audit for Phase-7 Test 2: model warmth/competence probe scores vs. human name-level warmth/competence ratings
**Status:** Complete; data accepted for the current paper analyses

## Artifacts

- **Scripts:**
  - `src/hiring_audit.py` — creates the probe-vs-human audit tables and logs
  - `src/extract_vectors.py` — creates the model-specific warmth/competence direction vectors used as probe axes
- **Inputs:**
  - `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
  - `data/raw/SocialPerceptions-Predict-Callback-main/README.md`
  - `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/code.R`
  - `data/processed/concept_vectors/`
  - `data/processed/concept_vectors_gemma3_27b/`
  - `data/processed/concept_vectors_llama31_8b/`
  - `data/processed/concept_vectors_qwen3_14b/`
- **Outputs:**
  - `results/tables/hiring_audit_gemma3_12b.csv`
  - `results/tables/hiring_audit_gemma3_27b.csv`
  - `results/tables/hiring_audit_llama31_8b.csv`
  - `results/tables/hiring_audit_qwen3_14b.csv`
  - `results/logs/hiring_probe_vs_human_gemma3_12b.json`
  - `results/logs/hiring_probe_vs_human_gemma3_27b.json`
  - `results/logs/hiring_probe_vs_human_llama31_8b.json`
  - `results/logs/hiring_probe_vs_human_qwen3_14b.json`
- **Figures:**
  - `paper/figures/fig16_hiring_probe_vs_human.{png,pdf}`

---

## Summary

The data used for Test 2 are suitable for the current paper. The audit score is:

**Overall score: 8.0 / 10.**

This is a solid external-validation dataset: it comes from a published labor-market meta-analysis, includes 24,220 human rater rows over 282 first names, has almost no missingness, and is directly aligned with the paper's warmth/competence construct. The model-side inputs are also complete: all four concept-vector directories contain balanced concept activations, finite vectors, model metadata, and no missing or duplicate names in the produced audit tables.

The main weakness is not data provenance or file integrity. The main weakness is **uneven human-rating reliability by name**: some names have 300+ ratings, while 44 names have only one rating. The current Spearman correlation treats every name equally, so a one-rating name and a 300-rating name have the same weight. This must be disclosed as a limitation.

Despite that caveat, the data are usable. Filtering to better-rated names does not weaken the headline patterns; it generally strengthens them. This indicates that low-rater names are adding noise rather than creating the reported effects.

**Decision:** proceed with this dataset for Test 2. The data should be described as a valid name-level human-perception benchmark with explicit reliability caveats.

---

## 1. What data are used in Test 2?

Test 2 asks whether the model's internal warmth/competence scores for names resemble human perceptions of those same names.

It combines two data sources:

1. **Human side:** `ratings/names/df_all.csv` from Gallo, Hausladen et al. (2024), containing human warmth and competence ratings for names used in North American labor-market correspondence studies.
2. **Model side:** `concept_vectors*/` directories, containing the warmth and competence directions extracted earlier from synthetic concept stories for each model.

For each of the 282 names, `src/hiring_audit.py` inserts the name into a neutral sentence:

> "The job applicant's name is X."

It then reads the model's residual-stream activation and projects that activation onto the warmth and competence direction vectors. This produces two model-side scores per name: `model_warmth` and `model_competence`.

Those model scores are compared with the human name ratings using Spearman rank correlation. A positive Spearman rho means the model and human raters order names similarly. A near-zero rho means little relationship. A negative rho means the model's ordering runs opposite to the human ordering.

This is not a hiring-decision test. It is a name-level alignment audit: do model-internal social representations for names resemble human name perceptions?

![Probe-vs-human alignment across four models](figures/fig16_hiring_probe_vs_human.png)

**Figure 16.** The figure reused from the Phase-7 consolidated report visualises the
name-level Spearman correlations audited here. It shows positive Gemma alignment,
Llama/Qwen warmth anti-alignment, and strong Qwen competence alignment.

---

## 2. Source and provenance

The human ratings come from the public replication repository for:

> Gallo, M., Hausladen, C. I., Hsu, M., Jenkins, A. C., Ona, V., & Camerer, C. F. (2024). *Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments*. PLOS ONE.

The local source repository identifies the project as a meta-analysis of North American labor-market discrimination and states that the `ratings` directory contains warmth and competence ratings collected via Prolific for names and categories. The local R construction script `0_data/ratings/names/code.R` builds `df_all.csv` by extracting, cleaning, and combining warmth/competence name ratings across the underlying correspondence-study name lists.

The source studies represented in the ratings file are:

| Study | Rows | Unique names | Unique raters |
|-------|-----:|-------------:|--------------:|
| bertrand | 3,784 | 36 | 108 |
| farber | 1,200 | 12 | 100 |
| flake_leasure | 7,663 | 76 | 101 |
| gorzig | 706 | 14 | 200 |
| jacquemet | 1,114 | 17 | 264 |
| kline | 7,663 | 76 | 101 |
| neumark | 235 | 123 | 95 |
| nunley | 800 | 8 | 100 |
| oreopoulos | 526 | 44 | 119 |
| widner | 529 | 12 | 190 |

The model-side vectors come from the project's own Phase 4-5 extraction pipeline. Each model has 50 high-warmth, 50 low-warmth, 50 high-competence, and 50 low-competence activation rows. The audit confirmed that the stored vectors and activation matrices are finite and have the expected dimensionality for all four models.

---

## 3. Structural audit

| Dimension | Value | Assessment |
|-----------|------:|------------|
| Human rating rows | 24,220 | Strong |
| Unique names | 282 | Good coverage for name-level audit |
| Unique raters | 787 | Strong |
| Source studies | 10 | Strong provenance |
| Human warmth scale | 0-100 | Good continuous range |
| Human competence scale | 0-100 | Good continuous range |
| Missing `warm` values | 0 | Clean |
| Missing `competent` values | 16 / 24,220 | Negligible |
| Audit output rows per model | 282 | Complete |
| Duplicate names in audit outputs | 0 | Clean |
| Missing values in audit outputs | 0 | Clean |
| Concept-vector conditions | 50 / 50 / 50 / 50 per model | Balanced |
| Concept-vector NaN / non-finite values | 0 | Clean |

Human rating distributions:

| Variable | Min | Max | Mean | SD |
|----------|----:|----:|-----:|---:|
| `warm` | 0 | 100 | 54.58 | 22.28 |
| `competent` | 0 | 100 | 56.35 | 22.71 |

The human warmth and competence ratings are correlated at the name level:

| Correlation | Value |
|-------------|------:|
| Pearson | +0.612 |
| Spearman | +0.610 |

This is not a data flaw. It means human name perceptions of warmth and competence are partially coupled, which is consistent with the broader Stereotype Content Model framing.

---

## 4. Rating-count imbalance

The largest limitation is the uneven number of human ratings per name.

| Per-name rater count | Value |
|----------------------|------:|
| Mean | 85.9 |
| Median | 16.5 |
| Minimum | 1 |
| Maximum | 309 |
| Names with 1 rating | 44 |
| Names with <5 ratings | 102 |
| Names with <10 ratings | 116 |
| Names with <20 ratings | 146 |
| Names with >=200 ratings | 76 |

This matters because `src/hiring_audit.py` computes one average human warmth score and one average human competence score per name, then correlates names equally. A name with one human rating receives the same rank-correlation weight as a name with more than 300 ratings.

This does not invalidate the analysis, but it limits how strongly we can interpret individual low-rater names. The correct interpretation is at the aggregate pattern level, not at the level of any single sparse-rated name.

---

## 5. Robustness to better-rated names

The most important audit finding is that the headline correlations are not created by low-rater names. When names with very few human ratings are filtered out, the main patterns usually become stronger.

### Gemma-3-12B

| Filter | N names | Warmth rho | Competence rho |
|--------|--------:|-----------:|---------------:|
| all names | 282 | +0.366 | +0.239 |
| n >= 5 | 180 | +0.591 | +0.460 |
| n >= 10 | 166 | +0.614 | +0.472 |
| n >= 20 | 136 | +0.691 | +0.591 |
| n >= 100 | 97 | +0.771 | +0.661 |

### Gemma-3-27B

| Filter | N names | Warmth rho | Competence rho |
|--------|--------:|-----------:|---------------:|
| all names | 282 | +0.396 | +0.272 |
| n >= 5 | 180 | +0.619 | +0.450 |
| n >= 10 | 166 | +0.643 | +0.471 |
| n >= 20 | 136 | +0.710 | +0.534 |
| n >= 100 | 97 | +0.684 | +0.567 |

### Llama-3.1-8B

| Filter | N names | Warmth rho | Competence rho |
|--------|--------:|-----------:|---------------:|
| all names | 282 | -0.300 | -0.063 |
| n >= 5 | 180 | -0.554 | -0.157 |
| n >= 10 | 166 | -0.589 | -0.165 |
| n >= 20 | 136 | -0.640 | -0.243 |
| n >= 100 | 97 | -0.571 | -0.247 |

### Qwen3-14B

| Filter | N names | Warmth rho | Competence rho |
|--------|--------:|-----------:|---------------:|
| all names | 282 | -0.193 | +0.465 |
| n >= 5 | 180 | -0.443 | +0.598 |
| n >= 10 | 166 | -0.462 | +0.607 |
| n >= 20 | 136 | -0.525 | +0.525 |
| n >= 100 | 97 | -0.548 | +0.499 |

Interpretation: low-rater names add noise. They do not explain away the positive Gemma alignment, the Llama/Qwen warmth anti-alignment, or the strong Qwen competence alignment.

---

## 6. Scored rubric

| Criterion | Score | Rationale |
|-----------|------:|-----------|
| Source provenance | 9/10 | Published PLOS ONE paper; public repository; local code reconstructs the rating file |
| Construct relevance | 9/10 | Direct human warmth and competence ratings for the same names used in the hiring benchmark |
| Coverage | 8/10 | 282 names is strong for name-level validation; 10 source studies represented |
| Missingness / file integrity | 9/10 | No missing warmth values; only 16 missing competence rows; model audit outputs are complete |
| Model-side input integrity | 9/10 | Concept-vector files are present, balanced, finite, and metadata-linked for all four models |
| Rating-count balance | 5/10 | Serious imbalance: 44 names have one rating; 102 names have fewer than five |
| Cross-study consistency | 7/10 | 82 names appear in multiple studies; aggregation is useful but loses some study-specific structure |
| Statistical robustness | 8/10 | Filtering to better-rated names strengthens the main findings |
| Cultural / domain generality | 7/10 | Strong for North American labor-market name perceptions; limited for broader cultural generalization |
| Interpretability of benchmark | 8/10 | Measures perceived name warmth/competence, not true personal traits; this is appropriate but must be stated |
| **OVERALL** | **8.0/10** | Good, usable benchmark with one major reliability caveat |

---

## 7. Limitations to disclose

1. **Uneven human-rating reliability by name.** Some names have hundreds of ratings, while others have only one. The current main analysis treats names equally in the Spearman correlation.

2. **Human ratings are perceptions, not ground truth traits.** The benchmark measures how people perceive names, not whether real individuals with those names are warm or competent.

3. **North American labor-market scope.** The name set and correspondence-study context are North American. Claims should not be generalized to global name perception without additional data.

4. **Study-level structure is compressed.** The audit aggregates ratings by name. For names appearing in multiple source studies, study-specific differences are not modeled in the main correlation.

5. **Warmth and competence are partially correlated.** Human ratings correlate around +0.61, and model concept vectors are also not orthogonal. This is substantively meaningful but limits any interpretation that treats warmth and competence as fully independent channels.

6. **Model-side directions depend on synthetic concept stories.** The probe axes come from the project's concept-story corpus. That corpus passed quality audit and is accepted for current analyses, but it still carries the previously documented caveats: mono-source generation and no independent human manipulation check.

---

## 8. Audit conclusion

The Test-2 data pass quality audit for the current paper.

The human benchmark is well sourced, construct-relevant, and large enough for the intended name-level alignment test. The model-side inputs are complete and internally consistent. The main limitation — uneven rating counts per name — should be disclosed clearly, but it does not block the analysis. Robustness checks show that the headline alignment and anti-alignment patterns are stronger among better-rated names.

**Final decision:** use this data. The project should proceed with `ratings/names/df_all.csv`, the four `concept_vectors*/` directories, and the existing `hiring_audit_<label>.csv` / `hiring_probe_vs_human_<label>.json` outputs as the trusted Test-2 data basis.
