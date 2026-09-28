# Data quality audit

An independent audit of the RSNA Knee Abnormality Detection bundle (metadata, DICOM headers on a
sample, and report text), run by the user against the actual competition data (not available in the
authoring container for this repository, so these findings are recorded as reported and attributed
here, not independently re-verified by Claude). Scope: integrity (leakage, corruption, missing files)
and content (label distribution, resolution, report duplication).

## Clean / no issue found

* No duplicate `StudyInstanceUID` in train or test; zero train/test leakage.
* Label columns are exactly `{0.0, 0.5, 1.0, NaN}` — no stray values.
* Gold/silver split is exact: 58 fully-labelled, 4,349 fully-unlabelled, 0 partially-labelled studies.
  No silver study has a stray non-NaN label leaking through.
* 100% plane coverage: all 4,407 studies (including all 58 gold studies) reference Sagittal, Coronal
  and Axial series in the metadata.
* DICOM files match the metadata: 813 sampled (study, series) pairs — 0 missing directories, 0 empty
  directories, 0 header-read errors.
* No sampled series falls below the pipeline's 10-slice floor.

## Genuine deviations found

1. **`Fluid_Sensitive` and `Fat_Suppression` are 100% redundant in this cohort.** Across all 24,371
   series rows, `Fluid_Sensitive == Fat_Suppression` with zero exceptions (crosstab is perfectly
   diagonal: 10,361 / 14,010 split, no off-diagonal cells). They are documented as two independent
   acquisition-technique flags, but carry identical information here — either they are always
   co-acquired in this cohort, or one column is a derived copy of the other. `select_series()`
   ([`src/kneemor/ingest.py`](../src/kneemor/ingest.py), inlined in the Kaggle notebook) scores
   candidate series as `-2*Fluid_Sensitive - Fat_Suppression`; with perfect correlation this is
   equivalent to weighting one signal 3x, not combining two independent ones. This does not change
   the ranking `select_series()` produces (the two terms move together), so it is not a bug — but the
   docstring's implication of two independent signals was wrong, and is now corrected in code with a
   comment pointing here.
2. **Large heterogeneity in native image resolution.** Sampled DICOM headers show at least 15 distinct
   `(Rows, Columns)` values, from 256×256 up to 1024×1024 (512×512: 354, 640×640: 182, 384×384: 140,
   320×320: 130 most common). Normal across scanners/protocols, but a real source of information loss
   once everything is downsampled to the pipeline's fixed 224×224 (Kaggle notebook) or 160×160 (CPU
   study): a 1024×1024 series loses far more relative detail at that target size than a 320×320 one.
   Neither pipeline currently accounts for source resolution when choosing target size or when
   comparing series.
3. **Slice counts per series range from 11 to 320 (median 30).** The pipeline samples a fixed number
   of evenly-spaced slices per series (16 in the Kaggle notebook, 24 in the CPU study) via
   `np.linspace`. For series near the low end of the range, this can sample the same source slice more
   than once (a duplication/interpolation artifact, not a crash — no sampled series fell below the
   10-slice floor, so nothing breaks, but a handful of the most slice-sparse series effectively see
   less independent information per requested slice than the fixed sample count implies).
4. **Report language mix roughly matches the documented eight-language claim** (heuristic,
   stopword-based check, not authoritative): English ~40%, Dutch ~15%, Turkish ~12%, Croatian ~9%,
   German ~6%, Bulgarian ~5%, Spanish ~4%, French ~1.5%, ~7% unmatched by the heuristic. Consistent
   with, not contradicting, §2.1's reported per-language counts.
5. **Template/boilerplate reports reduce the pool's effective diversity.** 46 distinct report texts
   are repeated verbatim across 177 studies. The worst offender is a single Turkish "everything
   normal" template appearing identically in 37 studies; two Spanish templates appear 14 and 12 times.
   These studies collapse to a handful of identical silver labels regardless of any actual imaging
   differences between them. Not a pipeline bug, but a real ceiling: the silver pool's nominal 4,349
   studies contain measurably fewer than 4,349 independent label signals, and any error-analysis split
   by report content (as in §5.6's by-language breakdown) should be read with this in mind.
6. **Series-per-plane duplication is common, not rare.** Only 4,556 of 13,221 (study, plane) pairs
   have exactly one candidate series; 6,636 have 2, 1,646 have 3, and one pair has 7 series for the
   same plane. Combined with finding 1, `select_series()`'s tie-break is doing real, single-signal
   work on more than two-thirds of all study-plane pairs, not occasionally breaking rare ties.

## Bottom line

The dataset is structurally very clean — no leakage, no corruption, no missing files. The deviations
are in *content*, not *integrity*: a redundant metadata column, heterogeneous native resolution, and a
meaningful cluster of template/duplicate reports that quietly reduces the effective diversity of the
4,349-study silver-labelled pool below its raw count. None of these findings change any measured
number already reported in this repository; they identify where future work (e.g. de-duplicating
near-identical reports before computing per-language error breakdowns, or an alternative series-choice
signal beyond `Fluid_Sensitive`/`Fat_Suppression`) could plausibly matter.
