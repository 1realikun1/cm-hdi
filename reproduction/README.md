# Supplementary evaluation data

This archive accompanies *CM-HDI: A constrained multimodal method for computing regional human development estimates*.

CM-HDI is the complete component-to-composite procedure adopted in the manuscript. It independently estimates health, education and income, constructs seven component and geometric terms, and learns non-negative synthesis weights that sum to one. The primary synthesis comparator applies the conventional fixed geometric mean to the same held-out component predictions. Machine-readable files retain the historical strategy label `C` for CM-HDI so that archived scripts and field values remain reproducible. Labels `A` and `B` denote internal direct-HDI and shared-input complete-system references.

## Study samples and validation

The primary Chinese Mainland sample contains 240 scored cities in 24 province groups. Forty further mainland cities have a development-only role and never enter outer scores. The United States sample contains 3,104 counties or county equivalents with constructed 2019 HDI labels and 100-m population support. The Indonesian sample contains 443 regencies/cities in the adopted application scope. Component models and constrained synthesis weights are fitted separately in all three settings under province- or state-held-out evaluation.

Archived filenames and the internal `China` value in CSV country fields are retained for reproducibility; they refer to the Chinese Mainland study sample. The archived role `development_train_only` means exclusion from outer scoring. D14 records the Chinese Mainland inner-fold assignments. Population-weighted EO aggregation uses a 100-m GHS-POP sampling grid in the Chinese Mainland and United States and a 1-km grid in Indonesia.

## Recalculation and verification

The following scripts operate on the archived files without a network connection:

- `verify_metrics.py` checks the Chinese Mainland and Indonesian A/B/CM-HDI metrics in D01 against D02.
- `verify_USA_metrics.py` checks the United States 100-m A/B/CM-HDI metrics in D18 against D17.
- `bootstrap_uncertainty.py` reproduces D30--D32 and Fig. 8 from held-out CM-HDI predictions.
- `paired_comparison_inference.py` reproduces the complete-system CM-HDI-versus-B comparisons in D33.
- `threshold_sensitivity.py` reproduces D34--D35 and Fig. S1.
- `eo_incremental_inference.py` reproduces the United States fixed-input EO analysis in D36--D38.
- `cross_country_eo_inference.py` reproduces D39, D41 and D42 from D40 and D18.
- `benchmark_threshold_profile.py` reproduces D43 from D01 and D05.
- `verify_neural_baselines.py` checks the cohort, folds, predictions, metrics, inference records and hashes in D49--D55. Full neural refitting uses `run_china_neural_baselines.py` and additionally requires PyTorch.
- `compare_traditional_geometric.py` reproduces the matched conventional-synthesis metrics, paired inference and audit in D60--D62 from D56.

Python, pandas and numpy are required for the tabular checks; the bootstrap scripts also require matplotlib.

## File groups

Files D01--D105 correspond to the machine-readable artifacts described in the manuscript Supplement.

- D01--D15 contain Chinese Mainland and Indonesian predictions, metrics, cohorts, components, selections and fold audits.
- D16 preserves predictions from the earlier 505-unit Indonesian sensitivity fit. That fit used the retired synthesis and is not evidence for the adopted CM-HDI method.
- D17--D29 contain United States predictions, metrics, components, selections, diagnostics, prefit scope and label construction.
- D30--D35 contain CM-HDI prediction uncertainty, threshold sensitivity and paired complete-system inference.
- D36--D43 contain fixed-input EO and benchmark analyses. These analyses measure conditional predictive information and do not establish a causal effect of EO.
- D44--D48 preserve exploratory EO-removal experiments from the earlier fixed-synthesis analysis. They are archival records and are not used as evidence for the adopted CM-HDI results.
- D49--D55 contain the matched Chinese Mainland neural-system experiment.
- D56--D59 contain the adopted nested constrained-synthesis predictions, fold-specific coefficients, penalty tuning and audit for CM-HDI.
- D60--D62 contain the matched comparison between the conventional fixed geometric mean and CM-HDI, including outer-group inference and a source audit.
- D63--D72 contain selected cross-setting metrics, computational-cost records, standardized-Ridge checks, held-out-group permutation results and resource-reproduction audits.
- D73--D80 contain the current CM-HDI EO-removal experiment, basis diagnostics and power-mean sensitivity analysis.
- D81--D92 contain few-label transfer, the 505-unit Indonesian scope analysis, synthesis-weight diagnostics and revision-sensitivity protocols.
- D93--D102 contain matched regional EO gains and the Chinese 160-m/320-m spatial-resolution sensitivity analysis.
- D103--D105 contain Chinese component-level permutation summaries, repeated permutation results and the associated audit.

All threshold denominators count regions equally, and thresholds are applied to unrounded values. Region identifiers refer to administrative statistical units, not individuals.

## Inference boundaries

The uncertainty analysis uses 20,000 resamples and seed 20260927. Aggregate intervals resample complete outer province or state groups. Region-level intervals use held-out CM-HDI residuals, exclude the target's complete outer group, and calibrate within predicted-HDI quintiles. They represent predictive variation in the archived residuals and do not include uncertainty in the source HDI labels.

D33 pairs B and CM-HDI predictions for the same units, resamples complete outer groups, and applies two-sided group sign flips with Holm adjustment across the stated family. These are complete-system comparisons: CM-HDI changes both component allocation and final synthesis, so the results do not identify the effect of the synthesis module alone.

D34 evaluates archived A/B/CM-HDI predictions at 3%, 5%, 8% and 10% relative-error tolerances. D35 evaluates paired MAE and RMSE contrasts; positive stored estimates favour CM-HDI. D36--D43 hold the learner and tabular inputs fixed when assessing EO's incremental predictive information.

D49--D55 compare complete predictive systems on the same 74 regional variables, component targets, scored cities and outer folds. The FT-Transformer and residual MLP retain their archived final synthesis, whereas CM-HDI uses its constrained synthesis. This is therefore a matched full-system benchmark rather than an aggregation-matched ablation or an experiment that isolates downstream fusion.

D60--D62 provide the synthesis-isolating comparison. Within each setting, the conventional and CM-HDI rows use identical spatially held-out health, education and income predictions from D56; only the final mapping differs. Point estimates are equal or slightly favour the conventional geometric mean, and no paired contrast remains significant after six-test Holm adjustment. This result limits the empirical claim to a constrained, learnable generalization rather than predictive superiority over the conventional rule.

## Reproducibility scope

Source checksums identify the frozen inputs used in manuscript assembly. The archive supports evaluation reproducibility and rerunning the model-selection and synthesis stages from frozen region-level inputs in `revision_code/input_snapshots`. It is not a self-contained raster-processing distribution. Original EO rasters, third-party boundaries and the complete upstream geospatial environment are not bundled. Source datasets and citations are specified in the manuscript and Supplement.
