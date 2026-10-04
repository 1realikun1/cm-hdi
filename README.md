# CM-HDI reproducibility repository

This repository accompanies the manuscript **“CM-HDI: A constrained multimodal method for computing regional human development estimates.”** It contains the analysis code and processed, region-level evaluation artifacts needed to inspect and reproduce the reported CM-HDI comparisons for the Chinese Mainland, the United States, and Indonesia.

CM-HDI estimates the health, education, and income components separately from a shared candidate library of Earth observation (EO), population, economic, and healthcare-access information. It then combines the three component predictions with a non-negative seven-term synthesis whose weights sum to one.

## What is included

- The nested component-selection and constrained-synthesis implementation.
- Scripts for metric recalculation, uncertainty estimation, paired inference, EO-removal analyses, threshold sensitivity, and neural-baseline verification.
- Frozen processed input snapshots for the model-selection experiments.
- Spatially held-out predictions, fold assignments, tuning records, synthesis weights, and audit files used in the manuscript and Supplement.
- A SHA-256 manifest generated from the public release.

The repository is intended for **evaluation and computational reproducibility**. Original EO rasters, third-party administrative boundaries, and the complete upstream raster-processing environment are not redistributed. The public sources used to construct those inputs are cited in the manuscript and Supplement.

## Repository structure

```text
.
├── README.md
├── DATA_AND_CODE_AVAILABILITY.md
├── requirements.txt
├── requirements-neural.txt
└── reproduction/
    ├── README.md
    ├── D01_... D105_...        processed analysis artifacts
    ├── verify_*.py             deterministic metric and archive checks
    ├── *_inference.py          uncertainty and paired-inference analyses
    ├── run_china_neural_baselines.py
    └── revision_code/
        ├── run_aggregation_experiment.py
        ├── run_constrained_aggregation.py
        ├── run_revision_sensitivities.py
        └── input_snapshots/    frozen region-level modelling inputs
```

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The matched FT-Transformer and residual-MLP refit additionally requires PyTorch:

```bash
python -m pip install -r requirements-neural.txt
```

## Fast verification

Run these commands from the repository root:

```bash
python reproduction/verify_metrics.py
python reproduction/verify_USA_metrics.py
python reproduction/verify_C_no_EO.py
python reproduction/verify_neural_baselines.py
python reproduction/compare_traditional_geometric.py
```

These checks use the archived held-out predictions and do not require the original rasters.

## Reproduce the main statistical analyses

```bash
python reproduction/bootstrap_uncertainty.py
python reproduction/paired_comparison_inference.py
python reproduction/threshold_sensitivity.py
python reproduction/eo_incremental_inference.py
python reproduction/cross_country_eo_inference.py
python reproduction/benchmark_threshold_profile.py
```

The bootstrap and sign-flip analyses use the frozen seeds and replicate counts documented in the scripts and audit files.

## Refit the constrained synthesis

The frozen region-level inputs and split definitions are stored under `reproduction/revision_code/input_snapshots/`.

```bash
cd reproduction/revision_code
python run_aggregation_experiment.py
python run_constrained_aggregation.py
```

This reruns the component-selection and second-stage synthesis calculations from the processed snapshots. It does not rebuild population-weighted EO embeddings from source rasters.

## Interpretation boundaries

- EO-removal analyses estimate incremental predictive information conditional on the archived learner, inputs, and spatial validation design; they do not establish a causal effect of EO.
- Synthesis coefficients are predictive calibration parameters, not normative welfare weights.
- The matched geometric-mean comparison uses the same held-out component predictions and isolates only the final component-to-composite mapping.
- Administrative identifiers refer to statistical regions, not individuals.

## Citation

Please cite the accompanying CM-HDI manuscript. The final bibliographic record and DOI will be added after publication.

## Code and data availability

See [DATA_AND_CODE_AVAILABILITY.md](DATA_AND_CODE_AVAILABILITY.md) for the repository statement and the boundary between public reproducibility artifacts and upstream source data.
