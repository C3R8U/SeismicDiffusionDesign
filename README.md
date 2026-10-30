# SeismicDiffusionDesign

Graph-conditioned mixed discrete/continuous diffusion for steel moment-resisting frame design, accompanied by a probabilistic seismic response predictor and a finite-element staging interface.

## Data

The input archive is the DesignSafe project **A Database of Seismic Designs, Nonlinear Models, and Seismic Responses for Steel Moment-Resisting Frame Buildings**, DOI: https://doi.org/10.17603/ds2-8yc7-1285. The project contains 621 independent steel-frame designs and 240 ground-motion histories, producing 149,040 design/record response combinations. The archive also includes site-specific seismic response datasets.

Keep the original `DesignSafe.zip` file outside this project and pass its location on the command line. The original publicly accessible archive is not repackaged here. The supplied project readme encourages research use; distribution rights should be checked against the current DesignSafe project page.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The reference research configuration uses PyTorch, a 12-layer graph-conditioned diffusion Transformer, 500 diffusion steps, and a five-member response ensemble. This CPU pilot uses a 3-layer, 96-dimensional generator and a single 128-dimensional response network. Training results in the results package belong to the pilot, not the full research configuration.

## Run

```bash
python scripts/run_pipeline.py --archive /path/to/DesignSafe.zip --profile pilot
python scripts/run_baselines.py
python scripts/run_ablations.py
python scripts/run_risk_guidance.py
python scripts/run_guided_sampling.py
python scripts/plot_results.py
python scripts/draw_architecture.py
python -m unittest discover -s tests -v
```

For the research-scale network, use:

```bash
python scripts/run_pipeline.py --archive /path/to/DesignSafe.zip --profile research
```

The research profile may require a modern accelerator and substantial runtime. If data preprocessing has already completed with the intended seed and archive, add `--skip-preprocess` to avoid re-reading the full archive.

## Data handling

- All 621 building design rows are assigned to exactly one of 435 training, 93 validation, and 93 test structures.
- The 240 record identifiers are divided into 168 training, 36 validation, and 36 test records. The archive does **not** provide earthquake-event identities sufficient to assert event-level independence.
- The response predictor learns from building/record pairs; the generator learns from independent building designs.
- Peak floor accelerations use the response-table conversion `in/s² × 0.0254 = m/s²`. Miranda ground-motion accelerograms are read in `g` and converted to `m/s²` for feature extraction.
- The W-shape names contain a nominal depth and a linear weight. Those are sufficient to derive nominal depth and cross-sectional area using a steel density of 7,850 kg/m³. The supplied archive does **not** provide a validated catalog of each section's moment of inertia or plastic section modulus. Those missing properties are never invented.
- Section layouts represent three story-wise groups: exterior columns, interior columns, and beams. A W-shape appearing in the known catalog does **not** guarantee compliance with beam-column design rules.
- Performance conditions used in the generator are formed from the 90th percentile across **training** ground motions. Test-event response summaries are not passed to candidate generation.

## Implementation

`src/preprocessing.py` reads design geometries, section configurations, response matrices, and ground-motion features directly from the user-supplied archive. `src/models.py` defines masked graph aggregation, the conditioned Transformer, continuous Gaussian and categorical diffusion, their consistency objective, and response prediction. `src/risk.py` contains empirical CVaR, differentiable proxy guidance, legal catalog projection, and active-learning scoring. `src/baselines.py` provides the conditional VAE. Experiment entry points are in `scripts/`.

`configs/default.json` documents the research-scale hyperparameters. The `--profile pilot` flag changes the model dimensions, training length, and candidate count to keep an initial CPU investigation practical.

## Finite-element analysis

The archive contains OpenSees TCL models. `scripts/opensees_interface.py` stages an **existing** model with shared TCL scripts. It does not claim to validate newly generated designs. Generating trustworthy new structural models requires AutoSDA or an equivalent verified section-property and connection-design pipeline. A supported `OpenSees` binary, ground-motion time-series inputs, and verified analysis settings are required before nonlinear response results can be reported.

To stage an existing model without executing it:

```bash
python scripts/opensees_interface.py --archive /path/to/DesignSafe.zip --building-id 0 --destination results/opensees_staging/building_0
```

## Scientific interpretation

Surrogate prediction metrics are measured on held-out building/record pairs. Proxy risk screening of a generated design is **not** a verified seismic design, and catalog membership is **not** engineering feasibility. The pilot's 4-epoch ablation comparison is only a software/architecture diagnostic. The package does not establish a verified seismic compliance rate, any reduction in steel use at equal structural performance, or relative performance of full-scale diffusion baselines. The original OpenSees software was not available in this execution environment.

See `RESULTS_REPORT.md` in the results-containing archive for measurements, figure descriptions, and execution status.


## Additional experimental analysis

The following experiments use the completed pilot checkpoints and published response data.
Run the data alignment and independent screening scripts after preprocessing and baseline model training:

```bash
python scripts/process_site_specific.py --archive /path/to/DesignSafe.zip
python scripts/audit_experiments.py
python scripts/train_risk_surrogate.py --epochs 35
python scripts/evaluate_risk_surrogate.py
python scripts/calibrate_risk_probabilities.py
python scripts/compare_design_generators.py
python scripts/catalog_projection_checks.py
python -m unittest discover -s tests -v
```

`TECHNICAL_FINDINGS.md` provides observed metrics and the remaining engineering limitations. The executed result package includes a risk-model checkpoint and console training milestones; an externally imposed execution time limit stopped the long training command after the epoch-25 printed milestone. No unobserved training epochs were added to the logs.

The engineering section catalog is available separately in the AutoSDAPlatform repository. Once obtained, process it with `python scripts/load_section_properties.py --catalog path/to/AllSectionDatabase.csv`. The catalog's presence does not automatically update checkpoints trained on nominal weight and depth.

## Extended hierarchical graph and engineering-property workflow

`ADVANCED_TECHNICAL_REPORT.md` describes the executed 12-block training runs and the verified boundary of the computational evidence.

```bash
python scripts/prepare_advanced_data.py --archive /path/to/DesignSafe.zip
python scripts/run_advanced_training.py --epochs 6 --seeds 42 2026 3407 --batch-size 32
python scripts/generate_postfit.py
python scripts/compare_risk_guidance.py --seed 42 --examples 2
python scripts/archival_active_learning.py --rounds 5
python scripts/prepare_opensees_cases.py --archive /path/to/DesignSafe.zip --buildings 0 756
python scripts/plot_advanced_results.py
python scripts/plot_generated_layouts.py
python -m unittest discover -s tests -v
```

The full-size architecture and source code do not imply that a full 300-epoch model has been trained. The installed runtime has no OpenSees binary, and the included original-model TCL input cases are staged, not executed. Real structural compliance values must follow an independent nonlinear engineering analysis.
