# DNS exfiltration research

An offline defensive research scaffold for comparing Do53/DoH detectors and investigating **Shift-Aware Transferable Feature Selection (SATFS)**. Initialization is complete; genuine-data experimental effectiveness and SATFS novelty are **not established**. No dataset archive has been downloaded.

The supplied PDF is a **20-page project methodology guide**, with reported historical results that lack local run evidence. Its unchanged copy is `research/ICS_Research_Paper.pdf`. The original manuscript/results template is not separately available. Read [the source analysis](docs/paper_analysis.md), [dataset verification](docs/datasets.md), [protocol](docs/protocol.md), [SATFS specification and prior art](docs/satfs.md), [reference audit](docs/references.md), and [implementation roadmap](docs/roadmap.md).

## Start on this Apple Silicon Mac

Python 3.11 is required. Python dependencies are pinned in `requirements-lock.txt`; the current arm64 macOS environment was tested. XGBoost also requires the Homebrew OpenMP runtime (`brew install libomp`), installed during initialization. Node/npm are available but are not needed for this Python CLI.

```sh
cd ~/Development/Projects/dns-exfiltration-research
sh scripts/bootstrap.sh
source .venv/bin/activate
dns-research doctor
dns-research demo
```

If using the staging workspace, `cd ~/IcsMiniProject` instead. The demo generates synthetic numeric observations, fits all-feature/source-importance models, then performs separately labelled zero-shot and adaptation-assisted toy-domain tests. **These are integration checks, not measured DNS/DoH detection results.** SATFS is unavailable in within/zero-shot mode because it requires a reserved adaptation sample.

```sh
# Six families, k=10/15/20, three modes on shared toy splits:
dns-research demo --all-models --all-k
# Audit an already acquired local CSV without fitting:
dns-research audit data/raw/dataset.csv --label label --output results/data-audit.json
# Run an explicit, audited config:
dns-research run configs/your-reviewed-config.json
# Score extracted numeric observations using a saved local model:
dns-research predict results/runs/RUN_DIRECTORY extracted-features.csv --output results/predictions.csv
```

`predict` expects the saved subset's canonical feature column names; raw packets/domain strings require an audited extractor first. Outputs are created exclusively and refuse overwrite. Each training run gets a unique directory and records configuration, environment, file hashes, explicit features, disjoint split IDs, rankings/bin supports, selected columns, separate trained pipelines, metrics and held-out predictions. A failure leaves a failed manifest where a run directory was allocated.

## Data and scientific gates

The official UFU published feature lists share **zero exact predictors**. Current CSVs cannot directly establish primary cross-protocol transfer. Transfer configs require explicit matched definitions, units and observation windows; the runner does not infer compatible measurements. The official DoH extractor also differs from its published schema and uses 5-second windows. Do53 full-capture aggregates and released normalization need leakage review. Details and pinned sources are in the dataset audit.

Raw archives belong under `data/raw/`, derived data under `data/processed/`; both are ignored by Git. Research results and models are ignored as generated evidence and retained on disk. Never use a full repository clone as an accidental multi-gigabyte download. Real-data example configs are intentionally gated until paths, labels, typed feature semantics, grouping and provenance are audited. Their acknowledgement fields record an audit decision; they do not automatically certify authenticity.

The baseline loader supports **numeric CSVs only**, loads training inputs in memory, removes full-feature duplicates and rejects conflicting/label-copy rows. Streaming `audit` avoids a full-memory inventory, but does not certify duplicates or provenance. Integer record types/classes need typed categorical support before final scientific full-feature comparisons. Existing released split ingestion, causal packet feature extraction, controlled family ablations, confidence intervals, SHAP workflows, and a live dashboard are documented next steps, not completed claims.

Source test rows and target test/adaptation groups are isolated. Target membership is label blind, and no target labels fit preprocessing, rank features, tune settings or choose thresholds. Every A/B/C subset gets a fresh fit on the same source training rows. Importance uses source-validation AP; JS supports use source training values only. AP and trapezoidal PR-AUC are reported separately. All-zero importance fails explicitly. Fixed threshold baselines use 0.5 for probability scores and 0 for SVM decisions.

## Repository map

```text
research/    Unchanged supplied PDF, extracted text
docs/        Source review, source registry, protocol, SATFS and references
configs/     Gated real-data examples and research plan
src/         CSV audit, loading, splits, models, metrics, SATFS, runner and inference
tests/       Numerical and scientific-invariant regression checks
scripts/     Bootstrap, safe workspace installation, evidence verification
data/        Local raw/derived/synthetic data (ignored)
results/     Unique measured run artifacts and initialization validation
```

To inspect or reproduce a result, locate its `run_manifest.json`, then `config.json`, `datasets.json`, `split_ids.csv`, `selected_features.json`, `predictions.csv` and `metrics.json`. Historical guide scores are not imported. The initialization validation report identifies the actual synthetic sweep and verifies metrics from saved predictions. Generated models are local serialized artifacts; only load your trusted run files.
