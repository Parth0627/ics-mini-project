# Experimental protocol, version 0.1

This is a proposed protocol for approval after acquisition and schema audit. No genuine-data experiment has run. The executable runner supplies an initial numeric baseline; confirmatory results need the additional gates below.

## Experiment matrix

| ID | Source and held-out evaluation | Required evidence | Current readiness |
|---|---|---|---|
| E1 | UFU-Do53 → held-out UFU-Do53 | Released-split reproduction plus separate domain/capture/time evaluation; known labels and causal feature construction | Runner available; data/split/normalization/typed schema audit pending |
| E2 | UFU-DoH → held-out UFU-DoH | Released random-split reproduction plus session/capture/time isolation | Runner available; archived headers and window/session metadata pending |
| E3 | UFU-Do53 → UFU-DoH | One extractor, compatible observables, units and windows; fixed target holdout | Blocked on compatible genuine representation |
| E4 | UFU-DoH → UFU-Do53 | Same as E3, opposite direction, source-only tuning | Blocked on compatible genuine representation |
| E5 | Family-only and leave-one-family-out | Audited explicit groups; common frozen split/threshold; new fit for every subset | Design specified; controlled ablation automation pending |
| E6 | All common vs importance-k vs SATFS-k | Same source rows/model/hyperparameters; k=10/15/20 only when eligible; target test frozen | Ranking and controlled fitting implemented; genuine compatible data pending |
| E7 | Independent CIC-Bell/CIRA benchmark fits and optional verified transfer | Each dataset's actual target definition and split; label benchmark fits separately from transfer | Data/access/schema pending |
| E8 | Optional DNSTunnel supplied-split fit and separate external transfer | Licence, true window schema and company/domain/history separation | Optional; licence/schema/transfer pending |

Primary model families are LR, KNN, SVM, RF, XGBoost and MLP. Initial factories have fixed documented defaults. A final comparison must use the same outer split, data fraction, feature representation and source-only tuning budget for every family. Scale only inside pipelines; no test-fitted imputation, correlation filtering, resampling or selection. Record convergence warnings instead of silently accepting unfinished optimization.

## Dataset and split contracts

Raw input stays immutable. Record exact URL/revision/licence, local SHA-256, extraction version, observation unit, feature definitions/units, labels and their origin. Preserve separate row/capture/session/domain/campaign IDs and timestamps. Exclude scenario names, endpoints, collection paths, group IDs, timestamps and target-derived fields from predictors. Train-only transforms do not cure leakage in precomputed aggregate features.

The initial source baseline uses train/validation/test fractions 0.60/0.20/0.20 and seed 42. This is an **engineering smoke protocol**, not the authors' approved published split. Labels may stratify source development splits; source groups, when configured, stay intact and actual row fractions can differ from requested group fractions. Do not search seeds until test classes or metrics look favourable. Report infeasible partitions and revisit dataset design.

For final reproduction retain published test memberships where released. For robust evaluation, predeclare outer held-out captures/domains/campaigns or chronological captures. Reserve enough independent positive groups, not just enough positive rows. Preserve supplied train/test files without re-randomizing them; externally supplied split ingestion is a subsequent runner extension. Overlapping windows and shared causal histories require purge gaps based on the actual window/history length. Do not combine reproduction and leakage-controlled results in one table.

In transfer runs target sampling is **label blind**: reserve fixed target test groups first, then an adaptation subset from the remainder. Adaptation membership depends only on the predeclared group/time/sample rule. Target validation/training portions in the config are unused reserves; they supply no labels or transformations to the learner. Keep target test membership identical in zero-shot and assisted experiments. Label custodians can check ingestion and report class counts, but must not tune a target split or adapt using target class balance.

The current loader globally removes exact feature/label duplicates, retaining the first row ID, and rejects contradictory labels for identical selected measurements. This is conservative and must be reviewed for real observations: coarse features can naturally collide. A final deduplication rule should use observation provenance, not merely a reduced predictor subset. The runner uses the full approved feature set for all within-run deduplication and partitioning, then selects subsets. Cross-dataset exact-overlap refusal is an initial contamination gate, not proof that every collision is leakage.

## Strict zero-shot and adaptation-assisted comparison

Strict zero-shot uses source development information only. Evaluate A all common features and B source-importance top-k; ordinary target-assisted SATFS is unavailable. Do not call SATFS with target test values or relabel adaptation as zero-shot. Within-dataset runs likewise compare all and importance-k; a validation-as-target variant would answer a different question.

Adaptation-assisted E6 adds C SATFS-k. Use a separately reserved, unlabelled target adaptation sample with the identical source rows and held-out target test as A/B. Importance comes from the source-trained all-feature pipeline evaluated on source validation. Raw permutation drops use positive-class average precision; negative drops clip to zero only for normalized ranking. All-zero importance fails explicitly. Histogram supports fit raw source training observations only. Persist both probability mass vectors, edges, importance variability and complete rankings.

Preregister k=10,15,20. Unsupported k fails explicitly in a single run and is marked in a sweep; never clamp silently. k equal to eligible feature count is equivalent to all features. Each subset trains a fresh copy of the same model with the same fixed source hyperparameters and seed. Changing k after viewing target results is exploratory and needs a fresh target holdout for confirmation. No target labels choose model, subset, bins, hyperparameters, threshold or stopping.

## Metrics, thresholds and cost

Positive class is explicitly mapped to integer 1. Save precision, recall, F1, false-positive rate, ROC-AUC, AP and separately trapezoidal precision–recall area. PR areas depend on target prevalence; save both class counts and the prevalence baseline. Save raw score type and predictions. SVM uses decision-function scores and fixed threshold 0; probability models use threshold 0.5. These fixed thresholds are the initial baseline. If source-validation F1 or a source-FPR budget selects a later threshold, freeze it before transfer and record the objective/tie rule. Never recalibrate on the target holdout.

Training times cover imputation/scaling/model fitting; inference covers preprocessing and scored batch predictions. Save row count/batch size, model fit time, selector overhead separately, total time, rows per second, thread configuration, hardware, memory and versions. A single small batch is an engineering timing, not stable production latency. Full-size RBF SVM, KNN and SHAP have substantial laptop scaling costs. Begin with a group-preserving smoke subset, then measure capacity before increasing size.

## Repeats, uncertainty and reporting

Proposed confirmatory seeds: 42, 43, 44, 45, 46. Freeze group assignments for model-randomness repeats; independent outer capture splits answer a separate generalization question. Pair A/B/C test rows and seeds. Report every direction/k/model outcome, mean/dispersion and paired AP/F1/FPR differences. Compute capture/session-level paired bootstrap intervals for the primary AP difference, resampling the same groups for every arm. Row bootstrap is secondary where independence is defensible. Five model seeds on one capture are not five independent deployments. Uncertainty automation and multi-seed orchestration remain follow-up work.

E5 must list actual columns and save family-only/leave-family-out fits on one fixed partition. Feature groups can overlap only if the design says so. Report observed effects without inventing an expected drop. SHAP is computed after evaluation freeze on a recorded subset, with saved background/explanation row IDs and raw values. Test-set explanations must not feed another feature/hyperparameter decision on that same test. SHAP workflow is planned; the dependency is installed, not a claim that explanations already ran.

Report an outcome status independently of code execution: not run, data/schema gated, synthetic demonstration, executed awaiting audit, diagnostic proxy or validated for a specific claim. Keep missing metrics null, never zero. All genuine experiments presently remain not run. The final manuscript must cite each result's run ID, data hash, split hash and prediction artifact; guide-reported scores cannot populate new results.

