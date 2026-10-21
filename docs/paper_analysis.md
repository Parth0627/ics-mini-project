# Supplied-document analysis and engineering specification

Reviewed 2026-10-14. The initial workspace `/Users/avglarry0627/IcsMiniProject` contained only `ICS_Rsearch_Paper.pdf` (412,922 bytes), with no Git repository, source code, dataset, or run artifacts. The requested destination did not exist. Python 3.11.17, Node 22.23.3, npm 10.9.9 and Git were available on arm64 macOS. A byte-identical copy is preserved at `research/ICS_Research_Paper.pdf`; the original file is untouched.

## Source identity and evidence limits

The supplied PDF contains **20 pages**, titled **Project Overview and Methodology Guide**, dated 14 October 2026. Its subtitle matches the requested research-paper title. It is a team guide based on a draft and conversation summaries; its page-one evidence note states that the local repository was not inspected during its creation. It is not the original manuscript with an unpopulated results template. The user confirmed that the source is in the current project folder; no other PDF was present during inspection.

All pages and extracted tables were read, with rendered-page inspection of the cover, figures/captions, tables and bibliography. Figure captions 1–4 are present, but the supplied pages show no corresponding visible diagram artwork. Equations appear as inline metric definitions, rather than a full original mathematical-method section. The original draft's table numbering, equations, complete methodology, bibliography formatting, and blank result template cannot be independently reconstructed from this guide.

The pasted initialization request ends at E6 mid-list; no requirements after that point were present in the attachment. This setup implements the explicit initialization, verification, protocol and SATFS work and records E7/E8 from the guide as later evaluation tasks.

## Section-by-section engineering extraction

| Guide pages | Design extracted | Required implementation / evidence |
|---|---|---|
| 1–3 | Defensive detection of exfiltration-labelled Do53 and DoH records; six classifier families; generalization as the central challenge | Offline feature input, explicit binary labels, no covert-channel generation; retain authentic/proxy/synthetic provenance |
| 4–5 | RQ1/RQ2 detection, RQ3/RQ4 transfer, RQ5 observability, RQ6 cost, RQ7 explanations; hypotheses H1–H4 | Experiment matrix with outcome/status per question, not a claim that hypotheses are already proven |
| 5–6 | UFU primary datasets, CIC/CIRA benchmarks, optional DNSTunnel; published counts versus sampled-run counts | Source registry, streamed inventory, source hashes, rows before/after deduplication, explicit sampled size and class counts |
| 6–8 | Load/audit → split → fitted preprocessing → model → held-out metrics/explanations → report | Explicit feature allowlist and label mapping; train-only imputation/scaling; disjoint row/group manifests; per-run artifacts |
| 7–8 | Lexical, DNS/behavioral, temporal, flow and optional TLS families; MI, RF importance and SHAP | Use only actual audited features; typed categorical treatment remains necessary for RR types/classes; no invented lexical values for encrypted traffic |
| 8–9 | LR, KNN, SVM, RF, XGB, MLP; precision/recall/F1, ROC/PR area, FPR and timings | Shared partitions and model budgets; explicitly distinguish AP from trapezoidal PR area; store TP/FP/TN/FN and thresholds |
| 9–11 | E1/E2 within data; E3/E4 opposite-protocol transfer; mismatched matrices and observation units | Gate transfer on semantic feature definitions, units, observation windows and extractor versions; keep proxy experiments separately named |
| 11–12 | E5 remove feature groups, E6 top-k 10/15/20; conflicting reported completion states | Fresh model fits after split on actual selected columns; identical test rows; train/development-only rankings; no reused baseline metrics |
| 12–13 | E7 benchmarks; E8 DNSTunnel within and transfer; SHAP visual readings | Benchmark-specific label meanings and split evidence; saved raw attribution values; explanations are not causal proof |
| 14–16 | Reported near-perfect random-split scores, mixed-run summaries, lower grouped scores, synthetic/proxy/blocked statuses | Do not import those numbers into this project's measured results; group/temporal tests and provenance audits before generalization claims |
| 16–18 | Run IDs/config/prediction checks, previous Windows/Vite commands, checkpoint claims and viva guidance | Verify current executable commands locally; preserve originals; generate manifests and make silent/no-op runs impossible |
| 19–20 | Sixteen bibliography entries and final reporting caveats | Independent reference audit, corrected metadata, DOI/publisher links and explicit unresolved access limits |

## What is and is not evidence here

The guide's E1/E2 F1 scores, E3 proxy confusion counts, E4 AUC discrepancies, E7 small-sample results, E8 scores, SHAP magnitudes, archive missing-part counts, and claimed 235-file checkpoint are **reported historical statements**. No associated files exist in this initial workspace. None has been recomputed or accepted as local experimental evidence.

The guide reports zero exact primary common features; the official published schemas independently support an empty predictor-name intersection. This is not a proof that harmonized packet observables can never be derived. It is a gate against direct CSV matrix transfer. The guide's claimed 66 DoH archive parts differs from the currently observed official repository packaging; use the pinned inventory in `datasets.md`, not the historical missing-part count.

## Concrete architecture delivered

`dns_exfiltration.data` loads explicitly named numeric CSV features, validates class mappings/IDs, logs deduplication and prevents label columns/copies from entering the model. `audit` supplies bounded-memory file inventories. `splits` reserves test first, optional target adaptation second, then source validation/training; it supports disjoint group partitions and label-blind target sampling. `models` supplies six source-fitted preprocessing pipelines. `metrics` saves classwise confusion counts, positive-class AP, trapezoidal PR-AUC, ROC-AUC, precision, recall, F1 and FPR. `satfs` supplies the implemented candidate ranking. `experiments` fits each subset afresh and saves unique evidence directories. The CLI runs audits, explicit configs and a clearly synthetic integration demo.

The numerical baseline supports numeric features only. Integer encodings for DNS record type/class are not automatically meaningful continuous features; an audited typed representation and categorical pipeline are required before a final full-feature E1 comparison. The schema registry documents this concern. No packet extractor, traffic replay, live sensor or production alerting claim is included in initialization.

## Immediate research gates

1. Verify/reconcile the original manuscript when available; preserve this guide as the source actually reviewed.
2. Acquire selected CSV archive parts only during the acquisition stage; confirm licences, headers, true counts and normalization provenance.
3. Reconcile published Do53 split files and add capture/domain metadata. Regenerate noncausal full-data aggregates or exclude them for leakage-controlled evaluations.
4. Approve observation units and group/time partitions in writing before measuring final outcomes.
5. Build a common extractor for E3/E4 and cross-protocol SATFS, or transparently revise the question to compatible within-protocol external transfer. Do not populate blocked cells.
6. Run the preregistered model/seed/k matrix, preserve unsuccessful outcomes, and revise the paper from saved predictions only.

