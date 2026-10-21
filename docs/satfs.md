# SATFS: candidate method and evaluation specification

Status: research hypothesis and evaluation specification; no measured effectiveness or novelty claim. Literature checked on 14 October 2026. The numeric prototype exists in `src/dns_exfiltration/satfs.py`; categorical support and the additional research analyses below are planned. This document contains no real-dataset experimental result.

The current selector accepts audited numeric original features only. Its initialization defaults are source-validation average-precision permutation importance, five permutation repeats, ten requested source quantile intervals, negative importance clipped to zero then divided by the largest positive importance, and equal-mixture base-2 JS divergence with no pseudocount. The JS routine directly sums masked KL-to-mixture terms, rescales finite mass vectors by their maximum before normalization to avoid sum overflow, and clips only cancellation errors within an absolute tolerance of \(10^{-12}\). It does not take a square root. Missing and finite out-of-source-support observations receive explicit buckets. It returns a per-feature score table; the experiment runner sorts that table to select subsets. Absence of an adaptation frame yields null divergence/SATFS values, not a fabricated zero-shot SATFS ranking. Categorical vocabularies, group/conditional importance, uncertainty intervals, additional comparators, and adaptive bin-count rules are prospective protocol requirements rather than implemented selector features. Defaults must be frozen before target evaluation; separate sensitivities can request twenty bins or ten permutation repeats.

## 1. Question, scope, and prior work

The candidate **Shift-Aware Transferable Feature Selection (SATFS)** asks whether selecting original features that are useful on labelled source data and exhibit less marginal source-to-target shift improves defensive DNS/DoH exfiltration detection. The intended contribution is a reproducible, inexpensive selection heuristic and its cross-protocol evaluation. Common feature names alone do not establish compatible measurement semantics: units, extraction tools, aggregation windows, packet directions, and observability must also match. SATFS cannot create lexical DNS information that a passive DoH observer cannot see.

Relevant verified primary literature already covers much of this motivation:

| Primary source | Relation to the candidate |
| --- | --- |
| Persello and Bruzzone (2016), [Kernel-Based Domain-Invariant Feature Selection in Hyperspectral Images for Transfer Learning](https://research.utwente.nl/en/publications/kernel-based-domain-invariant-feature-selection-in-hyperspectral-/), DOI [10.1109/TGRS.2015.2503885](https://doi.org/10.1109/TGRS.2015.2503885) | Selects original features for both discrimination and domain stability using conditional kernel embeddings and multiobjective optimization. The combined relevance/stability objective clearly predates SATFS; the application and estimator differ. |
| Gautheron, Redko, and Lartizien (2018), [Feature Selection for Unsupervised Domain Adaptation using Optimal Transport](https://arxiv.org/abs/1806.10861); [original conference paper](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/former-websites/2018/wp-content/uploads/2018/09/107.pdf) | Ranks original features using source/target similarity derived from optimal transport. Shift-based transfer-oriented feature selection is existing work. |
| Salakhutdinov, Kakade, and Foster (2012), [Domain Adaptation: A Small Sample Statistical Approach](https://proceedings.mlr.press/v22/salakhutdinov12.html) | Studies feature selection and domain-dependent variability across sampled domains. Stability need not be measured by one marginal histogram. |
| Ben-David et al. (2010), [A theory of learning from different domains](https://link.springer.com/article/10.1007/s10994-009-5152-4), DOI [10.1007/s10994-009-5152-4](https://doi.org/10.1007/s10994-009-5152-4) | Provides adaptation theory with conditions beyond small observable domain discrepancy. It does not justify a universal target-error guarantee for this product score. |
| Shui et al. (2022), [A novel domain adaptation theory with Jensen–Shannon divergence](https://www.sciencedirect.com/science/article/pii/S0950705122009200), DOI [10.1016/j.knosys.2022.109808](https://doi.org/10.1016/j.knosys.2022.109808) | Establishes bounds involving joint distributions and decomposes marginal/conditional shifts. Using JS divergence in adaptation is established; univariate SATFS histograms do not inherit these joint-distribution guarantees. |
| Zhao et al. (2019), [On Learning Invariant Representations for Domain Adaptation](https://proceedings.mlr.press/v97/zhao19a.html) | Supplies counterexamples and error bounds showing why source usefulness plus marginal invariance can fail under conditional or label-distribution shift. |
| Lipton, Wang, and Smola (2018), [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html) | Makes the label-prior-shift assumption explicit: class proportions can change while class-conditional feature distributions remain fixed. Marginal SATFS can penalize such informative features. |
| Cantone, Marrocco, and Bria (2024), [On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection](https://arxiv.org/abs/2402.10974) | Original author-submitted study of within- versus cross-dataset NIDS results, feature-selection strategies, and dataset heterogeneity. Its results support measuring transfer separately; they are not DNS/DoH SATFS evidence. |
| [SHAP-Guided Feature Selection for Cross-Dataset Generalization in Network Intrusion Detection Systems](https://ieeexplore.ieee.org/document/11563466/?signout=success) (IEEE Access, published 15 June 2026), DOI [10.1109/ACCESS.2026.3703481](https://doi.org/10.1109/ACCESS.2026.3703481) | The verified publisher abstract reports harmonized flow features and source-trained RF/XGBoost feature subsets evaluated across CIC datasets. Importance-guided cross-dataset IDS selection is existing work. Only publisher metadata/abstract were accessible here; full-paper comparison remains outstanding. |

These sources establish conceptual overlap, not identity with the exact SATFS product. The search is a targeted initial review, not an exhaustive novelty search. The exact equation, its variants, and DNS/DoH applications still require a documented wider search and full-paper comparisons before a novelty statement. A defensible eventual claim may be an empirical contribution, a transparent implementation, or a negative result. Do not call the method novel merely because this acronym or exact search phrase was not found.

## 2. Data access contract

Use the same approved, compatible feature universe \(\mathcal F\), source split, and target evaluation split for each comparator. Remove target labels, identifiers that encode labels or capture provenance, and post-event information from all model and selector inputs. Maintain row or capture manifests independently of the feature matrix.

| Partition | Allowed use |
| --- | --- |
| \(S_{\mathrm{train}}=(X_S,y_S)\) | Fit model preprocessing, numeric bin boundaries, categorical vocabulary, and the source predictive model. Fit imputation/scaling/encoding only here. |
| Source tuning/selection validation \(S_{\mathrm{val}}\) | Source-only hyperparameter selection and held-out labelled permutation importance. Nested source folds or separate tuning and importance folds are preferable. This validation is development data and cannot also serve as an unbiased reported test. |
| Source threshold/calibration validation | Choose any decision threshold or probability calibration using source labels under a predeclared rule. Use a separate fold or nested out-of-fold estimates when feasible. |
| \(S_{\mathrm{test}}\) | Final within-source evaluation after the pipeline is frozen. |
| \(T_{\mathrm{adapt}}=X_A\) | Separately reserved, unlabelled target sample; estimate target marginal histograms only for adaptation-assisted SATFS. No labels may be supplied to the selector. |
| \(T_{\mathrm{test}}=(X_T,y_T)\) | Final external evaluation only. Neither feature values nor labels inform scores, preprocessing, subset choice, thresholds, hyperparameters, or stopping decisions. |

Reserve adaptation/test groups before fitting. Prefer distinct captures, sessions, hosts, campaigns, or chronological periods according to the approved data protocol. An unlabelled row split can still leak repeated flows or adjacent-session structure; row disjointness alone is insufficient. Choose the adaptation sample using a predeclared label-blind rule and seed, never a target class-balanced sampling operation. Record its capture period, available observables, row/group count, and intended deployment analogy. Dataset custodians may audit labels after experiment freeze for reporting, but labels cannot alter adaptation membership or settings.

Two results families must remain distinct:

1. **Strict zero-shot:** preprocessing, feature selection, model fitting, tuning, and thresholds use source development data alone. Report all-common-feature and source-importance top-k comparators. Ordinary cross-domain SATFS cannot be calculated here because its \(D_j\) requires target data. Setting \(D_j=0\) reduces exactly to importance ranking and must be labelled that way. Estimating stability across source environments is a separate proposed variant, not target-assisted SATFS.
2. **Adaptation-assisted:** only \(T_{\mathrm{adapt}}\) supplies unlabelled target values for shift estimation. The classifier still trains on source labels. Use the same reserved \(T_{\mathrm{test}}\) for all-common, importance, and SATFS comparisons; mark the extra information used by SATFS. This is unsupervised adaptation-assisted selection, not zero-shot transfer.

## 3. Source importance \(I_j\)

Fit a source model \(f\) on \(S_{\mathrm{train}}\). First verify that it performs usefully on source validation; importance from a poor model is not a reliable estimate of feature usefulness. The recommended primary ranking metric is average precision (AP) for the positive exfiltration class, evaluated on the unchanged, naturally distributed source validation sample. Name AP explicitly instead of treating it as interchangeable with trapezoidal PR-AUC.

For a higher-is-better validation score \(s\), \(R\) independent permutations, and raw feature \(j\), define:

\[
a_j=\frac1R\sum_{r=1}^{R}\left[s(f,X_{\mathrm{val}},y_{\mathrm{val}})-s(f,\pi_{jr}X_{\mathrm{val}},y_{\mathrm{val}})\right],
\qquad w_j=\max(a_j,0),
\qquad I_j=\frac{w_j}{\max_{\ell\in\mathcal F}w_\ell}.
\]

Use the fitted preprocessing pipeline for every prediction. The numeric prototype permutes raw columns before transformation. If categorical or missingness-derived model columns are introduced, permute the entire raw feature so its one-hot columns and derived missingness indicator move together; independently permuting encoded children creates impossible feature combinations. Categorical grouping is planned, not implemented. Report \(R\), seed, source validation size, every raw mean drop, and permutation variability. The initialization default is \(R=5\); \(R=10\) or more is a separately recorded sensitivity analysis.

Negative drops must not become negative SATFS scores: retain them in diagnostics and clip only for the candidate ranking. If every \(w_j=0\), normalization is undefined and the selector is degenerate. Record an explicit failure/degenerate status and skip claims about meaningful selection; a deterministic fallback subset, if needed for a demo, must be labelled a fallback. Do not silently assign equal positive importance. If fewer than \(k\) features have positive importance, report how many zero-score features the top-k subset contains.

The denominator is a shared positive scalar, so it does not change ranking relative to unnormalized \(w_j(1-D_j)\). Sum normalization would also preserve ranking but change the score interpretation; fix one convention. Neither convention turns importance into a probability or a causal quantity. The algorithm follows the permutation-score-drop idea originating in [Breiman's Random Forests](https://doi.org/10.1023/A:1010933404324); held-out implementation and model-dependence are documented by [scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html).

## 4. Shared distribution estimation

Estimate shift on raw, harmonized features before model imputation. Imputation can hide different missing-value rates. Each feature has one fixed bucket map \(b_j\) fitted from \(S_{\mathrm{train}}\), then applied unchanged to source training and target adaptation values. Never construct separate source/target quantiles or bins, which can make shifted features look artificially identical. Never use pooled source/target-test bounds.

**Numeric features (implemented):** use the initialization default of ten requested source quantile intervals, with at least two requested intervals required by the API. Obtain the source finite-value minimum \(u_0\), maximum \(u_B\), and interior quantiles from source training only using NumPy's default linear quantile interpolation. Remove repeated boundaries and persist the effective edges and quantile convention; the effective number of nonempty intervals can be smaller than the requested count. Twenty requested intervals are a preregistered sensitivity. A sparse-feature rule that further reduces interval count using source sample size may be investigated later; it is not implemented or an automatic default.

Include separate `UNDERFLOW` for \(x<u_0\), `OVERFLOW` for \(x>u_B\), and `MISSING` buckets. Interior boundary equality goes to the interval on its right (`searchsorted(..., side="right")`); both training minimum and maximum remain in support. A source constant has one exact-value interior bucket plus both tails and missing. A feature with no finite source values cannot acquire target-fitted edges: the prototype instead uses two buckets, `UNKNOWN_FINITE` and `MISSING`, so later finite observations remain visible. Such a feature has no learned finite support and should be flagged in diagnostics; in the current source-fitted models it normally has zero importance. The prototype retains it in the common feature universe rather than automatically excluding it. Preserve tail probabilities in diagnostics rather than clipping target values to source bounds. Bin counts represent probability mass, not histogram density, so unequal quantile widths need no width correction.

Canonicalize invalid numeric values consistently before bucketing under a declared schema policy. A conservative default treats NaN, infinities, and parse failures as missing while reporting each cause separately. If legitimate infinities exist in a feature definition, use an explicit predeclared infinity policy instead; silently changing units or parse rules between protocols is unacceptable.

**Categorical features (planned, excluded from the current numeric loader/selector):** fit the vocabulary on source training only. For small finite vocabularies, use one bucket per valid category. For high cardinality, retain a predefined number of the most frequent source categories with deterministic tie handling, then reserve `OTHER_SEEN`, `UNSEEN`, and `MISSING`. Apply the exact same mapping in both domains: rare known source categories go to `OTHER_SEEN`; categories absent from source training go to `UNSEEN`. Never interpret arbitrary category codes as continuous measurements or use adaptation data to discover/retain category buckets. Persist vocabulary, cap, rare-category rule, normalization, and missing tokens. Reserved bucket names must be distinct from literal data values. High-cardinality host/domain identifiers require exclusion or justified behavioural aggregation because low divergence does not remove leakage. Audited numeric 0/1 variables may use the numeric prototype, but nominal integer codes must not be treated as numeric merely because their storage dtype is numeric.

For the common bucket set \(\mathcal B_j\), form normalized frequency vectors:

\[
p_{jb}=\frac{n^{S}_{jb}}{|S_{\mathrm{train}}|},\qquad
q_{jb}=\frac{n^{A}_{jb}}{|T_{\mathrm{adapt}}|},\qquad
\sum_b p_{jb}=\sum_b q_{jb}=1.
\]

Include missing rows in the denominators. Reject empty adaptation/source samples, nonfinite probabilities, negative counts, inconsistent bucket dimensions, or malformed schemas. Use the natural source distribution, before class rebalancing, oversampling, or SMOTE: the histogram is meant to describe an observed population, not a manufactured training class mix. If source sampling is unavoidable, record its selection mechanism and sample size. One fixed source sample, chosen without target labels, may be used for all selectors to fit a laptop budget.

## 5. Divergence and product score

Use the equal-mixture, base-2 Jensen–Shannon **divergence**:

\[
m_{jb}=\tfrac12(p_{jb}+q_{jb}),\quad
D_j=\tfrac12\sum_{b:p_{jb}>0}p_{jb}\log_2\frac{p_{jb}}{m_{jb}}
    +\tfrac12\sum_{b:q_{jb}>0}q_{jb}\log_2\frac{q_{jb}}{m_{jb}}.
\]

Zero-mass terms contribute zero; if both masses are zero, omit that bucket. No smoothing is needed to avoid infinite KL terms because the mixture contains each nonzero input mass. Adding pseudocounts changes the estimator and must be a separately reported sensitivity setting, rather than an invisible numerical fix. The equal mixture gives source and target distributions equal weight regardless of their sample sizes.

For normalized discrete distributions, \(0\le D_j\le1\) in bits, with zero for identical bucket distributions and one for disjoint supports. Thus:

\[
\operatorname{SATFS}_j=I_j(1-D_j),\qquad 0\le\operatorname{SATFS}_j\le1.
\]

Base-2 logarithms give the bounded normalization directly; natural-log JS would need division by \(\ln2\). Clip only tiny floating-point excursions at a stated tolerance; materially out-of-range values indicate an error. This definition is based on [Lin (1991), Divergence measures based on the Shannon entropy](https://ieeexplore.ieee.org/abstract/document/61115/authors), DOI [10.1109/18.61115](https://doi.org/10.1109/18.61115).

Do not confuse divergence with Jensen–Shannon distance \(\sqrt{D_j}\). In particular, [SciPy's `jensenshannon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html) returns distance; `base=2` followed by squaring represents this equation mathematically. SATFS based on \(1-\sqrt{D_j}\) is a different heuristic. The current prototype instead calculates the KL-to-mixture sum directly: this avoids a square root producing NaN when nearly identical distributions cause a tiny negative cancellation error before the root. It validates finite bounded results and uses the explicit \(10^{-12}\) clipping tolerance.

Rank descending by the product and break exact ties deterministically. The current runner uses canonical feature name for SATFS ties and source-importance ties; it does not add a raw-importance secondary tie rule. Save the complete score table and chosen subsets. Evaluate each supported \(k\in\{10,15,20\}\) without choosing the most favourable target outcome. Let \(p=|\mathcal F|\): skip and explain \(k>p\); if \(k=p\), flag equivalence to the all-feature set. A fixed-k run can include zero-score features, but this must be visible in results. Refit the same classifier family on each frozen subset using identical source fitting data and the declared source-only tuning policy.

## 6. What the score cannot establish

* **Prior shift:** \(P(X_j)=\sum_yP(X_j\mid y)P(y)\). A change in exfiltration prevalence can increase \(D_j\) even when the feature's class-conditional behaviour remains stable and useful. Removing that feature may harm transfer. Unlabelled histograms alone cannot identify the cause of shift; do not balance adaptation using its hidden labels. This limitation follows from the distinction formalized by [Lipton et al.](https://proceedings.mlr.press/v80/lipton18a.html).
* **Conditional/concept shift:** equal feature marginals do not imply equal \(P(y\mid X)\). For example, if a binary feature is equally common in both domains but the feature-to-label association reverses, \(D_j=0\) and a source-trained classifier can fail. Marginal invariance is therefore a heuristic, consistent with the counterexamples in [Zhao et al.](https://proceedings.mlr.press/v97/zhao19a.html).
* **Joint shift and interactions:** two domains can have identical one-feature distributions but different correlations and class-relevant combinations. SATFS scores each feature separately; it cannot certify stable joint representations. Interaction-only features may also receive little importance from a model that has not learned the interaction.
* **Redundancy and correlation:** a useful correlated feature can have a small permutation drop because another feature substitutes for it. Conversely, unconditional permutation can create off-manifold combinations. [Strobl et al. (2008), Conditional variable importance for random forests](https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/), DOI [10.1186/1471-2105-9-307](https://doi.org/10.1186/1471-2105-9-307), examines conditional importance and correlated predictors. Group or conditional importance are possible follow-up variants, not interchangeable primary results.
* **Estimator limits:** coarse bins hide within-bin shifts; many bins and small samples exaggerate noise; missing and unseen buckets collapse distinct causes; low observed JS is not evidence that the underlying distributions are identical. Finite-sample uncertainty can reorder close scores.
* **Model and environment dependence:** importance depends on the source learner, score, validation population, and capture conditions. A stable but irrelevant feature has low product because importance is low, while a source-important shortcut can still survive a low shift penalty. The method does not prove causality or robustness to unseen attack campaigns.
* **Fixed tradeoff:** multiplication imposes a particular stability/usefulness tradeoff, eliminating features with completely disjoint binned support. A shifted useful feature may be recoverable with calibration or a transformation, but this selector discards it. The heuristic is not a theorem about target risk.
* **Adaptation representativeness:** success on a reserved sample assumes its observable traffic resembles deployment. Later traffic, different resolvers, and collection artefacts can break that assumption. Additional target adaptation data also has collection/storage and computation costs.

## 7. Experiments, controls, and uncertainty

The mandatory primary comparison in both DNS→DoH and DoH→DNS directions is:

| Arm | Subset rule | Target information used before evaluation |
| --- | --- | --- |
| A | All approved compatible common features | None |
| B | Top-k by the same source permutation importance \(I_j\) | None |
| C | Top-k by SATFS \(I_j(1-D_j)\) | Reserved unlabelled adaptation features only |

Compare B versus C at the same k to measure the shift term; compare each against A to measure selection as a whole. The primary controlled analysis holds classifier family, source model/tuning result, training rows, split manifests, preprocessing policy, class weights, random seed, and threshold policy constant. If per-subset source-only retuning is additionally run, give every arm the same search space and budget and label it as a separate analysis. A and B should also be evaluated on the same target test used by C so test composition is not a confound.

Useful preregistered secondary controls are random-k selection, stability-only top-k by \(1-D_j\), and SATFS with the feature-wise divergence values randomly reassigned as a control for merely perturbing the importance ranking. Measure whether a general reduction in dimension accounts for an apparent benefit. Source-only stability across capture folds and alternative divergences are optional distinct variants. Synthetic controls should include no shift, pure prior shift, a source-specific nuisance feature, identical marginals with changed dependencies, and a reversed feature-label relationship. Their purpose is to demonstrate behaviour and failure cases; they cannot establish effectiveness on real DNS traffic.

Run a predefined set of at least five paired seeds, subject to capture availability. Distinguish model/permutation randomness from independent capture resampling: different seeds on one fixed test capture are not independent deployments. Pair each arm's split, model seed, adaptation membership, and source-importance permutations within a replicate. Keep target evaluation groups fixed when measuring model randomness; repeated outer capture splits answer a different question and should be reported separately.

Preregister one primary outcome, such as positive-class AP, and report precision, recall, F1, ROC-AUC, false-positive rate, confusion counts, class prevalence, training time, inference latency/throughput, and selector overhead. F1/FPR thresholds must be fixed from source validation. Record hardware, thread count, batch size, memory sampling, preprocessing time, and adaptation histogram time. A constant or one-class test label vector makes some metrics undefined; report that explicitly instead of fabricating a value. Always include class counts and trivial-baseline context because PR metrics depend on prevalence.

Report paired effect sizes \(\Delta\mathrm{AP}=\mathrm{AP}_C-\mathrm{AP}_B\) and corresponding differences versus A, per direction and k. Use capture/session-level paired bootstrap confidence intervals when such groups exist, resampling the same groups for all arms; row bootstrap can be overconfident for dependent traffic. Bootstrap source importance and unlabelled adaptation histograms separately for ranking uncertainty, keeping the source-fitted bucket maps fixed. Report rank intervals or selection frequency, B/C subset overlap, and missing/tail/unseen rates. Do not present permutation-repeat standard deviations as confidence intervals over deployment performance.

Declare confirmatory comparisons and handling of multiple models/directions/k values before evaluation, or call the study exploratory and report all outcomes. Do not pick a best k from target labels, discard losing seeds, tune bin counts after inspecting target performance, or use target-test SHAP values to revise selection. If SATFS fails, preserve the rankings and results, then examine redundancy, prevalence, support, and capture semantics. Post-evaluation labelled target diagnostics are allowed for interpretation once the evaluation is frozen, but any revised method needs fresh external validation.

The broader evaluation concerns align with [Arp et al. (2022), Dos and Don'ts of Machine Learning in Computer Security](https://www.usenix.org/conference/usenixsecurity22/presentation/arp). A successful outcome requires credible splits, transparent baselines, and honest uncertainty; completing the formula alone is not evidence of generalization.

## 8. Required saved artefacts and unresolved work

Save the schema/compatibility decision, source/adaptation/test manifest hashes, extraction and unit definitions, model/preprocessing parameters, source validation score, raw and normalized importance, permutation repeats, frozen bucket maps, \(p_j\), \(q_j\), JS divergences, SATFS scores, ties/fallbacks, chosen feature sets, seeds, thresholds, metrics with counts, runtimes, and software versions. Keep adaptation labels physically absent from selector inputs and test data outside fitting/selection interfaces.

Unresolved before empirical claims: obtain and audit the real dataset schemas; approve capture-aware partitions; verify compatible observability; establish actual eligible feature count and supported k values; complete wider and full-paper prior-art review; run both transfer directions and paired controls; retain negative findings; and revise the manuscript only from measured results. This document contains no experimental result and no statement that SATFS is effective or novel.
