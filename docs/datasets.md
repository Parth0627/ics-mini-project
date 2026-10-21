# Dataset verification and acquisition plan

Access date: **2026-10-14**. This document records primary-source inspection, not a completed data audit. No dataset archives or captures were downloaded, no archive was extracted, no external registration form was submitted, and no upstream executable code was run. Small repository metadata and source files were read to verify the available artifacts and extraction logic.

Evidence terms used below:

- **Artifact metadata verified:** a file, byte size, or archive member is present in the official repository/index.
- **Publisher reported:** the official documentation gives the value; rows have not been independently counted.
- **Source code verified:** the published extractor/loader implements the described behavior at the pinned commit.
- **Unresolved:** actual downloaded-file verification or further author clarification is required.

## 1. Principal datasets: UFU-Do53-EXF and UFU-DoH-EXF

The starting point is the [authors' DNS-360-Dataset repository](https://github.com/CborgesUFU/DNS-360-Dataset). The inspected revision is [1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae](https://github.com/CborgesUFU/DNS-360-Dataset/tree/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae). The [GitHub recursive tree API](https://api.github.com/repos/CborgesUFU/DNS-360-Dataset/git/trees/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae?recursive=1) returned 657 entries and `truncated=false`.

| Dataset | Publisher-reported rows | Benign | Exfiltration | Calculated exfiltration share | Published predictors |
|---|---:|---:|---:|---:|---:|
| UFU-Do53-EXF | 5,351,064 | 4,812,119 | 538,945 | 10.0717% | 29 |
| UFU-DoH-EXF | 1,048,575 | 1,016,438 | 32,137 | 3.0648% | 28 |

The labels are documented as `Benign` and `Exfiltration`; both inspected RF loaders expect target column `label`. Do53 documents a pre-existing 80/20 train/test split, while DoH documents exact-row deduplication followed by stratified 80/20 splitting with seed 42. Reported deduplicated DoH counts are 1,022,257 total, 992,111 benign, and 30,146 exfiltration, with 204,452 test rows. These are publisher claims awaiting local verification. The README's Do53 percentage, 10.0706%, differs slightly from the ratio calculated from its counts. [Official README](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/README.md)

### Actual available packaging and sizes

There are no loose dataset `.csv`, `.pcap`, or `.pcapng` files in the inspected tree. Data is distributed as multipart 7-Zip archives. Byte totals below are Git blob metadata sizes for compressed volumes, excluding directory documentation. They are not expanded sizes or download checksums.

| Actual repository directory | Artifact names | Parts | Compressed bytes | Decimal size |
|---|---|---:|---:|---:|
| `data/CSV/UFU-Do53-EXF` | `dataset_Do53_normalized.7z.001`–`.002` | 2 | 51,208,455 | 51.21 MB |
| `data/CSV/UFU-DoH-EXF` | `dataset_DoH_normalized.7z.001`–`.005` | 5 | 123,631,039 | 123.63 MB |
| `data/PCAP/UFU-Do53-EXF` | `Do53.7z.001`–`.034` | 34 | 880,360,679 | 880.36 MB |
| `data/PCAP/UFU-DoH-EXF` | `DoH.7z.001`–`.569` | 569 | 14,900,809,878 | 14.90 GB |

CSV parts are 26,214,400 bytes each except the final Do53 part (24,994,055 bytes) and final DoH part (18,773,439 bytes). All listed data volumes together total **15,956,010,051 bytes**. The entire current tree's blob sizes total 16,007,319,313 bytes; this is not the size of a Git clone with history. [Pinned tree](https://github.com/CborgesUFU/DNS-360-Dataset/tree/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/data)

The README names `data/CSV/Do53`, `data/CSV/DoH`, and a different clone URL (`UFU-Do53-EXF-and-UFU-DoH-EXF`), while the live artifact tree uses the paths above. It also calls the dependency file `Requirements.md`; the tree contains `Requirements.txt`. Use the pinned, observed tree when preparing acquisition, rather than copying those setup instructions verbatim. Internal archive filenames, CSV dialects, extracted sizes, actual headers, sample counts, and split manifests remain unresolved.

### Published predictor schemas

These are documentation/extractor schemas, **not independently observed CSV headers**. Identifier or extra columns found after extraction require a fresh audit.

UFU-Do53-EXF, 29 predictors in documented/source order:

```text
query_length, subdomain_length, num_labels, max_label_length,
avg_label_length, entropy, ratio_digits, ratio_uppercase,
ratio_lowercase, ratio_special_chars, rr_type, query_class, ttl,
contains_base64_pattern, contains_hex_pattern, contains_dictionary_word,
ratio_vowels, ratio_consonants, num_distinct_characters,
mean_query_length_per_domain, std_query_length_per_domain,
mean_entropy_per_domain, num_unique_ips, num_unique_asns,
num_unique_ttls, mean_ttl, var_ttl, num_queries_per_domain,
ratio_failed_queries
```

Source: [Do53 feature dictionary](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/data/CSV/UFU-Do53-EXF/Readme), [Do53 extractor](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/Pipe/PCAP%20to%20FLOW-DO53.py).

UFU-DoH-EXF, 28 published predictors:

```text
src2dst_min_length, src2dst_max_length, src2dst_mean_length,
src2dst_stddev_length, dst2src_min_length, dst2src_max_length,
dst2src_mean_length, dst2src_stddev_length, bidirectional_max_length,
bidirectional_mean_length, bidirectional_stddev_length,
src2dst_min_time, src2dst_max_time, src2dst_mean_time,
src2dst_stddev_time, dst2src_min_time, dst2src_max_time,
dst2src_mean_time, dst2src_stddev_time, bidirectional_min_time,
bidirectional_max_time, bidirectional_mean_time,
bidirectional_stddev_time, num_src2dst, num_dst2src, flow_duration,
byte_count, handshake_time
```

Source: [DoH feature dictionary](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/data/CSV/UFU-DoH-EXF/Readme).

The exact predictor-name intersection between the principal schemas is **empty**. Lexical DNS fields are unavailable from encrypted packet payloads without a suitable observation point. Equal feature counts or broadly similar feature groups do not establish equal semantics.

### Important source-code discrepancies and audit gates

1. The [DoH extractor](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/Pipe/PCAP%20to%20FLOW-DOH.py) defines **31** predictors: the 28 above plus `sni`, `alpn`, and `ja3`. Its output schema adds `label`. The archived CSV header has not been checked. Reproduction must explicitly choose the published 28 columns; extra TLS/client identifiers require a separate experiment and confounding audit.
2. The same extractor defaults to `SAMPLE_MODE = "window"`, 5-second windows, and a 120-second flow inactivity timeout. This differs from interpreting every sample as a whole TCP flow. Adjacent windows from a connection should share a split group. `flow_duration` and its first-return-packet approximation `handshake_time` are expressed in milliseconds; the latter is not proof of a completed TLS handshake.
3. The [Do53 extractor](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/Pipe/PCAP%20to%20FLOW-DO53.py) concatenates records from captures and then computes aggregates over every row sharing a base domain. A random row split of that output would share aggregate context across partitions and can incorporate future records. For rigorous evaluation, split capture/session/domain groups before fitting aggregates, or regenerate causal aggregates using only prior observations.
4. Its base-domain function uses the final two labels, not a public-suffix-aware registered-domain parser. Multi-level suffixes require care. `subdomain_length` uses only the first label. `num_unique_asns` is always zero in this revision because no ASN cache is integrated. These facts limit equivalence with independently extracted features.
5. Do53 finalization coerces numeric features and replaces missing/unparseable values with zero. DoH fills missing numeric values with zero and missing TLS strings with `unknown`. Actual missing-value rates, infinities, constants, and conflicting labels have not been measured.
6. Both extractors recover scenario labels from capture filenames; their original names must be preserved as provenance. Filename, scenario, resolver, endpoint, capture time, and tool identity must not become model inputs.
7. The [Do53 RF loader](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/scripts/Machine%20Learning%20Models/ML_Do53_RF_V3.py) expects external train/test directories using author-local Windows paths. The [DoH RF loader](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/scripts/Machine%20Learning%20Models/ML_DoH_RF_V1.py) implements deduplication and the stratified split. Each takes every non-label CSV column as a predictor. Neither behavior establishes capture/domain separation; enforce an explicit predictor allowlist in this project.
8. Archive names say `normalized`, but the inspected extractors do not establish the origin, parameters, or fit scope of that normalization. Do not assume normalization was fitted on training rows only. Preserve released data for reproduction and label any leakage-controlled re-extraction as a separate protocol.

Published random-split results alone cannot establish transfer to new networks, times, resolvers, domains, or tunneling implementations. Cross-protocol E3/E4 and SATFS are gated on a measured semantic common-feature schema. A defensible route is to re-extract packet/flow observables with one extractor and one observation unit for both protocols, preserving group/time metadata separately. This remains a design requirement, not a verified dataset capability.

### License

The [actual `License. md` file](https://github.com/CborgesUFU/DNS-360-Dataset/blob/1b3e59ffc3d14f41ccbf143be155cf791eb5e7ae/License.%20md) places datasets, captures, documentation, and metadata under **CC BY-NC 4.0**. It separately specifies academic, educational, and non-commercial research terms for source code; do not relabel it as MIT. Preserve author/paper attribution, mark modifications, and retain the license with local copies. Inspecting offline data is sufficient for initial detection work. Acquisition does not authorize replay or traffic generation.

## 2. CIC-Bell-DNS-EXF-2021

| Item | Verified official statement / status |
|---|---|
| Source | [UNB/CIC dataset page](https://www.unb.ca/cic/datasets/dns-exf-2021.html) |
| Reported samples | 641,642 distinct benign; 323,698 heavy attack; 53,978 light attack |
| Reported capture volume | 270.8 MB |
| Schema | 30 named DNS features; stateful and stateless exports |
| Labels | Benign/light/heavy scenarios; exact released target strings and column unresolved |
| Split | Published shuffled 70/30 split; stateful window and step each 100 packets |
| License | Redistribution permitted with dataset and associated-paper citation |
| Access | [Official download form](https://cicresearch.ca/CICDataset/CICBellEXFDNS2021/) loaded but displayed a server error; fields require identity information. No submission made. |

Official feature names include `rr_type`, `rr_count`, `rr_name_length`, `rr_name_entropy`, `rr_type_frequency`, `rr`, `distinct_ns`, `a_records`, `unique_country`, `unique_asn`, `unique_ttl`, `distinct_ip`, `distinct_domains`, `reverse_dns`, `ttl_mean`, `ttl_variance`, `FQDN_count`, `subdomain_length`, `upper`, `lower`, `numeric`, `entropy`, `special`, `labels`, `labels_max`, `labels_average`, `longest_word`, `sld`, `len`, and `subdomain`. [Feature definitions](https://www.unb.ca/cic/datasets/dns-exf-2021.html)

Audit implications: `entropy`, `subdomain_length`, and `rr_type` match UFU Do53 names, but require definition, unit, window, and normalization checks. `labels` here is a predictor counting domain labels, not the class target. Similar names such as `ttl_mean`/`mean_ttl` are candidate mappings only. Benign records are reused between light/heavy exports; the table's benign subtotals sum to 949,711, exceeding the reported distinct benign count. Do not concatenate them and assume independent samples. The page also reverses light/heavy capture sizes in one paragraph versus its table, so preserve that discrepancy until archive verification. Exact file sizes/headers, deduplication, timestamps, provenance groups, and released numeric label encodings remain unresolved.

## 3. CIRA-CIC-DoHBrw-2020

| Item | Verified official statement / status |
|---|---|
| Source | [UNB/CIC dataset page](https://www.unb.ca/cic/datasets/dohbrw-2020.html) |
| Formats | CSV features and PCAP captures |
| Targets | First stage DoH/non-DoH; second stage benign/malicious DoH |
| Schema | 28 statistical features, plus extractor metadata |
| Traffic | Chrome/Firefox browsing; dns2tcp, DNSCat2, Iodine; four resolver services |
| Counts | Sum of published per-tool table: 917,300 browsing HTTPS flows; 249,750 malicious DoH flows. Browsing total combines benign DoH and non-DoH. |
| License | Redistribution permitted with DoHMeter and paper citation |
| Access | [Official download form](https://cicresearch.ca//CICDataset/DoHBrw-2020/) displayed a server error; no submission made |

Source implementation: [DoHLyzer repository](https://github.com/ahlashkari/DoHLyzer), pinned revision `86bd1dc970b919a04f12eeda679893beee32ec8d`. Its [flow output schema](https://github.com/ahlashkari/DoHLyzer/blob/86bd1dc970b919a04f12eeda679893beee32ec8d/meter/flow.py) includes `SourceIP`, `DestinationIP`, `SourcePort`, `DestinationPort`, `TimeStamp`, `Duration`, and `DoH`, and these statistical families:

```text
FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate
PacketLength{Variance,StandardDeviation,Mean,Median,Mode,
             SkewFromMedian,SkewFromMode,CoefficientofVariation}
PacketTime{Variance,StandardDeviation,Mean,Median,Mode,
           SkewFromMedian,SkewFromMode,CoefficientofVariation}
ResponseTimeTime{Variance,StandardDeviation,Mean,Median,Mode,
                 SkewFromMedian,SkewFromMode,CoefficientofVariation}
```

Braces above expand exact case-sensitive prefixes/suffixes; they are not literal columns. This is an extractor schema, not a verified downloaded header. The published UFU DoH names have no exact-name overlap. Cross-dataset flow features require semantic harmonization: DoHLyzer's [PacketTime](https://github.com/ahlashkari/DoHLyzer/blob/86bd1dc970b919a04f12eeda679893beee32ec8d/meter/features/packet_time.py) uses time relative to the first packet, while UFU computes inter-arrival times. Recompute compatible observables rather than simply renaming columns. Endpoint-based labeling creates an obvious identifier confound: keep IPs, ports, timestamps, and label-like `DoH` fields outside predictors. Separate resolver, browser/tool, connection, and capture groups for held-out tests. Released file sizes, exact benign/malicious counts, actual split policy, missingness, and class encodings remain unresolved.

## 4. Optional DNSTunnel2026

The [authors' Zenodo record](https://zenodo.org/records/20137065), DOI `10.5281/zenodo.20137065`, is accessible. It lists `DNSTunnel2026.zip`, **497.6 MB**, MD5 `e2a4c675b480f9c10b8629494d9ae7dc`. Its [archive preview](https://zenodo.org/records/20137065/preview/DNSTunnel2026.zip?include_deleted=0) verifies members `Archive.zip` (450.8 MB), `Readme.md` (16.6 kB), and `dga_quantized_model.bin` (52.1 MB), without downloading them.

Publisher-reported values: 37,447,179 benign and 31,135 tunneling samples (calculated prevalence 0.0831%); 38 predictors; approximately 5.3 GB processed CSV and over 10 GB raw data; 25 anonymized enterprise clients. Features use company/domain pairs, 5-minute windows advanced by one minute, and 24-hour history. It describes train/test CSVs preserving class proportions. Actual CSV headers, raw-data availability, split membership, and group isolation remain unverified. [Record description](https://zenodo.org/records/20137065)

This is a potentially useful **Do53 external-validation candidate**, not direct DoH ground truth. Overlapping windows, shared company/domain histories, cross-company popularity features, and an auxiliary readability model require separate leakage audits. Its feature schema differs from UFU's; its CSV cannot be substituted into a UFU model without verified harmonization. The readable page's Rights/License area is blank, and a small API request timed out; reuse permission is unresolved. Keep it optional until license, schema, provenance, and observational compatibility are established. Zenodo's download-traffic statistics are not the dataset's expanded disk size.

## 5. Storage and memory budget

These are project planning estimates, not measured dataset resource use.

| Workload | Known/estimated resource floor | Practical first-pass approach |
|---|---|---|
| UFU CSV acquisition | 174,839,494 compressed bytes (174.84 MB) | Acquire only the seven CSV parts when the download stage is authorized; list archive contents before extraction |
| Do53 numerical matrix | 620,723,424 bytes float32; 1,241,446,848 bytes float64 | 29 predictors × 5,351,064 rows; excludes labels, DataFrame index, copies, and model |
| DoH numerical matrix | 117,440,400 bytes float32; 234,880,800 bytes float64 | 28 predictors × 1,048,575 rows; excludes all overhead |
| UFU capture acquisition | 15.78 GB compressed | Delay complete capture acquisition until expanded sizes and a harmonized extractor justify it |
| DNSTunnel numeric matrix | Approx. 5.70 GB float32 or 11.39 GB float64 | 38 predictors × 37,478,314 reported rows; chunking required on a constrained laptop |

Use an initial 50,000–100,000-row per-protocol smoke subset, retaining both classes and provenance groups. A float32 100,000 × 29 predictor matrix alone is 11.6 MB. Measure actual process memory and decoded row footprint before scaling. Parsing strings, deduplication hashes, train/test/adaptation copies, preprocessing, SHAP, and trees can multiply the numeric matrix floor; dense one-hot encoding can be much larger. The current Air's RAM and free disk must be inspected before accepting a full-run budget.

Reserve approximately 10 GB free disk for initial CSV-only work as an adjustable planning budget; expanded CSV sizes remain unknown. Do not infer expansion ratios from the unusually small compressed Do53 archive. List/test archives, calculate a free-space gate from actual expanded member sizes, preserve original volumes, and use bounded/chunked extraction and loading. Full capture regeneration has no verified laptop budget yet. The publisher's hardware recommendations are server-scale rather than a guarantee that these small compressed CSV volumes require those resources.

Store raw archives separately from derived CSV/Parquet, split manifests, models, and results. Record source URL, retrieval timestamp, pinned revision, byte count, local SHA-256, upstream hash when available, archive member list, schema version, and label mapping. Keep original filenames and bytes immutable. Never estimate verified row counts by filenames or archive compression.

## 6. Acquisition and experiment acceptance checklist

- [ ] Confirm the existing workspace contains no untracked raw dataset files before acquisition.
- [ ] Recheck pinned URLs/access and license; acquire only selected artifacts with an explicit byte budget.
- [ ] Validate each part's size/hash, multipart completeness, archive integrity, safe member paths, expanded size, and free disk before extraction.
- [ ] Inspect CSV headers/dialect and count rows/classes in chunks; record categorical cardinality, nulls, infinities, constants, and duplicate features with conflicting labels.
- [ ] Resolve the principal DoH 28-versus-31 schema and window-versus-flow discrepancy.
- [ ] Identify normalization provenance, fit population, and any author-provided split files; record reproducibility gaps.
- [ ] Preserve source capture/domain/session/time identifiers in sidecar metadata. Audit train/validation/test/adaptation group and temporal overlap, exact duplicates, and near duplicates.
- [ ] Establish a shared feature dictionary with observation unit, direction, units, aggregation horizon, missingness rule, and extraction point before cross-protocol transfer or SATFS.
- [ ] Freeze external evaluation and target adaptation partitions before training. Target test labels must not affect preprocessing, feature selection, hyperparameters, or threshold selection.
- [ ] Retain released-split reproduction and leakage-controlled evaluation as distinct reported experiments.
- [ ] Resolve CIC server errors through official access; verify downloaded headers/counts instead of adopting mirror claims.
- [ ] Resolve DNSTunnel2026 license and archive contents before treating it as an executable external-validation plan.

No measured classifier results, corrected publication counts, compatible common-feature count, or demonstrated SATFS benefit follows from this initial verification.
