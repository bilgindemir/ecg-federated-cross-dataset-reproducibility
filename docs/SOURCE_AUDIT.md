# Source inventory and authority decisions

Audit scope: seven supplied attachments (one manuscript, six notebooks), both referenced Drive research roots, their observable descendants, and selected saved protocol/results/run metadata. Originals remain untouched. The separate `original-sources` archive holds byte-preserved attachments and fetched evidence, with SHA256SUMS.

## Attachment decisions

| Attachment | Cells | Classification | Reason / disposition |
|---|---:|---|---|
| FEDERATED_LEARNING_METHODS_FOR_ECG_CLASSIFICATION (1).pdf | — | Target protocol, with documented internal/source conflicts | Used to determine manuscript scope, split assignment, canonical settings and reported tables; not treated as proof of unobserved execution |
| FLjournal0526(1).ipynb | 61 | Historical exploratory notebook | Multiple preprocessing/training/evaluation variants and old result paths; not imported into final package |
| FLjournal0621.ipynb | 13 | Superseded dense sensitivity experiment | Restricted source-record subsets, capped dense windows, five reporting rounds, batch 256 and fixed 50-step budget; differs from final 40-checkpoint test-only D1 analysis |
| FLjournal0710(2).ipynb | 56 | Mixed historical/revision notebook; not a single canonical program | Contains both GroupNorm and manuscript BatchNorm architectures, repeated aggregation corrections, recording-wise INCART splits, fixed-step and double-imbalance branches, dense subsets, baselines and client exports. Useful provenance, unsafe to merge wholesale |
| ECG_FL_Revision_Lab_v2 (2).ipynb | 28 | Superseded revision engine | Five reporting seeds; earlier tuning and engine implementations. `common`, `runner`, `studies`, `training` differ from final matched source; shared preprocessing is not sufficient to establish final training |
| Untitled23.ipynb | 45 | Canonical engine plus final supplementary cells, with obsolete wrappers retained in notebook | Embedded engine hash exactly matches saved canonical run configs. Cells 23–25 and 27–34 contain final reporting stages. Pilot, old-checkpoint diagnostics, stale markdown and SSH timing are not canonical |
| Untitled23 (1).ipynb | 45 | Duplicate cell sources | All 45 source cells identical to preceding notebook; file bytes differ (outputs/metadata). Retain both originals; do not concatenate or execute twice |

## Research folder inventory

- [Revision root](https://drive.google.com/drive/folders/1Qq-GOFReKaaEfi-1x6Lp49waLucnUcBp): `data/metadata`, `data/manifests/9d68265b9e8940c9c1b1`, processed recordings, D2 packs, `study/runs`, `study/evaluations`, result tables, frozen selected parameters, and audit artifacts.
- [Historical root](https://drive.google.com/drive/folders/10rjazZSJhSHsxbyrBVj5AkLcF-m9DSdd): original datasets/caches, old checkpoints, comment-specific result folders, dense datasets, edge export, figures, and `distributed_comment16` arrays.
- `provenance/drive_inventory.csv.gz` records 53,418 observed entries with paths, IDs, types, sizes, modification times and source URLs. It is an inventory, not a claim that every payload was downloaded/read.
- `provenance/inventory_limits.json` lists 49 bulk raw/cache/processed directories that hit the connector's 1,000-child cap. All remaining observed folder traversal completed. No exhaustive claim is made for those capped directories.
- `provenance/legacy_inventory.csv` is the previously saved research audit; it is historical evidence, not a substitute for current traversal.
- Retrieved bytes include all 46 study-level files, the saved manifest/provenance, source metadata used for split validation, environment/count/invocation JSONs, and configuration/completion/history files for all 40 canonical runs. The archive keeps superseded tuning-stage and intermediate planning files; the clean repository excludes these from active configuration.
- Bulk waveform arrays and checkpoints were inventoried, not copied into the repository. The only observed standalone `.py` under the research roots was the historical edge benchmark; the final engine is embedded in notebook cells.

## Duplicates and supersession

`provenance/notebook_cells.csv` contains 248 source-cell hashes and first-line summaries. `duplicate_cells.json` identifies exact repeated code cells. `definitions.csv` records 627 function/class definitions with AST hashes, allowing same-name implementations to be compared without confusing repeated names with identical behavior.

The operative lineage is:

1. Earlier notebooks define changing raw-rate/resampling/windowing branches, restricted dense experiments, multiple model families and recording-level splits.
2. Revision v2 introduces auditable patient-level data construction and checkpoint provenance, but has earlier grids/reporting settings.
3. `Untitled23` updates the core engine; boundary-extension tuning follows stage 1; epsilon sensitivity then updates the frozen FedAdam setting.
4. Cell 23 trains the canonical 40-run comparison from the frozen selected settings. Separate final cells perform thresholds, weighting, baselines, label sensitivity, D1 and privacy/control analyses.
5. Saved configs/history/complete files validate this lineage: every canonical run has the same engine fingerprint, corrected data ID, BatchNorm model, batch 64, equal aggregation, one local epoch and 20 rounds.

`selected_hyperparameters_stage1.json`, `*_extended.*`, and `*_before_epsilon.json` are intermediate selection stages, not alternate canonical configurations. The final `selected_hyperparameters.json` and `selected_hyperparameters_after_epsilon.json` establish μ=3, FedAvgM lr=3, FedAdam lr=1 and ε=0.001. Original file names are retained in the archive.

## Material implementation conflicts

| Topic | Superseded/conflicting behavior | Final executable decision |
|---|---|---|
| INCART partition | split recording IDs; one patient may cross partitions | fixed source-patient manifest, no overlap |
| PTB unmapped labels | default-positive fallback | exclude unmapped diagnostic records |
| Architecture | four-convolution GroupNorm network with different head | three-convolution BatchNorm CNN, 38,305 parameters |
| Training budget | fixed 50 mini-batch steps described as an epoch | exactly client-window-count replacement draws per local epoch |
| Imbalance | balanced sampling plus positive loss weighting | sampler only; unweighted BCEWithLogitsLoss |
| Reporting seeds | pilot, five-seed, three-seed and ten-seed settings coexist | canonical 0–9; specified sensitivities 0–2; tuning 101/202/303 |
| Server settings | implicit momentum step, small μ, original epsilon | explicit final method settings recovered from saved configs |
| BN aggregation | old implementations differ in treatment of buffers | floating buffers average; integer counters max; adaptive parameter updates only |
| Thresholds | fixed 0.5 and tuned thresholds mixed across stages | fixed primary D2; separate validation-only tuning; D1 reuses tuned values |
| D1 | restricted subsets, generic bootstrap, final source sampler versus manuscript sampler | final source evaluation isolated behind documented discrepancy guard |
| Privacy | generic/repeated split audits versus final patient control | final canonical INCART train/test plus val/test control |
| Deployment | SSH engine versus manuscript Flower | no substitution; missing actual Flower infrastructure flagged |

Old bare state dictionaries are not imported or reevaluated as corrected held-out models. The recovered legacy audit explicitly reports four GroupNorm-compatible checkpoints without embedded training provenance and a reconstructed recording-wise split with patient overlap.

## Evidence strength

High confidence: source-to-run hash linkage, recovered canonical hyperparameters, data ID and split, saved main tables. Moderate confidence: numerical equivalence of the refactor beyond tested synthetic cases; a complete GPU scientific rerun remains necessary. Unresolved: D1 sampling alignment and historical systems timing.
