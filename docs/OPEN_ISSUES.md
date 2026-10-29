# Evidence gaps and manuscript conflicts

These are release limitations, not silently repaired experimental choices.

1. **D1 sampling conflict (material).** Manuscript p.6 describes seed 42 reset for each record and random sampling without replacement from all stride-one candidates. `Untitled23.ipynb` cell 13 `dense_starts` instead seeds with `int(fingerprint(rid)[:8],16)`, includes all D2 starts, and samples the remainder; final cell 30 imports that function. Both notebook copies contain the same code. Saved D1 summary tables do not record selected starts or a sampling fingerprint, so they cannot adjudicate the conflict. The executable source-mode analysis is explicitly guarded and named; a manuscript-exact seed-42 implementation has not been passed off as recovered code. Resolve by recovering executed code/start manifests or correcting the manuscript and rerunning if necessary.

2. **Distributed Flower timing is not recovered.** Manuscript describes Flower and five GCP CPU VMs. The revision notebook embeds a persistent-SSH transport, not Flower. The Drive `distributed_comment16` folder contains client arrays/manifest, not the Flower deployment scripts or complete three-run timing logs. No Terraform, Docker image, firewall setup, region provisioning commands or versions have been invented. The SSH utility is preserved only in the separate originals archive. The seven attachment sources and research-folder inventory do not establish the exact historical Flower executable.

3. **Historical timing provenance unresolved.** Table 15's 191.23/639.67/182.88 seconds cannot be linked to a complete recovered canonical timing invocation. The new timing summarizer distinguishes round time excluding checkpoint I/O from a parallel training lower bound. It does not call either a replication of Table 15. Recorded canonical histories permit new summaries but are not proof of the historical table's exact protocol.

4. **Epsilon sweep reporting is narrower than saved execution.** Manuscript emphasizes ε=0.001 at server lr=1; saved `tuning_fedadam_epsilon_runs.csv` and notebook cell 22 cover both epsilons across all six rates (36 rows). Config preserves the recovered executed grid. Document that broader search in the manuscript if it informed selection.

5. **Patient-loss averaging wording.** Final privacy cell first averages window loss per recording and then averages recordings equally within each patient. This is not necessarily identical to pooling every patient's windows, especially where windows are excluded. The repository preserves recording-equal averaging and states it explicitly.

6. **Per-round test access is internally inconsistent in manuscript prose.** One paragraph says tests were evaluated every round for curves; canonical `training.run` loads only train and validation packs, while final evaluation is separate. Repository preserves the recovered validation-only selection/training behavior. Figure 2 can be rebuilt from validation histories; per-round test predictions were not recovered.

7. **Dependency information is observational, not a complete lock.** All 40 canonical config JSONs agree on Python 3.13.16, PyTorch 2.11.0+cu130 and the listed scientific versions. These saved values were not independently verified against a retrievable historical Colab image. Full transitive dependencies, GPU driver, RAM and memory peaks remain unknown. CPU validation during packaging uses a different environment, recorded separately.

8. **Full numerical rerun remains outstanding.** Synthetic tests, source parity, metadata/split reconstruction and saved-result verification pass. No full waveform download, 40-run training, D1 inference or patient-level privacy rerun was completed here. Original checkpoints are inventoried in Drive but not bundled or byte-validated; the 40 configs, completion JSONs and histories are included. Do not interpret saved reference outputs as freshly reproduced results.

9. **Inventory is bounded for bulk waveform/cache directories.** 53,418 Drive entries were observed. The connector lists at most 1,000 children per folder; 49 raw/processed/cache directories reached that bound. Their unlisted tail is not represented as inventoried or inspected. All observed study run folders and study-level result/protocol files were enumerated; raw/cache directory limits are listed explicitly in `provenance/inventory_limits.json`. An exported full Drive file listing is needed to claim an exhaustive inventory of every waveform/cache file.

10. **Baseline table rounding.** The saved pooled INCART ROC-AUC mean is 0.844704..., while Table 7 prints 0.844. Other pooled entries also appear truncated relative to saved precision. Use saved numerical tables as reference; review manuscript rounding rather than changing data.

11. **Release identity/rights.** The supplied manuscript DOI is a placeholder. Code license and repository citation/release DOI were not established. No license grant, Zenodo DOI or publication status has been fabricated.

Confidence: high in canonical engine/source linkage and saved D2 results; limited for D1 protocol alignment and historical systems execution.
