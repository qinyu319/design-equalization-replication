# IP&M supplementary experiments — 2026-09-16

**阅读入口：[含图表的本地阅读版](实验结果报告_阅读版.html)，或[实验结果与修改建议](实验结果与修改建议.md)。英文论文段落：[MANUSCRIPT_INSERTIONS.md](MANUSCRIPT_INSERTIONS.md)。**

This package contains executed supplementary simulations for *Auditing Assigned-Intensity Dependence in Social Simulation: A Design-Equalization Framework*, with the content-governance track as the intended thematic context. All numerical claims concern specified synthetic models. No real LLM, human-subject, or platform effectiveness experiment is included.

## What was executed

| Suite | Computation | Rows/datasets |
|---|---|---:|
| E1 | Independent crossing of dose, schedule, strength and topology at two reversion values | 6,480 runs |
| E2 | Governance configurations under matched exposure, including archived-selector sensitivity | 7,200 main + 2,400 sensitivity runs |
| E3 | Peer-averaging extension and paired no-input trajectories | 4,320 exposed + 240 control runs |
| E4 | Known-truth estimator validation under four covariance settings and three replication counts | 12,000 datasets; 6,000 with interval evaluation |
| E5 | Corrected hub selector replay with original configurations and seeds | 40,320 replacement runs in a 120,960-row archive |

E5 replaces existing runs and is not additional independent evidence. E1–E3 comprise 20,640 model evaluations including controls and sensitivity runs. Eight archived replay checks are separate verification runs. Independent replication blocks, not agents, determine reported Monte Carlo uncertainty.

## Main files

- `实验结果与修改建议.md`: Chinese findings, critical corrections, and changes recommended for each manuscript section.
- `MANUSCRIPT_INSERTIONS.md`: English methods/results/limitations and all six figure captions.
- `MODEL_SPECIFICATION.md`: exact dynamics, outcomes, noise covariance, estimators, pairing and uncertainty definitions.
- `PROTOCOL.md`: locally recorded prospective extension plan; not external preregistration.
- `ADDENDUM_hub_correction.md`: disclosed timing and scope of the later correction.
- `code/`: executable simulation, estimation, analysis, plotting, reproduction and validation code.
- `frozen/`: unmodified model and estimator snapshots; import these through `code/experiments.py`, rather than running their historical command-line entry points.
- `inputs/archive_reference/`: minimum original input snapshot needed to repeat E5 and source verification. This is the historical archive, not the corrected final dataset.
- `results/E1_runs.csv.gz`, `E2_runs.csv.gz`, `E3_runs.csv.gz`: complete new run-level outputs, including configurations, outcomes, seeds and stored trajectories.
- `results/E4_dataset_estimates.csv.gz`: dataset-level known-truth estimates and intervals.
- `results/E5_corrected_archive_120960.csv.gz`: corrected original-design archive.
- `results/E5_original_vs_corrected_allocations.csv`: comparison of original and corrected allocations for all 25 designs, three outcomes and four factors.
- `results/*allocations.csv`, `*contrasts.csv`, `*benefits*.csv`, `*endpoints.csv`: complete analysis tables, including outcomes not highlighted in the narrative.
- `figures/`: six figures, each exported as 300dpi PNG, vector PDF, editable SVG, and a source-data CSV.
- `validation/`: execution records, checks, provenance and source hashes.
- `MANIFEST_SHA256.csv`: delivered-file hashes, excluding the manifest itself and redundant JSONL run logs.

The source snapshot preserves historical descriptions and instructions as provenance. Its older instruction not to add ABM runs refers to the old analysis; it does not override the author's request for this separately versioned extension.

## Reproduce without changing delivered results

The execution environment was Python 3.13.7 on Windows, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.1, NetworkX 3.5, Matplotlib 3.10.6 and PyYAML 6.0.3. The tested executable on the author's machine is `D:/python/python.exe`. `requirements-experiments.txt` records dependencies for this extension. The old `requirements.txt` is retained for provenance and has a different Matplotlib pin; do not treat it as the executed environment.

From PowerShell in this package directory, create a **new sibling directory** and reproduce everything with:

```powershell
D:/python/python.exe code/reproduce.py --destination ../ipm_reproduced_20260916 --workers 6
```

The destination must not already exist. This copies the required code and inputs, verifies eight historical runs, executes E1–E5, recreates analysis tables and figures, then runs validation tests. It does not delete or overwrite the delivered folder. Runtime depends on the machine; the graph cache and six worker processes reduce repeated network generation. Recorded execution times include CSV compression and should not be used as a hardware-independent performance benchmark.

On another machine, create an appropriate Python environment and install `requirements-experiments.txt`; replace the executable path with that environment's Python. To avoid unnecessary installation, the current machine can use its existing tested environment. DataFrame numeric and trajectory equality with tolerances is the reproducibility criterion: gzip header timestamps and environment metadata can make compressed file hashes differ between otherwise equivalent regenerations.

## Rebuild tables and figures from delivered data

These commands update only derived files in the selected package directory; they do not rerun models:

```powershell
D:/python/python.exe code/analyze_results.py
D:/python/python.exe code/analyze_results.py --correction-only
D:/python/python.exe code/make_figures.py
D:/python/python.exe -m pytest code/test_experiments.py -q
```

For a fully untouched delivery, use `reproduce.py` instead. The raw simulation runner refuses to overwrite existing JSONL files. ZIP delivery omits redundant JSONL copies because compressed CSVs contain the retained outputs; regenerate into a new directory if another full run is needed.

## Numerical interpretation

- Dose is cumulative direct exposure events per agent, not unique reach.
- `B` is an allocation in squared-outcome units. `within_view_share` uses that view's total. New factorial views define new estimands.
- E5 `corrected_q`, `corrected_tau`, `corrected_rho` use corrected C-D50 denominators for the same outcome. Proportions become percentages only after multiplying by 100.
- Negative complete-U estimates are retained. Nonpositive ratio denominators are suppressed and counted; tiny positive denominators can still be unstable.
- `ci_low/ci_high` in raw mean tables are unadjusted t intervals for 30-block Monte Carlo means, not population confidence intervals.
- `B_low/B_high` and E5 percentile ranges are descriptive sensitivity ranges, not calibrated confidence intervals.
- E4 coverage has binomial Monte Carlo uncertainty, reported in the source table.
- E2 “benefit” means a lower specified simulated outcome; it is not directly measured harm reduction or equal-cost effectiveness.

## Verification status

Eight historical sample replays passed; new matrices have exact expected IDs and counts; direct exposure totals match assigned doses; all primary outcomes are finite; E3 no-input relative outcomes are zero; all archived nonhub rows are unchanged; original source hashes are unchanged. Ten tests cover the hub defect, zero intervention boundaries, paired-estimator identities, schedule budgets, no-input behavior, design uniqueness, preserved random targeting, seed separation, independent estimator implementation, graph-cache equivalence and nine new-run replays.

The entire computational reproduction was also executed successfully in an independent directory using only the packaged input snapshot. All 27 result CSV/CSV.gz files, including stored trajectory strings, matched the first execution (numeric tolerance rtol=1e-12, atol=1e-14); all ten tests passed again. See `validation/final_validation.json` and `validation/reproduction_console.log`.

Figures were visually inspected after rendering, and the HTML reading copy was checked in a local headless browser. A final validation record and file manifest accompany the package. The original Word manuscript has not been overwritten. The supplied English passages still need integration with the manuscript's numbering and references. `code/build_reading_copy.py` rebuilds the optional HTML reading copy using Markdown 3.9; it is separate from the numerical pipeline.

## Thematic scope and submission route

The official IP&MC2026 track page lists “Trustworthy Content Governance and Safe Generative AI in the Age of Large Language Models”. On the date checked, the track page listed 31 July 2026 for the conference deadline, while the participation page advertised late-breaking posters. An independent currently open journal route was not established from those pages. See the [official track page](https://www.elsevier.com/events/conferences/all/information-processing-and-management-conference/track-title) and [participation page](https://www.elsevier.com/events/conferences/all/information-processing-and-management-conference/participate).
