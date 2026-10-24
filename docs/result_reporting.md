# Export evaluation tables and budget curves

The offline reporting module consumes **completed evaluation summaries**, not
raw benchmark texts. It exports CSV, booktabs-compatible LaTeX and editable SVG.
It performs no downloads, API calls or neural inference. Keep its outputs separate
from source results. A pre-existing destination is rejected.

## Fully offline example

After installing the package from the release root:

```sh
python -m claimreplay.reproduce --config configs/toy_reproduction.json --data-dir . --output toy-results
python -m claimreplay.reporting --replay-results toy-results/synthetic/results.csv --output toy-tables
```

Expected: `replay_summary.csv`, `replay_summary.tex`, one SVG per task, and
`receipt.json`. The receipt hashes inputs and outputs without recording private
absolute paths. These are illustrative synthetic results, not ranking evidence.
The same command works with a condition's `results.csv` from native-data replay.
Do not mix different conditions in one input summary.

## Reference comparison formats

To render the reference comparison tables, explicitly supply analysis outputs:

```sh
python -m claimreplay.reporting --trec-results DATA/trec_summary.csv --priority-results DATA/priority_summary.csv --controlled-results DATA/controlled_summary.csv --controlled-results DATA/evidence_summary.csv --judge-results DATA/judge_summary.csv --output result-tables
```

Paths are user-controlled. Options may be omitted for unavailable analyses;
`--priority-results` requires `--trec-results`. No input files are bundled or
downloaded implicitly. Windows accepts the same commands with Windows paths.

Input contracts:

- TREC: `condition,policy,budget,topic_macro_recall,selected_total,selected_assessed,selected_unassessed,selected_assessment_coverage,realized_budget`.
  Three separate conditions: `without_prediction`, `with_prediction`, `webassess`.
  Each condition/policy/budget must have exactly one row. Optional `task` must be
  `review_need_detection` for this comparison. Standard policy identifiers are
  defined in `claimreplay.reporting.POLICIES`.
- Priority: same columns; the frozen text-only policy identifier is
  `llm:gpt-5.4-mini-review-priority-v1`. Without this input, its row is omitted
  explicitly, never filled with a fabricated result.
- Controlled: `dataset,target,queue_size,budget,policy,group_macro_recall`.
  Reference table: FEVEROUS/SciFact, native non-support/refutation, size 20,
  budget .2. Missing expected rows fail. Repeated reference rows across input
  files are coalesced only if every shared field agrees; conflicts fail.
  The receipt records the number of coalesced rows. FEVEROUS citation/evidence comparisons
  are unavailable and may be `N/A`; they are not invented.
- Judge: `condition,assessed_count,abstentions,exact_agreement,macro_f1,binary_agreement`.
  Agreement inputs must already use the intended nonabstaining population.

Outputs include `trec_primary`, `selection_coverage`, `controlled` and
`judge_agreement` CSV/LaTeX tables, plus separate TREC SVG budget curves.
Recall is rendered to four decimals; percentage agreement to one decimal.
No performance numbers or native topic identifiers are embedded in the tool.

## Reproducibility boundary

Rendering existing aggregate CSVs is **not** independent metric reconstruction.
The tool does not recompute judgments, policy scores, confidence intervals,
matched judge selections or sensitivity analyses. Those require their source
units, authorized assessments, scores/traces and analysis definitions. In
particular, the complete author-side historical analysis pipeline and its
private inputs are not shipped here. The public exporter covers three result
table formats and descriptive budget curves, not every manuscript figure.

Use `claimreplay.reproduce` for supported native-data metric reconstruction
and audit its trace provenance. Exact historical model replay additionally
requires authorized frozen outputs matching those inputs; fresh remote
inference is not guaranteed to reproduce historical rankings. Do not pool
TREC conditions or replace human reference judgments with surrogate labels.
Requested inspection fractions and realized ceiling quotas are different;
unassessed selections consume slots. Oracle remains a diagnostic upper bound.
