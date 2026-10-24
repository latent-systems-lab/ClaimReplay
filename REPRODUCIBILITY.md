# Reproducibility

## Synthetic examples

Install the wheel or source distribution following README.md. Run the CLI,
`configs/toy_reproduction.json` and the external-policy tutorial without network
access. Repeat runs preserve selections and trace content. Toy labels are authored
examples and are not evidence of ranking quality.

## Your evaluation data

Use a supported adapter or construct frozen responses through the Python API.
Preserve unit identities, source provenance and native assessment semantics.
Ordinary policies receive only permitted text/features/evidence. Gold assessments
are used by the evaluator; custom producers remain responsible for not disguising
labels as features.

For supported benchmarks, acquire native data under upstream terms using
DATA_ACCESS.md. `configs/benchmark_evaluation.json` records expected file hashes,
counts, targets and seeds. Configure private source locations rather than changing
expected identities to bypass a mismatch.

```sh
python -m claimreplay.reproduce --config configs/benchmark_evaluation.json --data-dir DATA --output results
```

TREC conditions stay separate. Controlled claim queues are not coherent generated
answers. Budgets use a ceiling per response. Missing assessments are not treated
as negative labels; report selected-assessment coverage alongside recall.

## Tables and figures from completed evaluations

`python -m claimreplay.reporting --replay-results toy-results/synthetic/results.csv --output toy-tables`
exports CSV, LaTeX and SVG from an existing replay summary. See
[result reporting](docs/result_reporting.md) for the reference TREC, controlled
and judge table formats, required columns and explicit reproduction limits.
This presentation step is not a substitute for computing and validating metrics.

## Optional model scoring

The default inference backend is deterministic mock scoring:

```sh
python -m claimreplay.infer_campaign --config configs/campaign_mock.json --resource-root . --data-dir . --output mock-results --role priority
```

`configs/inference.json` is an example for separately acquired data and models.
The supplied priority and support-judge prompts are reusable scoring templates.
Model assets, evidence caches and input hashes must match the configuration.
Local models and paid APIs require explicit opt-in; they are never run by the
default quickstart. Fresh remote outputs are not guaranteed bitwise identical.

Retain your input manifests, policy configuration, model revision, cache identity,
seeds, full rankings and selections to replay your own experiment. Distinguish
inspection-slot budgets from policy-scoring time, tokens and computational costs.

This repository distributes the reusable framework, examples and user-facing
configurations. It does not distribute native datasets, model weights, human
labels, private model responses or the authors' experiment archive. It does not
claim exact reproduction of external results without their required artifacts.
