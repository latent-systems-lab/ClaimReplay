# Tutorial: compare an external claim-length policy

This tutorial uses only an authored 20-unit fixture. It validates extension,
budget, metric, export, and determinism behavior; it does not measure real-world
ranking quality.

## 1. Install and run the default example

Follow `docs/installation.md`, then from the extracted release candidate run:

```bash
claimreplay examples/fixtures/toy_response.json \
  --policy feature --feature priority --task unsupported --budget 2
```

## 2. Inspect the external policy

`examples/claim_length_policy.py` is not part of the core package. It subclasses
`VerificationPolicy`, receives only `PolicyView`, and defines length as Python
Unicode code points (`len(text)`), including spaces and punctuation, without
normalization. Equal lengths use opaque unit ID as a deterministic tie-breaker.

Ordinary policies may use response-unit text, opaque IDs for tie-breaking, and
documented deployment-visible features. They must not use assessments, native
labels, evaluator metadata, privileged evidence, or provenance-derived labels.

## 3. Compare three policies

```bash
python examples/run_external_policy_tutorial.py \
  examples/fixtures/toy_response.json tutorial_run_1
```

The runner compares ClaimLength, seeded Random, and Lexical Check-Worthiness on
the identical response and requested budgets 5%, 10%, and 20%. It exports:

- `rankings.csv` — complete ranking and scores;
- `selected_units.csv` — selected IDs and text;
- `metrics.csv` — recall, precision, assessment coverage, counts, and budgets;
- `replay_traces.jsonl` — complete machine-readable traces; and
- `manifest.json` — input/output hashes and zero-call receipt.

## 4. Understand quotas and missingness

Budgets are resolved per response using `ceil(fraction × unit_count)`. For 20
units, 5%, 10%, and 20% become 1, 2, and 4 slots. For a four-unit response, all
three would become one slot, a realized 25%. Report requested budget, resolved
slots, and realized fraction together.

An `UNASSESSED` unit remains in the ranking and consumes a selected slot. It is
not treated as supported or unsupported. Recall measures how many eligible
positives were found; selected-assessment coverage reports how many selected
slots had usable assessments. A policy can have apparently useful recall while
spending slots on unknown units, so report both.

## 5. Repeat deterministically

```bash
python examples/run_external_policy_tutorial.py \
  examples/fixtures/toy_response.json tutorial_run_2 \
  --compare-to tutorial_run_1
```

The command fails if any deterministic CSV/JSONL output differs. Elapsed runtime
is intentionally not part of the deterministic manifest.

The fixture text is deliberately simple and exposes conspicuous patterns.
Neither policy ordering nor metric value establishes general effectiveness.
