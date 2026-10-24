# Public API reference

## Canonical records

- `VerificationUnit`: immutable unit ID, text, source provenance, optional
  observable features, and optional evidence bundle.
- `GoldAssessment`: evaluator-only canonical support level plus native label.
- `FrozenResponse`: immutable units and assessments. `response_hash` excludes
  gold; `gold_hash` identifies assessments.
- `EvaluationTask`: explicitly maps canonical labels to positives and eligible
  labels. `UNASSESSED` is never silently negative.

## Policy boundaries

- `VerificationPolicy.rank(PolicyView, seed=...)` receives text, opaque unit IDs,
  and validated observable features only.
- `EvidenceAwareVerificationPolicy.rank(EvidencePolicyView, seed=...)` receives
  claim text and position-preserving cited evidence text slots.
- `PrivilegedOraclePolicy` is not a `VerificationPolicy`; replay dispatches it
  through an explicit evaluator-only path.

Policies that consume structured features should implement
`observable_feature_schema()` with `"number"` or `"string"` values. Built-in
feature and citation policies do this automatically. Validation runs before
`rank` and rejects missing/type-invalid configured fields and known evaluator-like
structured names/values.

The validator minimizes accidental leakage. It cannot establish semantic
provenance or detect a label intentionally encoded as an innocently named number.

## Replay

`replay(response, policy, task, budget, seed=0)`:

1. Constructs and validates the appropriate policy view.
2. Requires a complete permutation of unit IDs.
3. Resolves integer or fractional inspection budget.
4. Selects the ranking prefix.
5. Computes metrics using evaluator-only assessments.
6. Returns `ReplayTrace`, including hashes, ranking, selected IDs, triage cost,
   cluster IDs, and metrics.

Fractional budgets use `ceil(fraction × unit_count)` per response.

## Cache v2

`EvidenceScoreCache` accepts only schema
`claimreplay.evidence-score-cache.v2`. Each entry binds model ID/revision,
method, claim/evidence SHA-256, scorer execution identity, and validated NLI
probabilities. The Transformers scorer identity includes preprocessing,
tokenizer mode, truncation, batching, runtime/library settings, implementation
hash, and optional declared model/tokenizer asset hashes.

Old v1 caches are not migrated or overwritten. Use frozen `d3dcda0` code for
historical replay.
