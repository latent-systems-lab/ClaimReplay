# Evidence-aware reference policy

Evidence is the NLI premise and the claim is the hypothesis.
`baseline:evidence-nli-risk-v2` uses explicit first-citation semantics and
versioned cache identity. Missing evidence must not silently select a later
citation. See the input and cache contracts below.

## Fixed method

- Model: `cross-encoder/nli-deberta-v3-base`.
- Revision: `0233b6e1207aa367aa01ac84abd8efecc842f2d1`.
- Model card: trained on SNLI and MultiNLI; labels are contradiction,
  entailment and neutral.
- Inputs: response-unit text as hypothesis and its first cited segment text as
  premise, matching TREC's first-citation support mapping.
- Risk score: `1 - entailment_probability`.
- Missing/uncited evidence: fixed risk 1.0, while the resulting low selected-
  assessment coverage remains visible in evaluation outputs.
- Aggregation: first citation only, no label-tuned selection.
- Interpretation: prioritization score only. It is not a calibrated probability
  of factual error, unsupportedness or review need.

The model is approximately 738 MB before framework dependencies.
The Transformers backend is CPU-only, evaluation-mode, seed-zero and
deterministic-algorithm configured; local-only loading remains the default.

## Cost and caching

Every trace reports units considered, logical NLI evaluations, actual model
batches, input tokens, evidence characters, cache hits and cache misses. The
v2 score-cache identity binds model ID/revision, method, claim/evidence hashes,
tokenizer and pair preparation, truncation, batching, implementation hash,
relevant runtime/library configuration, and optional declared model/tokenizer
asset hashes. It contains no response ID, support judgment, gold hash, or
dataset label. Cache hits avoid model execution but retain logical evaluation
and token-cost metadata. This prevents known incompatible reuse; it does not
promise cross-platform bitwise determinism. V1 caches are preserved and rejected
by current code rather than silently migrated. Keep older caches separate.

Evidence acquisition is a separate triage prerequisite and is not presented as
budget-free verification. Corpus download/extraction costs must be reported
alongside NLI inference costs.

