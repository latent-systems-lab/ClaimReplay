# Controlled FEVEROUS and SciFact inputs

Use a private, configurable data root, not a legacy project directory:

```text
DATA/
  feverous/train.jsonl
  scifact/claims_dev.jsonl
  scifact/corpus.jsonl
```

Obtain FEVEROUS training claims through the official provider linked in
[DATA_ACCESS.md](../DATA_ACCESS.md). Obtain SciFact development claims and the
abstract corpus from the official AllenAI release. Their terms differ; review
those sources before use. Do not mix train, development and test files.

Verify the exact hashes and record counts in `configs/benchmark_evaluation.json`. Run:

```sh
python -m claimreplay.reproduce --config configs/benchmark_evaluation.json --data-dir DATA --output native-results
```

That full config also requires the three separately acquired TREC assets.
For only controlled inputs, use a private config with the `trec` list omitted;
retain the exact controlled dataset hashes, split identities and target mappings.
If a current provider file differs, stop and investigate instead of bypassing
the check. Native files and generated evaluator traces stay outside Git.

Keep native assets and generated evaluator outputs in your own storage, outside
Git. Record their hashes and the evaluation configuration for repeatable replay.
