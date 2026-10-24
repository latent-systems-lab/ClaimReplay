# ClaimReplay

Compare claim-prioritization policies on the same frozen responses under explicit
inspection budgets. ClaimReplay provides dataset adapters, policy interfaces,
deterministic replay, metrics and machine-readable traces.

Python **3.10–3.12**. No core runtime dependencies. **Apache-2.0** for original
software; datasets and models have separate terms.

## Install and run offline

```sh
python -m venv .venv
# POSIX: .venv/bin/python; Windows: .venv\Scripts\python
python -m pip install --no-index --no-deps dist/claimreplay-0.6.0rc2-py3-none-any.whl
python -m claimreplay.cli examples/fixtures/toy_response.json --policy feature --feature priority --task unsupported --budget 2
python -m claimreplay.reproduce --config configs/toy_reproduction.json --data-dir . --output toy-results
```

Use the environment's Python for all commands. The CLI selects `u03` and `u10`
and finds both unsupported units. The reproduction example exports 18 synthetic
traces. No dataset download, API key or model inference is required.

For offline installation from source, use the included build tools:

```sh
python -m pip install --no-index --no-deps --find-links build_wheels dist/claimreplay-0.6.0rc2.tar.gz
```

## Implement a policy

The [external-policy tutorial](docs/external_policy_tutorial.md) implements
ClaimLength without changing the package:

```sh
python examples/run_external_policy_tutorial.py examples/fixtures/toy_response.json first
python examples/run_external_policy_tutorial.py examples/fixtures/toy_response.json repeat --compare-to first
```

It compares ClaimLength, Random and Lexical at 5%, 10% and 20% quotas and exports
rankings, selections, recall and coverage. Synthetic fixtures demonstrate software
behavior, not real-world ranking effectiveness.

## Use benchmark data

Acquire inputs from their official providers; see [DATA_ACCESS.md](DATA_ACCESS.md).
Configure a private data root and run:

```sh
python -m claimreplay.reproduce --config configs/benchmark_evaluation.json --data-dir DATA --output results
```

The adapters preserve native assessment meanings. TREC assessment conditions are
evaluated separately. Unassessed units consume slots. Per-response ceiling budgets
can exceed the requested fraction, and equal inspection quotas do not imply equal
scoring costs. Gold labels are evaluator-only; custom adapters must not encode
them as policy features. Oracle is privileged and diagnostic only.

## Documentation and testing

- [API reference](docs/api_reference.md)
- [Data acquisition](DATA_ACCESS.md), [controlled inputs](docs/controlled_data_acquisition.md) and [TREC evidence](docs/trec_hydration.md)
- [Reproducibility](REPRODUCIBILITY.md) and [optional evidence-aware policies](docs/evidence_aware_policy.md)
- [Export result tables and budget curves](docs/result_reporting.md)
- [Contributing](CONTRIBUTING.md), [maintenance](MAINTENANCE.md), [citation](CITATION.cff)

After installation, run `python release_tests/verify_release.py`.
For development, install `.[test]` and run `python -m pytest tests release_tests`.
`RELEASE_MANIFEST.json` and `SHA256SUMS.txt` identify distribution contents.

Authors: Feliks Bańka and Jarosław A. Chudziak, Warsaw University of Technology.
Repository: https://github.com/latent-systems-lab/ClaimReplay
