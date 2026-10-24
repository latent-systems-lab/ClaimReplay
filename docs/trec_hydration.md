# Hydrating the TREC RAG 2024 evaluation locally

ClaimReplay does not redistribute the official TREC assessment records, MS MARCO passages, or materialized evidence text. Obtain the assessment files from the [official TREC RAG 2024 data page](https://trec.nist.gov/data/rag2024.html) and provide their local directory explicitly.

Expected assessment identities are:

| Condition | SHA-256 | Canonical records |
|---|---|---:|
| `without_prediction` | `96a55934dfbfd5c2350d05f83faffe236976e9404931725f014b9107f4f5e373` | 381 responses / 3,406 units |
| `with_prediction` | `3224b7f2fadf3635b54f9f87320d557832234253a76600724511a7b1c3724369` | 585 responses / 5,020 units |
| `webassess` | `76fd9641fcd6699bdfc601a834c0e3c85f3a8a6a2b49b7c484eadf12bab280ba` | 549 responses / 4,737 nonempty units |

The official `webassess` file has 4,757 sentence entries. The canonical adapter deterministically omits 20 empty sentence texts.

Materialize only cited MS MARCO segments with the existing targeted extractor:

```powershell
python -m scripts.extract_trec_segments `
  --corpus-tar C:\path\to\msmarco_v2.1_doc_segmented.tar `
  --trec C:\path\to\trec2024\without_prediction.jsonl `
  --output-cache C:\private\evidence\primary_segments.jsonl `
  --report C:\private\evidence\primary_extraction.json

python -m scripts.extract_trec_segments `
  --corpus-tar C:\path\to\msmarco_v2.1_doc_segmented.tar `
  --trec C:\path\to\trec2024\with_prediction.jsonl `
  --trec C:\path\to\trec2024\webassess.jsonl `
  --output-cache C:\private\evidence\supplementary_segments.jsonl `
  --report C:\private\evidence\supplementary_extraction.json
```

Then validate hashes, counts, parsing and evidence joins without inference:

```powershell
python -m scripts.validate_trec_hydration `
  --trec-dir C:\path\to\trec2024 `
  --primary-evidence-cache C:\private\evidence\primary_segments.jsonl `
  --supplementary-evidence-cache C:\private\evidence\supplementary_segments.jsonl `
  --output hydration_validation.json
```

This local-input route has been validated against the project-held official assets. It is not advertised as automatic acquisition: users remain responsible for obtaining third-party files under the applicable source terms.
