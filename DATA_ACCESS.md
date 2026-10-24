# Component-specific rights and authorized acquisition

This is a provenance checklist, not legal advice. Apache-2.0 applies to original ClaimReplay software only; it grants no rights in upstream inputs.

| Component | Terms/source | Candidate treatment |
|---|---|---|
| Project-owned code/toy fixtures | Apache-2.0; author confirmed rights-holder permission | Included |
| TREC RAG 2024 citation/support assets | https://trec.nist.gov/data/rag2024.html | Official acquisition only; online access is not blanket redistribution permission |
| MS MARCO | https://microsoft.github.io/msmarco/Notice.html | Non-commercial research, no implied IP license; all native text/cache excluded |
| FEVEROUS | https://huggingface.co/datasets/fever/feverous ; https://fever.ai/dataset/feverous.html | Dataset card CC BY-SA 3.0; underlying Wikipedia obligations; native data excluded |
| SciFact claims/annotations | https://github.com/allenai/scifact/blob/master/LICENSE.md | CC BY 4.0; excluded |
| SciFact corpus abstracts | Same source | ODC-By 1.0; excluded; code upstream Apache-2.0 is a separate component |
| Evidence NLI | https://huggingface.co/cross-encoder/nli-deberta-v3-base | Historical model revision 0233b6e1207aa367aa01ac84abd8efecc842f2d1; model-card terms need author confirmation; weights absent |
| AFaCTA | https://github.com/EdisonNi-hku/AFaCTA ; https://huggingface.co/JingweiNi/roberta-base-afacta | Pinned model card records MIT; pinned author source had no root license. Treat source and weights separately; neither is bundled |
| OpenAI outputs | https://openai.com/policies/services-agreement/ | Customer output rights do not override third-party input rights; all private responses excluded |
| Build tools | build_wheels/*.whl | Embedded licenses/notices retained; not project-owned code |

## TREC acquisition steps (performed by an authorized researcher, not automatically)

1. Review official TREC and MS MARCO conditions and confirm eligibility.
2. Obtain the three official files linked by the TREC page:
   `final.citation_judgments_without_prediction.20241025.jsonl`,
   `final.citation_judgments_with_prediction.20241025.jsonl`,
   `final.citation_judgments_Webassess.20241031.jsonl`.
3. Copy/rename locally as without_prediction.jsonl, with_prediction.jsonl,
   webassess.jsonl. Never copy them into this public distribution.
4. Match the pinned SHA-256 and counts in docs/trec_hydration.md. A mismatch
   is a different data version, not permission to change the frozen cohort.
5. For evidence-aware use only, separately obtain the official segmented MS MARCO
   corpus under its own conditions and confirm MD5 3799e7611efffd8daeb257e9ccca4d60.
   Run the shipped targeted extractor; no complete corpus is required for core replay.
6. Run the exact extractor and local validator commands in docs/trec_hydration.md.
   Outputs remain in user-controlled private storage; never publish extracted text.

Source inspection date: 2026-10-23. The project did not acquire or redistribute
any additional corpus for this release task. Contact/permissions remain the
researcher's and rights holders' responsibility.

## Controlled inputs and exact byte identities

For FEVEROUS, obtain the training claim file through the official FEVEROUS
dataset page above; place it at `feverous/train.jsonl` in your private data root.
The historical file has 71,291 claims and the SHA-256 recorded in
`configs/benchmark_evaluation.json`. A current download of identical bytes has NOT been
verified; if it differs, preserve both versions and report the incompatibility.
This is a concrete historical-acquisition limitation, not permission to bypass
the hash or reconstruct labels from results.

For SciFact, follow the Data instructions in the official AllenAI/scifact
repository, obtain its data release, and extract `claims_dev.jsonl` (300 claims)
and `corpus.jsonl` into `scifact/` under your private data root. Verify both hashes
against the config before preparation. Do not mix train/test claims into dev.
Claims/annotations and abstracts have distinct terms as listed above.

Automatic acquisition is deliberately not asserted as permitted for these assets.
The manual official-provider route is required until access/rights are reviewed.
No restricted native record, model output, citation ID list or Oracle score is
included in this candidate. The dataset-level hashes identify required files;
they are not an assertion of redistribution rights over derived outputs.
