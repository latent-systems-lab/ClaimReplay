# Native adapters

The package exposes TREC RAG 2024, FEVEROUS and SciFact adapters. Preserve raw
labels and provenance in evaluator state; do not collapse support semantics into
universal factual truth. Controlled claim queues are not generated RAG answers.
Native files are not shipped. Acquisition is described in ../DATA_ACCESS.md and
local evidence validation in trec_hydration.md. Read the adapter signatures in
claimreplay/adapters/ for dataset-specific inputs. There is no public gold bundle.
