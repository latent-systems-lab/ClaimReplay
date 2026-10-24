"""Generate the redistributable TREC-RAG-shaped contract-test fixture.

The records exercise the public adapter schema and all support mappings.  They
are authored synthetic data: they are not TREC topics, submitted responses,
judgments, citations, or MS MARCO identifiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT = Path("examples/fixtures/trec_rag_2024_synthetic.jsonl")


def records() -> list[dict]:
    output = []
    for index in range(10):
        topic_id = f"synthetic-topic-{index // 2:02d}"
        run_id = f"synthetic-run-{index % 2:02d}"
        output.append({
            "topic_id": topic_id,
            "run_id": run_id,
            "synthetic_fixture": True,
            "sentences": [
                {
                    "sentenceID": 0,
                    "text": f"Synthetic response {index} contains a fully supported test sentence.",
                    "citations": [{
                        "citationID": 0,
                        "reference": f"synthetic-evidence-{index}-full",
                        "support": "2",
                    }],
                },
                {
                    "sentenceID": 1,
                    "text": f"Synthetic response {index} contains a partially supported test sentence.",
                    "citations": [{
                        "citationID": 0,
                        "reference": f"synthetic-evidence-{index}-partial",
                        "support": "1",
                    }],
                },
                {
                    "sentenceID": 2,
                    "text": f"Synthetic response {index} contains an unsupported test sentence.",
                    "citations": [{
                        "citationID": 0,
                        "reference": f"synthetic-evidence-{index}-none",
                        "support": "0",
                    }],
                },
                {
                    "sentenceID": 3,
                    "text": f"Synthetic response {index} contains an unassessed uncited sentence.",
                    "citations": [],
                },
            ],
        })
    return output


def render_jsonl() -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_jsonl(), encoding="utf-8")
    print(f"wrote {len(records())} synthetic records to {args.output}")


if __name__ == "__main__":
    main()
