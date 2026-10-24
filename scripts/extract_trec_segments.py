"""Extract only required cited segments from the verified official tar.

The tar remains intact. Required gzip members are streamed, and increasing
uncompressed byte offsets encoded in official segment IDs are sought directly.
No gold support labels are read into or written to the evidence cache.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile


OFFICIAL_MD5 = "3799e7611efffd8daeb257e9ccca4d60"
OFFICIAL_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar"
SEGMENT_ID = re.compile(r"^msmarco_v2\.1_doc_(?P<shard>\d{2})_\d+#\d+_(?P<offset>\d+)$")
MEMBER_SHARD = re.compile(r"_(?P<shard>\d{2})\.json\.gz$")


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def required_ids(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    identifiers: set[str] = set()
    hashes = {}
    for path in paths:
        hashes[path.as_posix()] = digest(path, "sha256")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for sentence in record.get("sentences", []):
                    for citation in sentence.get("citations") or []:
                        identifier = citation.get("reference")
                        if identifier:
                            identifiers.add(str(identifier))
    malformed = sorted(identifier for identifier in identifiers if not SEGMENT_ID.match(identifier))
    if malformed:
        raise ValueError(f"Malformed official segment IDs: {malformed[:3]}")
    return identifiers, hashes


def extract_segments(corpus_tar: Path, identifiers: set[str]) -> tuple[list[dict], list[str], dict]:
    targets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for identifier in identifiers:
        match = SEGMENT_ID.match(identifier)
        targets[match.group("shard")].append((int(match.group("offset")), identifier))
    found: dict[str, dict] = {}
    shard_stats = {}
    with tarfile.open(corpus_tar, mode="r:") as archive:
        members = {}
        for member in archive.getmembers():
            match = MEMBER_SHARD.search(member.name)
            if match:
                members[match.group("shard")] = member
        missing_shards = sorted(set(targets) - set(members))
        if missing_shards:
            return [], sorted(identifiers), {"missing_shards": missing_shards}
        for shard in sorted(targets):
            member = members[shard]
            compressed = archive.extractfile(member)
            if compressed is None:
                raise RuntimeError(f"Cannot read tar member {member.name}")
            max_position = 0
            with gzip.GzipFile(fileobj=compressed, mode="rb") as stream:
                for offset, identifier in sorted(targets[shard]):
                    stream.seek(offset)
                    if stream.tell() != offset:
                        continue
                    raw = stream.readline()
                    max_position = max(max_position, stream.tell())
                    try:
                        record = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if record.get("docid") != identifier:
                        continue
                    text = record.get("segment")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    found[identifier] = {
                        "evidence_id": identifier,
                        "text": text,
                        "title": record.get("title"),
                        "url": record.get("url"),
                        "source": {
                            "corpus_url": OFFICIAL_URL,
                            "tar_member": member.name,
                            "uncompressed_byte_offset": offset,
                            "record_sha256": hashlib.sha256(raw).hexdigest(),
                            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        },
                    }
            shard_stats[shard] = {
                "tar_member": member.name,
                "requested_count": len(targets[shard]),
                "found_count": sum(identifier in found for _, identifier in targets[shard]),
                "forward_decompressed_bytes": max_position,
            }
    return [found[key] for key in sorted(found)], sorted(set(identifiers) - set(found)), {"shards": shard_stats}


def serialize_cache(entries: list[dict]) -> bytes:
    """Canonical UTF-8/LF bytes so reported hashes are platform-independent."""
    return "".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-tar", type=Path, required=True)
    parser.add_argument("--trec", type=Path, action="append", required=True)
    parser.add_argument("--expected-md5", default=OFFICIAL_MD5)
    parser.add_argument("--output-cache", type=Path, default=Path("tmp/evidence/trec_rag_2024_segments.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/data_quality/trec_evidence_extraction_report.json"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    actual_md5 = digest(args.corpus_tar, "md5")
    if actual_md5 != args.expected_md5:
        raise ValueError(f"Corpus MD5 mismatch: expected {args.expected_md5}, found {actual_md5}")
    identifiers, trec_hashes = required_ids(args.trec)
    entries, missing, details = extract_segments(args.corpus_tar, identifiers)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    cache_bytes = serialize_cache(entries)
    args.output_cache.write_bytes(cache_bytes)
    report = {
        "schema_version": "claimreplay.trec-evidence-cache.v1",
        "corpus": {"path": args.corpus_tar.as_posix(), "bytes": args.corpus_tar.stat().st_size, "md5": actual_md5, "official_url": OFFICIAL_URL},
        "trec_input_sha256": trec_hashes,
        "required_count": len(identifiers),
        "extracted_count": len(entries),
        "missing_count": len(missing),
        "missing_ids": missing,
        "cache_path": args.output_cache.as_posix(),
        "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "contains_gold_assessments": False,
        **details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("required_count", "extracted_count", "missing_count", "cache_sha256")}, indent=2))
    if missing and not args.allow_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
