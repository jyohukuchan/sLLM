#!/usr/bin/env python3
"""Build a deterministic MTP draft vocabulary from pinned corpus inputs.

The corpus itself is deliberately not part of the repository.  A manifest is
the reproducible boundary: every source has a dataset/revision/file identity
and an observed SHA-256, while the tokenizer identity is checked before any
text is consumed.  Sources are streamed and each source has an explicit byte
and row budget.

Manifest schema (``qwen38-draft-vocab-v1``)::

    {
      "schema_version": "qwen38-draft-vocab-v1",
      "tokenizer": {"path": ".../tokenizer.json", "sha256": "..."},
      "vocab_size": 98304,
      "required_special_tokens": ["<|endoftext|>", 151643],
      "overlap_exclusions": [
        {"dataset": "mtp-bench-v1", "revision": "...",
         "file": "prompts.jsonl", "sha256": "...", "reason": "..."}
      ],
      "domains": [{
        "name": "en_web", "dataset": "...", "revision": "...",
        "files": [{
          "file": "train-00000.parquet", "path": "/cache/train-00000.parquet",
          "sha256": "...", "format": "parquet", "columns": ["text"],
          "filters": [{"column": "language", "op": "eq", "value": "en"}],
          "max_bytes": 60000000, "max_rows": 100000
        }]
      }]
    }

``columns`` is mandatory for parquet, so selecting all fields by accident is
impossible.  Text files use one UTF-8 line per row.  The optional
``exclude_text_sha256`` list hashes the selected UTF-8 row and can remove
known overlap rows without storing their contents.

The output is a raw little-endian uint32 sequence in vocabulary-ID order.
The accompanying metadata is canonical JSON and contains the hashes needed by
the model-side artifact consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = "qwen38-draft-vocab-v1"
SHA256_RE = set("0123456789abcdef")


class ManifestError(ValueError):
    """Raised when a manifest or pinned input violates the contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestError(f"cannot read pinned file {path}: {exc}") from exc
    return digest.hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in SHA256_RE for char in value.lower()):
        raise ManifestError(f"{label} must be a lowercase SHA-256 hex digest")
    if value != value.lower():
        raise ManifestError(f"{label} must use lowercase hexadecimal")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path)


def _identity(item: Mapping[str, Any], label: str) -> tuple[str, str, str, str]:
    return (
        _text(item.get("dataset"), f"{label}.dataset"),
        _text(item.get("revision"), f"{label}.revision"),
        _text(item.get("file"), f"{label}.file"),
        _sha(item.get("sha256"), f"{label}.sha256"),
    )


def _validate_filter(spec: Any, label: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ManifestError(f"{label} must be an object")
    column = _text(spec.get("column"), f"{label}.column")
    operator = spec.get("op")
    if operator not in {"eq", "ne", "in", "not_in", "exists"}:
        raise ManifestError(f"{label}.op must be eq, ne, in, not_in, or exists")
    if operator in {"eq", "ne"} and "value" not in spec:
        raise ManifestError(f"{label}.value is required for {operator}")
    if operator in {"in", "not_in"} and (not isinstance(spec.get("value"), list) or not spec["value"]):
        raise ManifestError(f"{label}.value must be a non-empty list for {operator}")
    if operator == "exists" and "value" in spec and not isinstance(spec["value"], bool):
        raise ManifestError(f"{label}.value must be boolean for exists")
    return dict(spec)


def _validate_source(source: Any, domain_name: str, index: int, manifest_path: Path) -> dict[str, Any]:
    label = f"domains[{domain_name!r}].files[{index}]"
    if not isinstance(source, dict):
        raise ManifestError(f"{label} must be an object")
    identity = _identity(source, label)
    local_path = source.get("path", source.get("local_path", source.get("file")))
    local_path = _resolve_path(manifest_path, _text(local_path, f"{label}.path"))
    source_format = source.get("format")
    if source_format is None:
        source_format = "parquet" if local_path.suffix.lower() == ".parquet" else "text"
    if source_format not in {"text", "parquet"}:
        raise ManifestError(f"{label}.format must be text or parquet")
    max_bytes = _positive_int(source.get("max_bytes"), f"{label}.max_bytes")
    max_rows = _positive_int(source.get("max_rows"), f"{label}.max_rows")
    columns = source.get("columns")
    if source_format == "parquet":
        if not isinstance(columns, list) or not columns or any(not isinstance(item, str) or not item for item in columns):
            raise ManifestError(f"{label}.columns must explicitly select one or more parquet columns")
        if len(set(columns)) != len(columns):
            raise ManifestError(f"{label}.columns contains duplicates")
    elif columns is not None:
        raise ManifestError(f"{label}.columns is only valid for parquet")
    filters = source.get("filters", [])
    if not isinstance(filters, list):
        raise ManifestError(f"{label}.filters must be a list")
    if source_format == "text" and filters:
        raise ManifestError(f"{label}.filters requires a parquet source")
    validated_filters = [_validate_filter(spec, f"{label}.filters[{i}]") for i, spec in enumerate(filters)]
    excluded = source.get("exclude_text_sha256", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise ManifestError(f"{label}.exclude_text_sha256 must be a list of SHA-256 digests")
    excluded = [_sha(item, f"{label}.exclude_text_sha256[{i}]") for i, item in enumerate(excluded)]
    if len(set(excluded)) != len(excluded):
        raise ManifestError(f"{label}.exclude_text_sha256 contains duplicates")
    return {
        "dataset": identity[0],
        "revision": identity[1],
        "file": identity[2],
        "sha256": identity[3],
        "path": local_path,
        "format": source_format,
        "columns": list(columns) if columns is not None else None,
        "filters": validated_filters,
        "exclude_text_sha256": excluded,
        "max_bytes": max_bytes,
        "max_rows": max_rows,
    }


def load_manifest(path: Path | str) -> tuple[dict[str, Any], str]:
    """Load and strictly validate a generator manifest.

    The returned dictionary is normalized for the generator.  The second
    return value is the SHA-256 of its canonical JSON representation.
    """

    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot parse manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION!r}")

    tokenizer = raw.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ManifestError("tokenizer must be an object")
    tokenizer_path = _resolve_path(manifest_path, _text(tokenizer.get("path"), "tokenizer.path"))
    tokenizer_sha = _sha(tokenizer.get("sha256"), "tokenizer.sha256")
    requested_n = _positive_int(raw.get("vocab_size"), "vocab_size")

    special = raw.get("required_special_tokens")
    if not isinstance(special, list) or not special:
        raise ManifestError("required_special_tokens must be a non-empty list")
    if any((isinstance(item, bool) or not isinstance(item, (str, int))) for item in special):
        raise ManifestError("required_special_tokens entries must be token strings or integer IDs")

    exclusions = raw.get("overlap_exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise ManifestError("overlap_exclusions must be a non-empty list")
    exclusion_ids: set[tuple[str, str, str]] = set()
    for index, item in enumerate(exclusions):
        if not isinstance(item, dict):
            raise ManifestError(f"overlap_exclusions[{index}] must be an object")
        identity = _identity(item, f"overlap_exclusions[{index}]")
        _text(item.get("reason"), f"overlap_exclusions[{index}].reason")
        logical_identity = identity[:3]
        if logical_identity in exclusion_ids:
            raise ManifestError(f"duplicate overlap exclusion {identity[:3]}")
        exclusion_ids.add(logical_identity)

    domains = raw.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ManifestError("domains must be a non-empty list")
    normalized_domains: list[dict[str, Any]] = []
    names: set[str] = set()
    source_ids: set[tuple[str, str, str]] = set()
    for domain_index, domain in enumerate(domains):
        label = f"domains[{domain_index}]"
        if not isinstance(domain, dict):
            raise ManifestError(f"{label} must be an object")
        name = _text(domain.get("name"), f"{label}.name")
        if name in names:
            raise ManifestError(f"duplicate domain name {name!r}")
        names.add(name)
        files = domain.get("files")
        if not isinstance(files, list) or not files:
            raise ManifestError(f"{label}.files must be a non-empty list")
        normalized_files = []
        for file_index, source in enumerate(files):
            normalized = _validate_source(source, name, file_index, manifest_path)
            identity = tuple(normalized[key] for key in ("dataset", "revision", "file", "sha256"))
            logical_identity = identity[:3]
            if logical_identity in source_ids:
                raise ManifestError(f"input source appears more than once: {identity[:3]}")
            if logical_identity in exclusion_ids:
                raise ManifestError(f"input source is listed as an overlap exclusion: {identity[:3]}")
            source_ids.add(logical_identity)
            normalized_files.append(normalized)
        normalized_domains.append({"name": name, "files": normalized_files})

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": {"path": tokenizer_path, "sha256": tokenizer_sha},
        "vocab_size": requested_n,
        "required_special_tokens": list(special),
        "overlap_exclusions": exclusions,
        "domains": normalized_domains,
    }
    return normalized, sha256_bytes(canonical_json_bytes(raw))


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ManifestError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ManifestError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _row_matches(row: Mapping[str, Any], filters: Sequence[Mapping[str, Any]]) -> bool:
    for spec in filters:
        column = spec["column"]
        operator = spec["op"]
        present = column in row and row[column] is not None
        value = row.get(column)
        if operator == "exists":
            expected = spec.get("value", True)
            if present != expected:
                return False
        elif operator == "eq" and value != spec["value"]:
            return False
        elif operator == "ne" and value == spec["value"]:
            return False
        elif operator == "in" and value not in spec["value"]:
            return False
        elif operator == "not_in" and value in spec["value"]:
            return False
    return True


def _selected_text(row: Mapping[str, Any], columns: Sequence[str]) -> str | None:
    parts: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ManifestError(f"selected source column {column!r} must contain strings")
        parts.append(value)
    if not parts:
        return None
    return "\n".join(parts)


def _iter_text_rows(source: Mapping[str, Any]) -> Iterator[str]:
    path = source["path"]
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                try:
                    yield raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ManifestError(f"{path} is not valid UTF-8") from exc
    except OSError as exc:
        raise ManifestError(f"cannot stream text source {path}: {exc}") from exc


def _iter_parquet_rows(source: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ManifestError("pyarrow is required for parquet corpus inputs") from exc
    path = source["path"]
    columns = list(source["columns"])
    filter_columns = [spec["column"] for spec in source["filters"]]
    read_columns = list(dict.fromkeys(columns + filter_columns))
    try:
        parquet_file = parquet.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = [column for column in read_columns if column not in available]
        if missing:
            raise ManifestError(f"{path} has no requested parquet columns: {missing}")
        yielded_rows = 0
        for batch in parquet_file.iter_batches(batch_size=min(source["max_rows"], 4096), columns=read_columns):
            for row in batch.to_pylist():
                if yielded_rows >= source["max_rows"]:
                    return
                yielded_rows += 1
                yield row
    except ManifestError:
        raise
    except Exception as exc:  # pyarrow exceptions are not stable across versions.
        raise ManifestError(f"cannot stream parquet source {path}: {exc}") from exc


def _iter_source_text(source: Mapping[str, Any]) -> tuple[Iterator[str], str]:
    if source["format"] == "text":
        return _iter_text_rows(source), "text"
    rows = _iter_parquet_rows(source)

    def selected() -> Iterator[str]:
        for row in rows:
            if _row_matches(row, source["filters"]):
                text = _selected_text(row, source["columns"])
                if text is not None:
                    yield text

    return selected(), "parquet"


def _count_source(source: Mapping[str, Any], tokenizer: Any) -> tuple[dict[int, int], dict[str, int]]:
    counts: dict[int, int] = {}
    rows = 0
    byte_count = 0
    scanned_rows = 0
    scanned_bytes = 0
    token_count = 0
    excluded_count = 0
    stream, _ = _iter_source_text(source)
    for text in stream:
        encoded = text.encode("utf-8")
        if scanned_bytes + len(encoded) > source["max_bytes"] or scanned_rows >= source["max_rows"]:
            break
        scanned_rows += 1
        scanned_bytes += len(encoded)
        row_sha = sha256_bytes(encoded)
        if row_sha in source["exclude_text_sha256"]:
            excluded_count += 1
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        rows += 1
        byte_count += len(encoded)
        token_count += len(ids)
        for token_id in ids:
            token_id = int(token_id)
            counts[token_id] = counts.get(token_id, 0) + 1
    if rows == 0 or token_count == 0:
        raise ManifestError(f"source produced no tokens within its bounds: {source['path']}")
    return counts, {
        "rows": rows,
        "bytes": byte_count,
        "scanned_rows": scanned_rows,
        "scanned_bytes": scanned_bytes,
        "tokens": token_count,
        "unique_tokens": len(counts),
        "excluded_rows": excluded_count,
    }


def score_domains(domain_counts: Sequence[Mapping[int, int]]) -> dict[int, float]:
    """Normalize every domain independently, then give every domain equal weight."""

    if not domain_counts:
        raise ManifestError("at least one domain is required")
    scores: dict[int, float] = {}
    weight = 1.0 / len(domain_counts)
    for index, counts in enumerate(domain_counts):
        total = sum(counts.values())
        if total <= 0:
            raise ManifestError(f"domain {index} has no token frequency")
        for token_id, count in counts.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ManifestError(f"invalid token ID {token_id!r}")
            scores[token_id] = scores.get(token_id, 0.0) + weight * (count / total)
    return scores


class _TokenizerAdapter:
    def __init__(self, path: Path, expected_sha: str) -> None:
        _verify_file(path, expected_sha, "tokenizer")
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ManifestError("the tokenizers package is required to read tokenizer.json") from exc
        try:
            self._tokenizer = Tokenizer.from_file(str(path))
        except Exception as exc:
            raise ManifestError(f"cannot load tokenizer {path}: {exc}") from exc
        self.vocab_size = int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Any:
        try:
            encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except Exception as exc:
            raise ManifestError(f"tokenizer failed to encode a corpus row: {exc}") from exc
        for token_id in encoded.ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 or token_id >= self.vocab_size:
                raise ManifestError(f"tokenizer returned out-of-range token ID {token_id}")
        return encoded

    def resolve_special(self, item: str | int) -> int:
        if isinstance(item, int) and not isinstance(item, bool):
            token_id = item
        else:
            token_id = self._tokenizer.token_to_id(item)
            if token_id is None:
                raise ManifestError(f"required special token is absent from tokenizer: {item!r}")
        if token_id < 0 or token_id >= self.vocab_size:
            raise ManifestError(f"required special token ID is out of range: {token_id}")
        return int(token_id)


def _write_output(path: Path, ids: Sequence[int]) -> str:
    output = bytearray()
    for token_id in ids:
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise ManifestError(f"token ID cannot be encoded as u32: {token_id}")
        output.extend(struct.pack("<I", token_id))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(output))
    except OSError as exc:
        raise ManifestError(f"cannot write vocabulary output {path}: {exc}") from exc
    return sha256_bytes(bytes(output))


def generate_vocabulary(manifest_path: Path | str, output_path: Path | str, metadata_path: Path | str | None = None) -> dict[str, Any]:
    """Generate IDs and metadata from a pinned manifest."""

    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_sha = load_manifest(manifest_file)
    tokenizer = _TokenizerAdapter(manifest["tokenizer"]["path"], manifest["tokenizer"]["sha256"])
    requested_n = manifest["vocab_size"]
    if requested_n > tokenizer.vocab_size:
        raise ManifestError(f"vocab_size {requested_n} exceeds tokenizer vocabulary {tokenizer.vocab_size}")
    special_ids = sorted({tokenizer.resolve_special(item) for item in manifest["required_special_tokens"]})

    domain_counts: list[dict[int, int]] = []
    domain_stats: list[dict[str, Any]] = []
    for domain in manifest["domains"]:
        counts: dict[int, int] = {}
        files: list[dict[str, Any]] = []
        for source in domain["files"]:
            _verify_file(source["path"], source["sha256"], f"source {source['file']}")
            source_counts, stats = _count_source(source, tokenizer)
            for token_id, count in source_counts.items():
                counts[token_id] = counts.get(token_id, 0) + count
            files.append({
                "dataset": source["dataset"], "revision": source["revision"],
                "file": source["file"], "sha256": source["sha256"],
                "rows": stats["rows"], "bytes": stats["bytes"],
                "scanned_rows": stats["scanned_rows"], "scanned_bytes": stats["scanned_bytes"],
                "tokens": stats["tokens"], "unique_tokens": stats["unique_tokens"],
                "excluded_rows": stats["excluded_rows"],
            })
        if not counts:
            raise ManifestError(f"domain produced no tokens: {domain['name']}")
        domain_counts.append(counts)
        domain_stats.append({"name": domain["name"], "files": files,
                             "tokens": sum(counts.values()), "unique_tokens": len(counts)})

    scores = score_domains(domain_counts)
    if len(special_ids) > requested_n:
        raise ManifestError("required special tokens exceed requested vocabulary rows")
    ranked = sorted(range(tokenizer.vocab_size), key=lambda token_id: (-scores.get(token_id, 0.0), token_id))
    selected = set(special_ids)
    for token_id in ranked:
        if len(selected) >= requested_n:
            break
        selected.add(token_id)
    if len(selected) != requested_n:
        raise ManifestError("could not fill requested vocabulary rows")
    ids = sorted(selected)
    output_file = Path(output_path).resolve()
    vocab_sha = _write_output(output_file, ids)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "N": requested_n,
        "vocab_count": len(ids),
        "vocab_sha256": vocab_sha,
        "manifest_sha256": manifest_sha,
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "special_token_ids": special_ids,
        "domains": domain_stats,
    }
    if metadata_path is None:
        metadata_file = output_file.with_name(output_file.name + ".metadata.json")
    else:
        metadata_file = Path(metadata_path).resolve()
    try:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_bytes(canonical_json_bytes(metadata) + b"\n")
    except OSError as exc:
        raise ManifestError(f"cannot write vocabulary metadata {metadata_file}: {exc}") from exc
    return metadata


# A short alias is convenient for callers embedding the tool in a host check.
build_vocabulary = generate_vocabulary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        metadata = generate_vocabulary(args.manifest, args.output, args.metadata)
    except ManifestError as exc:
        print(f"qwen38_draft_vocab: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
