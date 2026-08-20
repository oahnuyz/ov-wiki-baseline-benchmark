"""Validation for the baseline-neutral prepared-data contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import safe_relative_path, sha256_file


SCHEMA_VERSION = "1.0"


def validate_documents(
    root: Path, documents: list[dict[str, Any]], *, expected_count: int
) -> None:
    if len(documents) != expected_count:
        raise ValueError(f"Expected {expected_count} documents, got {len(documents)}")
    ids: list[str] = []
    paths: list[str] = []
    for index, document in enumerate(documents):
        document_id = str(document.get("id") or "").strip()
        source_id = str(document.get("source_id") or "").strip()
        relative_path = str(document.get("path") or "").strip()
        media_type = str(document.get("media_type") or "").strip()
        if (
            document.get("schema_version") != SCHEMA_VERSION
            or not document_id
            or not source_id
            or not relative_path
            or media_type not in {"application/pdf", "text/plain"}
            or not isinstance(document.get("metadata"), dict)
        ):
            raise ValueError(f"Invalid canonical document at index {index}")
        path = root / safe_relative_path(relative_path)
        if not path.is_file() or path.stat().st_size != document.get("size_bytes"):
            raise ValueError(f"Missing or invalid canonical document: {path}")
        if sha256_file(path) != document.get("sha256"):
            raise ValueError(f"Canonical document checksum mismatch: {path}")
        ids.append(document_id)
        paths.append(relative_path)
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Canonical document IDs and paths must be unique")


def validate_qas(
    qas: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    expected_count: int,
) -> None:
    if len(qas) != expected_count:
        raise ValueError(f"Expected {expected_count} QA records, got {len(qas)}")
    document_ids = {str(document["id"]) for document in documents}
    qa_ids: list[str] = []
    for index, qa in enumerate(qas):
        qa_id = str(qa.get("id") or "").strip()
        question = str(qa.get("question") or "").strip()
        gold_answers = qa.get("gold_answers")
        evidence = qa.get("evidence")
        referenced_ids = qa.get("document_ids")
        if (
            qa.get("schema_version") != SCHEMA_VERSION
            or not qa_id
            or not question
            or not isinstance(gold_answers, list)
            or not gold_answers
            or any(not isinstance(value, str) or not value.strip() for value in gold_answers)
            or not isinstance(evidence, list)
            or any(not isinstance(value, str) or not value.strip() for value in evidence)
            or not isinstance(referenced_ids, list)
            or any(value not in document_ids for value in referenced_ids)
            or not isinstance(qa.get("metadata"), dict)
            or "original_record" not in qa["metadata"]
        ):
            raise ValueError(f"Invalid canonical QA at index {index}")
        qa_ids.append(qa_id)
    if len(qa_ids) != len(set(qa_ids)):
        raise ValueError("Canonical QA IDs must be unique")
