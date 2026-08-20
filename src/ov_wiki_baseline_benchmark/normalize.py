"""Convert dataset-native snapshots into one baseline-neutral data contract."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .datasets import (
    enterprise_rag_bench,
    mdaqa,
    mudabench,
    paperscope_summary,
    scholarqa_multi,
    wildgraphbench,
)
from .io import (
    link_or_copy,
    load_json,
    load_jsonl,
    replace_directory,
    safe_relative_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from .schema import SCHEMA_VERSION, validate_documents, validate_qas
from .specs import ExperimentSpec


def _canonical_document(
    *,
    dataset: str,
    document_id: str,
    source_id: str,
    source_path: Path,
    output_path: str,
    media_type: str,
    original_record: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    if not document_id or not source_id:
        raise ValueError("Canonical document IDs must be non-empty")
    relative_path = safe_relative_path(output_path)
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "id": document_id,
            "dataset": dataset,
            "source_id": source_id,
            "path": relative_path.as_posix(),
            "media_type": media_type,
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "metadata": {"original_record": original_record},
        },
        source_path,
    )


def _canonical_qa(
    *,
    spec: ExperimentSpec,
    qa_id: str,
    question: str,
    gold_answers: list[str],
    evidence: list[str],
    category: str,
    document_ids: list[str],
    original_record: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": qa_id,
        "dataset": spec.dataset,
        "variant": spec.id,
        "question": question.strip(),
        "gold_answers": [answer.strip() for answer in gold_answers],
        "evidence": [item.strip() for item in evidence],
        "category": category,
        "document_ids": document_ids,
        "metadata": {
            **(metadata or {}),
            "original_record": original_record,
        },
    }


def _write_prepared(
    spec: ExperimentSpec,
    output_dir: Path,
    document_sources: list[tuple[dict[str, Any], Path]],
    qas: list[dict[str, Any]],
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build_dir = output_dir.parent / f".{output_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        documents: list[dict[str, Any]] = []
        for record, source in document_sources:
            destination = build_dir / safe_relative_path(record["path"])
            link_or_copy(source, destination)
            documents.append(record)
        validate_documents(
            build_dir, documents, expected_count=spec.expected_documents
        )
        validate_qas(qas, documents, expected_count=spec.expected_qas)
        write_jsonl(build_dir / "documents.jsonl", documents)
        write_jsonl(build_dir / "qa.jsonl", qas)
        write_json(
            build_dir / "dataset_info.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment": spec.id,
                "dataset": spec.dataset,
                "raw_dataset": spec.raw_dataset,
                "options": spec.options,
                "qa_count": len(qas),
                "document_count": len(documents),
                "source_info": source_info or {},
            },
        )
        replace_directory(build_dir, output_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)
    return {
        "experiment": spec.id,
        "qa_count": len(qas),
        "document_count": len(document_sources),
        "output_dir": str(output_dir),
    }


def _source_info(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "dataset_info.json"
    if not path.is_file():
        return {}
    value = load_json(path)
    return value if isinstance(value, dict) else {}


def normalize_paperscope(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    scope = str(spec.options["document_scope"])
    qa_type = str(spec.options["qa_type"])
    if not paperscope_summary.verify_paperscope_download(raw_dir, scope):
        raise ValueError(f"PaperScope raw dataset failed verification: {raw_dir}")
    records = paperscope_summary.load_jsonl(raw_dir / "summary_corpus_1.jsonl")
    manifest = paperscope_summary.load_jsonl(raw_dir / "documents.jsonl")
    canonical_by_source = {
        str(entry["id"]): f"paperscope:{entry['id']}" for entry in manifest
    }
    id_by_link = {
        str(entry.get("pdf_link") or ""): str(entry["id"]) for entry in manifest
    }
    documents = [
        _canonical_document(
            dataset=spec.dataset,
            document_id=canonical_by_source[str(entry["id"])],
            source_id=str(entry["id"]),
            source_path=raw_dir / "pdfs" / f"{entry['id']}.pdf",
            output_path=f"corpus/{entry['id']}.pdf",
            media_type="application/pdf",
            original_record=entry,
        )
        for entry in manifest
    ]
    qas: list[dict[str, Any]] = []
    for source_index, record in enumerate(records):
        if (
            not paperscope_summary.is_valid_summary_record(record)
            or record.get("prompt_type") != qa_type
        ):
            continue
        links = record.get("pdf_links")
        if not isinstance(links, list) or not links:
            raise ValueError(f"Invalid PaperScope links at row {source_index}")
        source_ids: list[str] = []
        for link in links:
            source_id = id_by_link.get(str(link))
            if source_id is None:
                source_id = paperscope_summary.paper_id_from_url(str(link))
            if source_id not in canonical_by_source:
                raise ValueError(
                    f"PaperScope row {source_index} references missing paper {source_id}"
                )
            source_ids.append(source_id)
        qas.append(
            _canonical_qa(
                spec=spec,
                qa_id=f"paperscope:{qa_type}:{source_index}",
                question=str(record["question"]),
                gold_answers=[str(record["answer"])],
                evidence=[],
                category=qa_type,
                document_ids=[canonical_by_source[value] for value in source_ids],
                original_record=record,
                metadata={"source_row_index": source_index},
            )
        )
    return _write_prepared(spec, output_dir, documents, qas)


def normalize_mdaqa(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if not mdaqa.verify_mdaqa_download(raw_dir):
        raise ValueError(f"MDA-QA raw dataset failed verification: {raw_dir}")
    records = mdaqa.select_first_100(
        mdaqa.load_json_records(raw_dir / "MDA-QA.json")
    )
    manifest = mdaqa.load_jsonl(raw_dir / "documents.jsonl")
    canonical_by_source = {
        str(entry["id"]): f"mdaqa:{entry['id']}" for entry in manifest
    }
    documents = [
        _canonical_document(
            dataset=spec.dataset,
            document_id=canonical_by_source[str(entry["id"])],
            source_id=str(entry["id"]),
            source_path=raw_dir / "pdfs" / f"{entry['id']}.pdf",
            output_path=f"corpus/{entry['id']}.pdf",
            media_type="application/pdf",
            original_record=entry,
        )
        for entry in manifest
    ]
    qas = [
        _canonical_qa(
            spec=spec,
            qa_id=f"mdaqa:{record['id']}",
            question=str(record["question"]),
            gold_answers=[str(record["answer"])],
            evidence=[],
            category="multi_document",
            document_ids=[canonical_by_source[str(value)] for value in record["support"]],
            original_record=record,
            metadata={"source_qa_id": record["id"]},
        )
        for record in records
    ]
    return _write_prepared(spec, output_dir, documents, qas)


def normalize_wildgraphbench(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    scope = str(spec.options["scope"])
    if not wildgraphbench.verify_wildgraphbench_download(raw_dir, scope):
        raise ValueError(f"WildGraphBench raw dataset failed verification: {raw_dir}")
    manifest = wildgraphbench.load_jsonl(raw_dir / "documents.jsonl")
    records = wildgraphbench.load_jsonl(raw_dir / "summary_questions.jsonl")
    documents: list[tuple[dict[str, Any], Path]] = []
    for entry in manifest:
        source_relative = safe_relative_path(str(entry["relative_path"]))
        if source_relative.parts[0] != "reference_pages":
            raise ValueError(f"Unexpected WildGraphBench path: {source_relative}")
        corpus_relative = Path("corpus").joinpath(*source_relative.parts[1:])
        documents.append(
            _canonical_document(
                dataset=spec.dataset,
                document_id=str(entry["id"]),
                source_id=str(entry["id"]),
                source_path=raw_dir / source_relative,
                output_path=corpus_relative.as_posix(),
                media_type="text/plain",
                original_record=entry,
            )
        )
    qas: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        statements = record.get("gold_statements")
        if not isinstance(statements, list) or not statements:
            raise ValueError(f"Invalid WildGraphBench gold statements at row {index}")
        combined_gold = "\n".join(f"- {str(value).strip()}" for value in statements)
        domain = str(record.get("domain") or "")
        source_row = record.get("source_row_index", index)
        qas.append(
            _canonical_qa(
                spec=spec,
                qa_id=f"wildgraphbench:{scope}:{domain}:{source_row}",
                question=str(record["question"]),
                gold_answers=[combined_gold],
                evidence=[],
                category="summary",
                document_ids=[],
                original_record=record,
                metadata={
                    "domain": domain,
                    "topic": record.get("topic"),
                    "ref_urls": record.get("ref_urls", []),
                    "document_mapping": "unresolved_upstream_reference_urls",
                },
            )
        )
    return _write_prepared(
        spec, output_dir, documents, qas, _source_info(raw_dir)
    )


def _scholarqa_augmented_gold(
    answer: str, contexts: list[dict[str, Any]]
) -> str:
    reference_lines: list[str] = []
    for index, context in enumerate(contexts):
        title = str(context.get("title") or "").strip()
        if not title:
            raise ValueError(f"ScholarQA context {index} has no title")
        reference_lines.append(f"[{index}] {title}")
    return (
        answer.strip()
        + "\n\nReference key for resolving citation labels only.\n"
        + "The generated answer does not need to reproduce this reference list.\n\n"
        + "\n".join(reference_lines)
    )


def normalize_scholarqa(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if not scholarqa_multi.verify_scholarqa_multi_download(raw_dir):
        raise ValueError(f"ScholarQA-Multi raw dataset failed verification: {raw_dir}")
    records = scholarqa_multi.load_json(
        raw_dir / "scholarqa_multi_valid_101.json"
    )
    manifest = scholarqa_multi.load_jsonl(raw_dir / "documents.jsonl")
    canonical_by_source = {
        str(entry["id"]): f"scholarqa:{entry['id']}" for entry in manifest
    }
    documents = [
        _canonical_document(
            dataset=spec.dataset,
            document_id=canonical_by_source[str(entry["id"])],
            source_id=str(entry["id"]),
            source_path=raw_dir / safe_relative_path(str(entry["relative_path"])),
            output_path=f"corpus/{entry['id']}.txt",
            media_type="text/plain",
            original_record=entry,
        )
        for entry in manifest
    ]
    document_ids_by_qa: dict[str, list[str]] = defaultdict(list)
    for entry in manifest:
        for qa_id in entry.get("qa_ids", []):
            document_ids_by_qa[str(qa_id)].append(
                canonical_by_source[str(entry["id"])]
            )
    qas: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        qa_id = str(record["id"])
        contexts = record.get("ctxs")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"ScholarQA-Multi {qa_id} has no contexts")
        evidence = [
            str(context["text"]).strip()
            for context in contexts
            if isinstance(context.get("text"), str) and str(context["text"]).strip()
        ]
        citation_map = {
            str(context_index): str(context.get("title") or "").strip()
            for context_index, context in enumerate(contexts)
        }
        qas.append(
            _canonical_qa(
                spec=spec,
                qa_id=f"scholarqa:{qa_id}",
                question=str(record["input"]),
                gold_answers=[
                    _scholarqa_augmented_gold(str(record["output"]), contexts)
                ],
                evidence=evidence,
                category=str(record["subject"]),
                document_ids=sorted(document_ids_by_qa[qa_id]),
                original_record=record,
                metadata={
                    "source_qa_id": qa_id,
                    "citation_map": citation_map,
                },
            )
        )
    return _write_prepared(
        spec, output_dir, documents, qas, _source_info(raw_dir)
    )


def normalize_mudabench(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    scope = str(spec.options["scope"])
    if not mudabench.verify_mudabench_download(raw_dir, scope):
        raise ValueError(f"MuDABench raw dataset failed verification: {raw_dir}")
    records = mudabench.load_json(raw_dir / f"{scope}.json")
    manifest = mudabench.load_jsonl(raw_dir / "documents.jsonl")
    canonical_by_source = {
        str(entry["id"]): f"mudabench:{entry['id']}" for entry in manifest
    }
    documents = [
        _canonical_document(
            dataset=spec.dataset,
            document_id=canonical_by_source[str(entry["id"])],
            source_id=str(entry["id"]),
            source_path=raw_dir / "pdfs" / str(entry["filename"]),
            output_path=f"corpus/{entry['filename']}",
            media_type="application/pdf",
            original_record=entry,
        )
        for entry in manifest
    ]
    qas: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        source_question_id = str(record["question_id"])
        source_metadata = record.get("metadata")
        if not isinstance(source_metadata, list):
            raise ValueError(f"Invalid MuDABench metadata at row {row_index}")
        source_document_ids = [str(value["id"]) for value in source_metadata]
        qas.append(
            _canonical_qa(
                spec=spec,
                qa_id=(
                    f"mudabench:{scope}:{row_index:03d}:{source_question_id}"
                ),
                question=str(record["question"]),
                gold_answers=[str(record["final_answer"])],
                evidence=[str(value) for value in record["source_answer"]],
                category=scope,
                document_ids=[
                    canonical_by_source[value] for value in source_document_ids
                ],
                original_record=record,
                metadata={
                    "source_row_index": row_index,
                    "source_question_id": source_question_id,
                },
            )
        )
    return _write_prepared(
        spec, output_dir, documents, qas, _source_info(raw_dir)
    )


def normalize_enterprise_rag_bench(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if not enterprise_rag_bench.verify_enterprise_rag_bench_download(raw_dir):
        raise ValueError(f"EnterpriseRAG-Bench raw dataset failed verification: {raw_dir}")
    records = enterprise_rag_bench.load_jsonl(
        raw_dir / "questions_selected_80.jsonl"
    )
    manifest = enterprise_rag_bench.load_jsonl(raw_dir / "documents.jsonl")
    documents = [
        _canonical_document(
            dataset=spec.dataset,
            document_id=str(entry["id"]),
            source_id=str(entry["logical_doc_id"]),
            source_path=raw_dir / safe_relative_path(str(entry["relative_path"])),
            output_path=(
                Path("corpus")
                / safe_relative_path(str(entry["archive_path"]))
            ).as_posix(),
            media_type="text/plain",
            original_record=entry,
        )
        for entry in manifest
    ]
    physical_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest:
        physical_by_source[str(entry["logical_doc_id"])].append(entry)
    for values in physical_by_source.values():
        values.sort(key=lambda entry: str(entry["archive_path"]))

    qas: list[dict[str, Any]] = []
    for record in records:
        occurrence_index: dict[str, int] = defaultdict(int)
        physical_ids: list[str] = []
        for source_id_value in record["expected_doc_ids"]:
            source_id = str(source_id_value)
            candidates = physical_by_source[source_id]
            index = occurrence_index[source_id]
            if index >= len(candidates):
                raise ValueError(
                    f"Unavailable EnterpriseRAG-Bench document occurrence: {source_id}"
                )
            physical_ids.append(str(candidates[index]["id"]))
            occurrence_index[source_id] += 1
        qas.append(
            _canonical_qa(
                spec=spec,
                qa_id=f"enterprise_rag_bench:{record['question_id']}",
                question=str(record["question"]),
                gold_answers=[str(record["gold_answer"])],
                evidence=[str(value) for value in record["answer_facts"]],
                category=str(record["question_type"]),
                document_ids=physical_ids,
                original_record=record,
                metadata={
                    "source_question_id": record["question_id"],
                    "source_document_ids": list(record["expected_doc_ids"]),
                },
            )
        )
    return _write_prepared(
        spec, output_dir, documents, qas, _source_info(raw_dir)
    )


NORMALIZERS: dict[
    str, Callable[[ExperimentSpec, Path, Path], dict[str, Any]]
] = {
    "paperscope_summary": normalize_paperscope,
    "mdaqa": normalize_mdaqa,
    "wildgraphbench_summary": normalize_wildgraphbench,
    "scholarqa_multi": normalize_scholarqa,
    "mudabench": normalize_mudabench,
    "enterprise_rag_bench": normalize_enterprise_rag_bench,
}


def normalize_experiment(
    spec: ExperimentSpec, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    try:
        normalizer = NORMALIZERS[spec.dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset normalizer: {spec.dataset}") from exc
    return normalizer(spec, raw_dir, output_dir)
