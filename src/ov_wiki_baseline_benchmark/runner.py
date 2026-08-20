"""Dispatch dataset downloads and canonical preparation by experiment spec."""

from __future__ import annotations

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
from .io import load_jsonl
from .normalize import normalize_experiment
from .schema import validate_documents, validate_qas
from .specs import ExperimentSpec


def _download_paperscope(
    spec: ExperimentSpec, raw_root: Path, force: bool
) -> bool:
    return paperscope_summary.download_paperscope_summary(
        output_dir=raw_root,
        dataset_name=spec.raw_dataset,
        document_scope=str(spec.options["document_scope"]),
        force=force,
        verify=True,
    )


def _download_mdaqa(spec: ExperimentSpec, raw_root: Path, force: bool) -> bool:
    return mdaqa.download_mdaqa_first_100(
        output_dir=raw_root,
        dataset_name=spec.raw_dataset,
        force=force,
        verify=True,
    )


def _download_wildgraphbench(
    spec: ExperimentSpec, raw_root: Path, force: bool
) -> bool:
    return wildgraphbench.download_wildgraphbench_summary(
        output_dir=raw_root,
        dataset_name=spec.raw_dataset,
        scope=str(spec.options["scope"]),
        force=force,
        verify=True,
    )


def _download_scholarqa(
    spec: ExperimentSpec, raw_root: Path, force: bool
) -> bool:
    return scholarqa_multi.download_scholarqa_multi_valid_101(
        output_dir=raw_root,
        dataset_name=spec.raw_dataset,
        force=force,
        verify=True,
    )


def _download_mudabench(
    spec: ExperimentSpec, raw_root: Path, force: bool
) -> bool:
    return mudabench.download_mudabench(
        output_dir=raw_root,
        dataset_name=spec.raw_dataset,
        scope=str(spec.options["scope"]),
        force=force,
        verify=True,
    )


def _download_enterprise(
    spec: ExperimentSpec, raw_root: Path, force: bool
) -> bool:
    return enterprise_rag_bench.download_enterprise_rag_bench_selected_80(
        output_dir=raw_root,
        dataset_name=spec.raw_dataset,
        force=force,
        verify=True,
    )


DOWNLOADERS: dict[str, Callable[[ExperimentSpec, Path, bool], bool]] = {
    "paperscope_summary": _download_paperscope,
    "mdaqa": _download_mdaqa,
    "wildgraphbench_summary": _download_wildgraphbench,
    "scholarqa_multi": _download_scholarqa,
    "mudabench": _download_mudabench,
    "enterprise_rag_bench": _download_enterprise,
}


def download_experiment(
    spec: ExperimentSpec, data_dir: Path, *, force: bool = False
) -> Path:
    raw_root = data_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    try:
        downloader = DOWNLOADERS[spec.dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset downloader: {spec.dataset}") from exc
    if not downloader(spec, raw_root, force):
        raise RuntimeError(f"Download verification failed for {spec.id}")
    return raw_root / spec.raw_dataset


def prepare_experiment(
    spec: ExperimentSpec,
    data_dir: Path,
    *,
    force_download: bool = False,
    skip_download: bool = False,
) -> dict[str, Any]:
    raw_dir = data_dir / "raw" / spec.raw_dataset
    if not skip_download:
        raw_dir = download_experiment(spec, data_dir, force=force_download)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw dataset is unavailable for --skip-download: {raw_dir}"
        )
    output_dir = data_dir / "prepared" / spec.id
    return normalize_experiment(spec, raw_dir, output_dir)


def verify_prepared(spec: ExperimentSpec, data_dir: Path) -> None:
    root = data_dir / "prepared" / spec.id
    documents = load_jsonl(root / "documents.jsonl")
    qas = load_jsonl(root / "qa.jsonl")
    validate_documents(root, documents, expected_count=spec.expected_documents)
    validate_qas(qas, documents, expected_count=spec.expected_qas)
