"""Download and validate the official ten-conversation LoCoMo snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests


LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    f"{LOCOMO_REVISION}/data/locomo10.json"
)
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_SIZE_BYTES = 2_805_274
EXPECTED_SAMPLES = 10
EXPECTED_SESSIONS = 272
EXPECTED_TURNS = 5_882
EXPECTED_QAS = 1_986
EXPECTED_SAMPLE_IDS = {
    "conv-26",
    "conv-30",
    "conv-41",
    "conv-42",
    "conv-43",
    "conv-44",
    "conv-47",
    "conv-48",
    "conv-49",
    "conv-50",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locomo(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"LoCoMo must contain a JSON list of objects: {path}")
    return value


def session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        suffix = key.removeprefix("session_")
        if suffix.isdigit() and isinstance(value, list):
            numbers.append(int(suffix))
    return sorted(numbers)


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"Unexpected LoCoMo sample count: {len(records)}")
    if {str(row.get("sample_id")) for row in records} != EXPECTED_SAMPLE_IDS:
        raise ValueError("Unexpected LoCoMo sample IDs")

    session_count = 0
    turn_count = 0
    qa_count = 0
    for row in records:
        sample_id = str(row.get("sample_id") or "")
        conversation = row.get("conversation")
        qas = row.get("qa")
        if not isinstance(conversation, dict) or not isinstance(qas, list):
            raise ValueError(f"Invalid LoCoMo record: {sample_id}")
        if not str(conversation.get("speaker_a") or "").strip() or not str(
            conversation.get("speaker_b") or ""
        ).strip():
            raise ValueError(f"Missing LoCoMo speakers: {sample_id}")
        numbers = session_numbers(conversation)
        if not numbers:
            raise ValueError(f"LoCoMo conversation has no sessions: {sample_id}")
        session_count += len(numbers)
        for number in numbers:
            turns = conversation[f"session_{number}"]
            if not all(
                isinstance(turn, dict)
                and str(turn.get("speaker") or "").strip()
                and str(turn.get("dia_id") or "").strip()
                and str(turn.get("text") or "").strip()
                for turn in turns
            ):
                raise ValueError(f"Invalid turns in {sample_id} session {number}")
            turn_count += len(turns)
        for qa in qas:
            if not isinstance(qa, dict) or not str(qa.get("question") or "").strip():
                raise ValueError(f"Invalid LoCoMo QA in {sample_id}")
            category = qa.get("category")
            if category not in {1, 2, 3, 4, 5}:
                raise ValueError(f"Invalid LoCoMo category in {sample_id}: {category}")
            if category != 5 and qa.get("answer") is None:
                raise ValueError(f"Missing LoCoMo answer in {sample_id}")
            if not isinstance(qa.get("evidence"), list):
                raise ValueError(f"Invalid LoCoMo evidence in {sample_id}")
        qa_count += len(qas)

    actual = (session_count, turn_count, qa_count)
    expected = (EXPECTED_SESSIONS, EXPECTED_TURNS, EXPECTED_QAS)
    if actual != expected:
        raise ValueError(f"Unexpected LoCoMo totals: {actual} != {expected}")


def verify_locomo_download(output_dir: Path) -> bool:
    path = output_dir / "locomo10.json"
    if (
        not path.is_file()
        or path.stat().st_size != LOCOMO_SIZE_BYTES
        or sha256_file(path) != LOCOMO_SHA256
    ):
        return False
    try:
        validate_records(load_locomo(path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def download_locomo(output_dir: Path, *, force: bool = False, verify: bool = True) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "locomo10.json"
    if force or not verify_locomo_download(output_dir):
        partial = destination.with_suffix(".json.part")
        response: requests.Response | None = None
        try:
            response = requests.get(LOCOMO_URL, stream=True, timeout=(30, 180))
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
            if partial.stat().st_size != LOCOMO_SIZE_BYTES:
                raise ValueError("LoCoMo download size mismatch")
            if sha256_file(partial) != LOCOMO_SHA256:
                raise ValueError("LoCoMo download checksum mismatch")
            validate_records(load_locomo(partial))
            partial.replace(destination)
        finally:
            if response is not None:
                response.close()
            if partial.exists():
                partial.unlink()
    return verify_locomo_download(output_dir) if verify else destination.is_file()


def render_session(sample: dict[str, Any], number: int) -> str:
    conversation = sample["conversation"]
    sample_id = str(sample["sample_id"])
    speaker_a = str(conversation["speaker_a"])
    speaker_b = str(conversation["speaker_b"])
    date_time = str(conversation.get(f"session_{number}_date_time") or "Unknown")
    lines = [
        f"# LoCoMo conversation {sample_id} — session {number}",
        "",
        f"Participants: {speaker_a} and {speaker_b}",
        f"Session date and time: {date_time}",
        "",
        "## Dialogue",
        "",
    ]
    for turn in conversation[f"session_{number}"]:
        lines.append(f"[{turn['dia_id']}] {turn['speaker']}: {turn['text']}")
        caption = str(turn.get("blip_caption") or "").strip()
        query = str(turn.get("query") or turn.get("search_query") or "").strip()
        if caption:
            lines.append(f"[Image caption] {caption}")
        if query:
            lines.append(f"[Image search query] {query}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
