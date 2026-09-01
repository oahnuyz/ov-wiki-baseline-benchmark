"""Deterministic PDF-to-Markdown conversion used by the TencentDB baseline.

This mirrors the no-wiki OpenViking local parser: pdfplumber extracts text,
tables, and image locations.  Images are deliberately replaced by textual
placeholders because MemoryKnowledge's raw/write contract is UTF-8 text only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class PdfConversionError(RuntimeError):
    pass


def convert_pdf_to_markdown(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runner environment
        raise PdfConversionError("pdfplumber is required for PDF ingestion") from exc

    parts: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"<!-- Page {page_no} -->\n{text.strip()}")
                for table_no, table in enumerate(page.extract_tables() or [], 1):
                    rendered = _table_markdown(table)
                    if rendered:
                        parts.append(f"<!-- Page {page_no} Table {table_no} -->\n{rendered}")
                for image_no, _image in enumerate(page.images or [], 1):
                    parts.append(f"[Image omitted: page {page_no}, image {image_no}]")
        result = "\n\n".join(parts).strip()
        if not result:
            raise PdfConversionError(f"no text or tables extracted from {path.name}")
        return result + "\n"
    except PdfConversionError:
        raise
    except Exception as exc:
        raise PdfConversionError(f"failed to parse {path.name}: {exc}") from exc


def read_source_as_markdown(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return convert_pdf_to_markdown(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PdfConversionError(f"non-UTF-8 source is unsupported: {path.name}") from exc


def split_markdown_by_chapter(content: str, *, max_bytes: int = 512 * 1024) -> list[str]:
    """Pack Markdown chapters into UTF-8 chunks no larger than ``max_bytes``."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    chapters = _chapters(content)
    chunks: list[str] = []
    current = ""
    for chapter in chapters:
        if len(chapter.encode("utf-8")) <= max_bytes:
            candidate = f"{current}\n\n{chapter}".strip() if current else chapter
            if len(candidate.encode("utf-8")) <= max_bytes:
                current = candidate
                continue
            if current:
                chunks.append(_terminated(current, max_bytes))
            current = chapter
            continue
        if current:
            chunks.append(_terminated(current, max_bytes))
            current = ""
        chunks.extend(_split_oversized(chapter, max_bytes))
    if current:
        chunks.append(_terminated(current, max_bytes))
    return chunks or ["\n"]


def _chapters(content: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^(#{1,6})\s+.+$", content))
    if not matches:
        return [content.strip()]
    out: list[str] = []
    if matches[0].start() > 0 and content[: matches[0].start()].strip():
        out.append(content[: matches[0].start()].strip())
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        value = content[match.start() : end].strip()
        if value:
            out.append(value)
    return out


def _split_oversized(value: str, max_bytes: int) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", value)
    out: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
            continue
        if current:
            out.append(_terminated(current, max_bytes))
            current = ""
        encoded = paragraph.encode("utf-8")
        while len(encoded) > max_bytes:
            cut = max_bytes
            while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
                cut -= 1
            if cut <= 0:
                raise PdfConversionError("unable to split UTF-8 Markdown safely")
            out.append(encoded[:cut].decode("utf-8"))
            encoded = encoded[cut:]
        current = encoded.decode("utf-8")
    if current:
        out.append(_terminated(current, max_bytes))
    return out


def _terminated(value: str, max_bytes: int) -> str:
    return value + "\n" if len((value + "\n").encode("utf-8")) <= max_bytes else value


def _table_markdown(table: list[list[Any]]) -> str:
    if not table or not table[0]:
        return ""
    rows = [["" if cell is None else str(cell).strip().replace("|", "\\|") for cell in row] for row in table]
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)
