from __future__ import annotations

import re
from typing import List, Tuple


WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_document_sections(content: str, scratchpad_heading: str) -> Tuple[str, str]:
    if scratchpad_heading not in content:
        raise ValueError(f"Document must contain the heading '{scratchpad_heading}'.")
    heading_index = content.index(scratchpad_heading)
    body = content[:heading_index].rstrip()
    scratchpad = content[heading_index + len(scratchpad_heading) :].lstrip("\n")
    return body, scratchpad


def normalize_paragraphs(text: str) -> List[str]:
    stripped_text = text.strip()
    if not stripped_text:
        return []
    return [
        block.strip() for block in re.split(r"\n\s*\n", stripped_text) if block.strip()
    ]


def count_words(text: str) -> int:
    return len(text.split())


def extract_headings(content: str) -> List[str]:
    headings: List[str] = []
    for line in content.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            hashes, title = match.groups()
            headings.append(f"{hashes} {title.strip()}")
    return headings


def extract_wikilinks(content: str) -> List[str]:
    targets: List[str] = []
    for match in WIKILINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        if target:
            targets.append(target)
    return targets


def build_document(body: str, scratchpad_heading: str, scratchpad_paragraphs: List[str]) -> str:
    trimmed_body = body.rstrip()
    parts = [trimmed_body, scratchpad_heading]
    if scratchpad_paragraphs:
        scratchpad_text = "\n\n".join(scratchpad_paragraphs).rstrip()
        parts.append(scratchpad_text)
    document = "\n\n".join(part for part in parts if part)
    if not document.endswith("\n"):
        document += "\n"
    return document


def format_duration(seconds: float) -> str:
    remaining_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)
