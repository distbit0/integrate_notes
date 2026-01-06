from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List

from spec_config import SpecConfig, default_summary_cache_path
from spec_llm import request_text
from spec_markdown import extract_wikilinks
from spec_notes import NoteRepository


@dataclass(frozen=True)
class SummaryRecord:
    content_hash: str
    summary: str


class SummaryCache:
    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._lock = Lock()
        self._data: Dict[str, SummaryRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = self._path.read_text(encoding="utf-8")
        if not raw.strip():
            return
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("Summary cache must contain a JSON object.")
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            content_hash = value.get("content_hash")
            summary = value.get("summary")
            if isinstance(content_hash, str) and isinstance(summary, str):
                self._data[key] = SummaryRecord(content_hash, summary)

    def get(self, path: Path, content_hash: str) -> str | None:
        record = self._data.get(str(path))
        if record and record.content_hash == content_hash:
            return record.summary
        return None

    def set(self, path: Path, content_hash: str, summary: str) -> None:
        with self._lock:
            self._data[str(path)] = SummaryRecord(content_hash, summary)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                key: {"content_hash": record.content_hash, "summary": record.summary}
                for key, record in self._data.items()
            }
            self._path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )

    def invalidate(self, path: Path) -> None:
        with self._lock:
            if str(path) in self._data:
                self._data.pop(str(path))
                self._path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    key: {"content_hash": record.content_hash, "summary": record.summary}
                    for key, record in self._data.items()
                }
                self._path.write_text(
                    json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
                )


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _default_summary_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(4, min(32, cpu_count * 4))


class SummaryService:
    def __init__(
        self,
        repo: NoteRepository,
        client,
        config: SpecConfig,
        cache_path: Path | None = None,
    ) -> None:
        self._repo = repo
        self._client = client
        self._config = config
        self._cache = SummaryCache(cache_path or default_summary_cache_path())
        self._executor = ThreadPoolExecutor(max_workers=_default_summary_workers())
        self._lock = Lock()
        self._inflight: Dict[Path, Future[str]] = {}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def invalidate(self, path: Path) -> None:
        with self._lock:
            self._inflight.pop(path, None)
        self._cache.invalidate(path)

    def get_summaries(self, paths: Iterable[Path]) -> Dict[Path, str]:
        unique_paths = list(dict.fromkeys(paths))
        standard_paths: List[Path] = []
        index_paths: List[Path] = []
        for path in unique_paths:
            if self._repo.is_index_note(path, self._config.index_filename_suffix):
                index_paths.append(path)
            else:
                standard_paths.append(path)

        futures = {path: self._ensure_future(path) for path in standard_paths}
        summaries: Dict[Path, str] = {
            path: future.result() for path, future in futures.items()
        }

        for path in index_paths:
            summaries[path] = self._compute_index_summary(path, stack=[])

        return summaries

    def get_summary(self, path: Path) -> str:
        if self._repo.is_index_note(path, self._config.index_filename_suffix):
            return self._compute_index_summary(path, stack=[])
        return self._ensure_future(path).result()

    def _ensure_future(self, path: Path) -> Future[str]:
        if self._repo.is_index_note(path, self._config.index_filename_suffix):
            raise RuntimeError(
                f"Index note summaries must be computed synchronously: {path.name}."
            )
        with self._lock:
            existing = self._inflight.get(path)
            if existing is not None:
                return existing
            future: Future[str] = self._executor.submit(
                self._compute_standard_summary, path
            )
            self._inflight[path] = future
            return future

    def _compute_standard_summary(self, path: Path) -> str:
        try:
            return self._compute_standard_summary_inner(path)
        finally:
            with self._lock:
                self._inflight.pop(path, None)

    def _compute_standard_summary_inner(self, path: Path) -> str:
        content = self._repo.get_note_content(path)
        content_hash = _hash_content(content)
        cached = self._cache.get(path, content_hash)
        if cached is not None:
            return cached
        summary = self._summarize_standard_note(path, content)
        self._cache.set(path, content_hash, summary)
        return summary

    def _summarize_standard_note(self, path: Path, content: str) -> str:
        prompt = (
            "Generate a {min_words}-{max_words} word summary of this note's content.\n"
            "Focus on: main topics, key claims, what questions it answers.\n\n"
            "<note>\n{content}\n</note>"
        ).format(
            min_words=self._config.summary_target_words_min,
            max_words=self._config.summary_target_words_max,
            content=content,
        )
        return request_text(self._client, prompt, f"summary {path.name}")

    def _compute_index_summary(self, path: Path, stack: List[Path]) -> str:
        if path in stack:
            cycle = " -> ".join(item.name for item in stack + [path])
            raise RuntimeError(f"Cycle detected while summarizing index notes: {cycle}")

        content = self._repo.get_note_content(path)
        content_hash = _hash_content(content)
        cached = self._cache.get(path, content_hash)
        if cached is not None:
            return cached

        stack.append(path)
        try:
            summary = self._summarize_index_note(path, content, stack)
        finally:
            stack.pop()

        self._cache.set(path, content_hash, summary)
        return summary

    def _summarize_index_note(self, path: Path, content: str, stack: List[Path]) -> str:
        linked_paths: List[Path] = []
        seen: set[Path] = set()
        for target in extract_wikilinks(content):
            resolved = self._repo.resolve_link(target)
            if resolved is not None and resolved not in seen:
                seen.add(resolved)
                linked_paths.append(resolved)

        standard_paths: List[Path] = []
        index_paths: List[Path] = []
        for linked_path in linked_paths:
            if self._repo.is_index_note(linked_path, self._config.index_filename_suffix):
                index_paths.append(linked_path)
            else:
                standard_paths.append(linked_path)

        futures = {linked_path: self._ensure_future(linked_path) for linked_path in standard_paths}

        summaries: List[str] = []
        for linked_path in index_paths:
            summaries.append(self._compute_index_summary(linked_path, stack))
        for linked_path, future in futures.items():
            summaries.append(future.result())

        joined_summaries = "\n\n".join(summaries)
        prompt = (
            "Generate a summary based on these summaries of linked notes:\n"
            "{summaries}\n\n"
            "Synthesize into {min_words}-{max_words} words describing what this index covers."
        ).format(
            summaries=joined_summaries,
            min_words=self._config.summary_target_words_min,
            max_words=self._config.summary_target_words_max,
        )
        return request_text(self._client, prompt, f"summary {path.name}")
