from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from spec_markdown import count_words, extract_headings, extract_wikilinks


@dataclass(frozen=True)
class ViewedNote:
    path: Path
    summary: str
    headings: List[str]
    links: List[Path]
    link_summaries: List[tuple[Path, str]]


class NoteRepository:
    def __init__(self, root_path: Path, root_body: str, notes_dir: Path) -> None:
        self._root_path = root_path
        self._root_body = root_body
        self._notes_dir = notes_dir
        self._content_cache: Dict[Path, str] = {}
        self._file_index = self._build_file_index()

    def _build_file_index(self) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for path in self._notes_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".md":
                mapping[path.name.lower()] = path
        return mapping

    def resolve_link(self, link_text: str) -> Optional[Path]:
        target = link_text.strip()
        if not target:
            return None
        if not target.lower().endswith(".md"):
            target = f"{target}.md"
        return self._file_index.get(target.lower())

    def get_note_content(self, path: Path) -> str:
        if path == self._root_path:
            return self._root_body
        cached = self._content_cache.get(path)
        if cached is not None:
            return cached
        content = path.read_text(encoding="utf-8")
        self._content_cache[path] = content
        return content

    def get_headings(self, path: Path) -> List[str]:
        return extract_headings(self.get_note_content(path))

    def get_links(self, path: Path) -> List[Path]:
        links: List[Path] = []
        for target in extract_wikilinks(self.get_note_content(path)):
            resolved = self.resolve_link(target)
            if resolved is not None:
                links.append(resolved)
        return links

    def get_word_count(self, path: Path) -> int:
        return count_words(self.get_note_content(path))

    def is_index_note(self, path: Path, index_suffix: str) -> bool:
        return path.name.lower().endswith(index_suffix.lower())

    def iter_reachable_paths(self) -> List[Path]:
        visited: Set[Path] = set()
        stack: List[Path] = [self._root_path]
        while stack:
            path = stack.pop()
            if path in visited:
                continue
            visited.add(path)
            for link in self.get_links(path):
                if link not in visited:
                    stack.append(link)
        return list(visited)

    def set_root_body(self, body: str) -> None:
        self._root_body = body
        self._content_cache.pop(self._root_path, None)

    def invalidate_content(self, path: Path) -> None:
        self._content_cache.pop(path, None)
