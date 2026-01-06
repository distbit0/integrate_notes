from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, List, Sequence
from uuid import uuid4

from loguru import logger

from spec_config import MAX_CONCURRENT_VERIFICATIONS, default_pending_prompts_path
from spec_llm import request_text


NOTIFY_SEND_PATH = shutil.which("notify-send")
_NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED = False


def notify_missing_verification(
    chunk_index: int, total_chunks: int, assessment: str
) -> None:
    global _NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED
    title = "Integration verification missing content"
    body = f"Chunk {chunk_index + 1}/{total_chunks}: {assessment}"
    if NOTIFY_SEND_PATH:
        try:
            subprocess.run(
                [
                    NOTIFY_SEND_PATH,
                    "--app-name=IntegrateNotes",
                    title,
                    body,
                ],
                check=True,
            )
        except Exception as error:
            logger.warning(
                f"notify-send failed for verification chunk {chunk_index + 1}: {error}"
            )
    else:
        if not _NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED:
            logger.warning(
                "notify-send not available; desktop alerts for verification issues disabled."
            )
            _NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED = True


@dataclass(frozen=True)
class DuplicateEvidence:
    body_text: str


class VerificationManager:
    def __init__(self, client, target_file: Path) -> None:
        self.client = client
        self.pending_path = default_pending_prompts_path()
        self.lock = Lock()
        self.active_lock = Lock()
        self.active_ids: set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VERIFICATIONS)
        self.new_prompt_event = Event()
        self.stop_requested = False
        self.tracked_file_name = Path(target_file).resolve().name
        self.worker = Thread(
            target=self._run,
            name="VerificationManager",
            daemon=True,
        )
        self.worker.start()

    def enqueue_prompt(
        self,
        prompt: str,
        context_label: str | None,
        chunk_index: int | None,
        total_chunks: int | None,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Verification prompt must be a non-empty string.")

        entry = {
            "id": str(uuid4()),
            "prompt": prompt,
            "context_label": context_label,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "file_name": self.tracked_file_name,
        }
        with self.lock:
            entries = self._read_entries_locked()
            entries.append(entry)
            self._write_entries_locked(entries)
        self.new_prompt_event.set()

    def shutdown(self) -> None:
        self.stop_requested = True
        self.new_prompt_event.set()
        if self.worker.is_alive():
            self.worker.join()
        self.executor.shutdown(wait=True)

    def _run(self) -> None:
        while True:
            try:
                self._dispatch_pending()
            except Exception as error:
                logger.exception(
                    f"Verification dispatcher encountered an error: {error}"
                )
            if self.stop_requested and not self._has_pending_work():
                break
            self.new_prompt_event.wait(timeout=0.5)
            self.new_prompt_event.clear()

    def _dispatch_pending(self) -> None:
        with self.lock:
            all_entries = self._read_entries_locked()
            entries = self._entries_for_current_file_locked(all_entries)

        for entry in entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            with self.active_lock:
                if entry_id in self.active_ids:
                    continue
                self.active_ids.add(entry_id)

            future = self.executor.submit(self._send_prompt, entry)
            future.add_done_callback(
                lambda fut, data=entry: self._handle_result(data, fut)
            )

    def _send_prompt(self, entry: dict[str, Any]) -> str:
        context_label = entry.get("context_label") or "verification"
        prompt = entry["prompt"]
        return request_text(self.client, prompt, f"verification {context_label}")

    def _handle_result(self, entry: dict[str, Any], future) -> None:
        entry_id = entry.get("id")
        try:
            assessment = future.result()
        except Exception as error:  # noqa: BLE001
            context_label = entry.get("context_label") or "verification"
            logger.exception(f"Verification for {context_label} failed: {error}")
            if entry_id:
                with self.active_lock:
                    self.active_ids.discard(entry_id)
            self.new_prompt_event.set()
            return

        self._log_assessment(entry, assessment)

        if entry_id:
            self._remove_entry(entry_id)
            with self.active_lock:
                self.active_ids.discard(entry_id)

        self.new_prompt_event.set()

    def _log_assessment(self, entry: dict[str, Any], assessment: str) -> None:
        chunk_index = entry.get("chunk_index")
        total_chunks = entry.get("total_chunks")
        context_label = entry.get("context_label") or "verification"
        file_name = entry.get("file_name")

        if not file_name:
            raise RuntimeError(
                "Verification entry missing required file_name; pending prompts file may be corrupted."
            )

        base_header = f'Verification "{file_name}"'

        if (
            isinstance(chunk_index, int)
            and isinstance(total_chunks, int)
            and 0 <= chunk_index < total_chunks
        ):
            if "MISSING" in assessment:
                notify_missing_verification(chunk_index, total_chunks, assessment)
            chunk_header = f"{base_header}:"
            if assessment.startswith(chunk_header):
                logger.info(assessment)
            else:
                logger.info(f"{chunk_header}\n{assessment}")
        else:
            if context_label != "verification":
                header = f"{base_header} ({context_label}):"
            else:
                header = f"{base_header}:"
            if assessment.startswith(header):
                logger.info(assessment)
            else:
                logger.info(f"{header}\n{assessment}")

    def _remove_entry(self, entry_id: str) -> None:
        with self.lock:
            entries = self._read_entries_locked()
            remaining = [item for item in entries if item.get("id") != entry_id]
            self._write_entries_locked(remaining)

    def _read_entries_locked(self) -> List[dict[str, Any]]:
        if not self.pending_path.exists():
            return []
        raw = self.pending_path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Pending verification prompts file {self.pending_path} is corrupted: {error}"
            ) from error
        if not isinstance(data, list):
            raise RuntimeError(
                f"Pending verification prompts file {self.pending_path} must contain a list."
            )
        return data

    def _write_entries_locked(self, entries: List[dict[str, Any]]) -> None:
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(entries, ensure_ascii=True, indent=2)
        self.pending_path.write_text(payload, encoding="utf-8")

    def _has_pending_work(self) -> bool:
        with self.lock:
            entries = self._read_entries_locked()
            has_entries = bool(self._entries_for_current_file_locked(entries))
        with self.active_lock:
            has_active = bool(self.active_ids)
        return has_entries or has_active

    def _entries_for_current_file_locked(
        self, entries: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        invalid_entries: List[dict[str, Any]] = []
        relevant_entries: List[dict[str, Any]] = []

        for entry in entries:
            file_name = entry.get("file_name")
            entry_id = entry.get("id")
            if not file_name or not entry_id:
                invalid_entries.append(entry)
                continue
            if file_name == self.tracked_file_name:
                relevant_entries.append(entry)

        if invalid_entries:
            invalid_count = len(invalid_entries)
            suffix = "y" if invalid_count == 1 else "ies"
            logger.warning(
                f"Removed {invalid_count} invalid verification prompt entr{suffix} missing file metadata or IDs."
            )
            cleaned_entries = [
                entry for entry in entries if entry not in invalid_entries
            ]
            self._write_entries_locked(cleaned_entries)

        return relevant_entries


def build_verification_prompt(
    chunk_text: str,
    patch_replacements: Sequence[str],
    duplicate_texts: Sequence[str],
) -> str:
    response_instructions = (
        "Report whether any note content is missing or materially altered."
        " Respond with a concise single paragraph beginning with 'OK -' if everything is covered"
        " or 'MISSING -' followed by details of any omissions."
        " Separate each omission by two newlines and for each omission, provide the following:\n"
        '    Notes:"..."\n'
        '    Body:"..."\n'
        '    Explanation: "..."\n'
        '    Proposed Fix: "..."\n'
        "Quote the exact text from the notes chunk containing the missing detail and quote the exact passage from the patch replacements or duplicate evidence that should cover it (or state Body:\"<not present>\" if nothing is relevant)."
        " Explain precisely what information is still missing or altered without omitting any nuance."
    )

    if patch_replacements:
        replacement_sections = []
        for index, replacement_text in enumerate(patch_replacements, start=1):
            replacement_sections.append(
                f"[Patch {index} Replacement]\n{replacement_text}"
            )
        replacements_block = "\n\n".join(replacement_sections)
    else:
        replacements_block = "<no patch replacements provided>"

    if duplicate_texts:
        duplication_sections = []
        for index, body_text in enumerate(duplicate_texts, start=1):
            duplication_sections.append(
                f"[Duplicate {index} Evidence]\nBody:\n{body_text}"
            )
        duplications_block = "\n\n".join(duplication_sections)
    else:
        duplications_block = "<no duplicate evidence provided>"

    sections = [
        (
            "<task>"
            "You are verifying that every idea/point/concept/argument/detail/url/[[wikilink]]/diagram etc. "
            "from the provided notes chunk has been integrated into the document body."
            " Use the patch replacements to understand what will be inserted or rewritten."
            " Duplicate evidence is existing body text claimed to already cover notes."
            " If duplicate evidence does not fully cover the notes text, treat the missing detail as missing."
            "</task>"
        ),
        f"<notes_chunk>\n{chunk_text}\n</notes_chunk>",
        f"<patch_replacements>\n{replacements_block}\n</patch_replacements>",
        f"<duplicate_evidence>\n{duplications_block}\n</duplicate_evidence>",
        f"<response_guidelines>\n{response_instructions}\n</response_guidelines>",
    ]
    return "\n\n\n\n\n".join(sections)


def format_verification_assessment(assessment: str) -> str:
    return (
        assessment.replace(" - Notes:", "\nNotes:")
        .replace(" Body:", "\nBody:")
        .replace(" Explanation:", "\nExplanation:")
    )
