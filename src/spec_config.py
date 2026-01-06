from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRATCHPAD_HEADING = "# -- SCRATCHPAD"
ENV_API_KEY = "OPENAI_API_KEY"

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_REASONING = {"effort": "medium"}

DEFAULT_MAX_RETRIES = 3
RETRY_INITIAL_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0

MAX_PATCH_ATTEMPTS = 3
MAX_TOOL_ATTEMPTS = 3
MAX_CHUNKING_ATTEMPTS = 3

MAX_CONCURRENT_VERIFICATIONS = 4

LOG_FILE_ROTATION_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class SpecConfig:
    max_exploration_rounds: int = 3
    max_files_viewed_per_round: int = 4
    max_files_viewed_total: int = 15
    max_files_checked_out: int = 3
    max_chunk_words: int = 600
    granularity_sample_size: int = 15
    granularity_sample_min_words: int = 300
    summary_target_words_min: int = 75
    summary_target_words_max: int = 100
    index_filename_suffix: str = "index.md"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_log_path() -> Path:
    return repo_root() / "logs" / "integrate_notes.log"


def default_pending_prompts_path() -> Path:
    return repo_root() / "logs" / "pending_verification_prompts.json"


def default_summary_cache_path() -> Path:
    return Path.home() / ".cache" / "integrate_notes" / "summary_cache.json"


def load_config(config_path: Path) -> SpecConfig:
    if not config_path.exists():
        return SpecConfig()

    raw = config_path.read_text(encoding="utf-8")
    if not raw.strip():
        return SpecConfig()

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("config.json must contain a JSON object.")

    defaults = SpecConfig()
    overrides: dict[str, Any] = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name not in data:
            continue
        value = data[field_name]
        expected_value = getattr(defaults, field_name)
        if not isinstance(value, type(expected_value)):
            raise ValueError(
                f"config.json field '{field_name}' must be {type(expected_value).__name__}."
            )
        overrides[field_name] = value

    return SpecConfig(**{**defaults.__dict__, **overrides})
