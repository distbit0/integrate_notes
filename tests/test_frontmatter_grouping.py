import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import integrate_notes  # noqa: E402


def test_set_frontmatter_block_field_preserves_body_and_replaces_field() -> None:
    content = (
        "---\n"
        "title: Existing\n"
        "grouping: old\n"
        "---\n"
        "# Body\n\n"
        "Text\n"
    )

    updated = integrate_notes.set_frontmatter_block_field(
        content,
        "grouping",
        "Group by topic\nThen by mechanism",
    )

    assert integrate_notes.read_frontmatter_field(updated, "title") == "Existing"
    assert integrate_notes.read_frontmatter_field(updated, "grouping") == (
        "Group by topic\nThen by mechanism"
    )
    assert updated.endswith("# Body\n\nText\n")
    assert "grouping: old" not in updated


def test_frontmatter_block_field_can_contain_separator_lines() -> None:
    updated = integrate_notes.set_frontmatter_block_field(
        "# Body\n",
        "grouping",
        "First group\n---\nSecond group",
    )

    assert integrate_notes.read_frontmatter_field(updated, "grouping") == (
        "First group\n---\nSecond group"
    )
    assert updated.endswith("# Body\n")


def test_set_frontmatter_block_field_creates_frontmatter_when_missing() -> None:
    updated = integrate_notes.set_frontmatter_block_field(
        "# Body\n",
        "grouping",
        "Group by question",
    )

    assert updated.startswith("---\ngrouping: |\n  Group by question\n---\n")
    assert updated.endswith("# Body\n")


def test_continuous_organise_paths_selects_non_empty_scratchpads(
    tmp_path: Path,
) -> None:
    ready_note = tmp_path / "ready.md"
    ready_note.write_text(
        "---\n"
        "organise: continuous\n"
        "grouping: |\n"
        "  Group by topic\n"
        "---\n"
        "# Body\n\n"
        "# -- SCRATCHPAD\n\n"
        "new point\n",
        encoding="utf-8",
    )
    empty_scratchpad = tmp_path / "empty.md"
    empty_scratchpad.write_text(
        "---\n"
        "organise: continuous\n"
        "grouping: topic\n"
        "---\n"
        "# Body\n\n"
        "# -- SCRATCHPAD\n",
        encoding="utf-8",
    )
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.md").write_text(
        "---\norganise: continuous\ngrouping: topic\n---\n# -- SCRATCHPAD\n\nx\n",
        encoding="utf-8",
    )

    assert integrate_notes.continuous_organise_paths(tmp_path) == [ready_note]


def test_continuous_organise_paths_adds_default_grouping_for_pending_notes(
    tmp_path: Path,
) -> None:
    note = tmp_path / "missing-grouping.md"
    note.write_text(
        "---\norganise: continuous\n---\n# Body\n\n# -- SCRATCHPAD\n\nnew point\n",
        encoding="utf-8",
    )

    assert integrate_notes.continuous_organise_paths(tmp_path) == [note]

    updated = note.read_text(encoding="utf-8")
    assert integrate_notes.read_frontmatter_field(updated, "grouping") == (
        integrate_notes.DEFAULT_GROUPING
    )
    assert "# -- SCRATCHPAD\n\nnew point\n" in updated
