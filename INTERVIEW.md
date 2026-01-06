# SPEC.md Implementation

## Scope & entry point
> Should the SPEC flow fully replace the current scratchpad-based integration in `src/integrate_notes.py`, or should it live as a new CLI/module (e.g., `src/zet_integrator.py`) with its own entry point?

the inbox referred to in the spec i.e. where the text is coming from should still be the scratchpad of the file specified in the cli, the same way it is currently. the "root" note mentioned in the spec is just whichever file is passed to integrate_notes.py i.e. the file which we read the scratchpad of.

> If it replaces the current flow, should we remove the existing scratchpad/grouping CLI flags and related logic entirely (per “no backward compatibility”), or keep any pieces (which ones)?

ok actually so what i want to do is keep the old version of integrate_notes.py untouched, but implement the SPEC.md in a new py file, and in this file, remove all grouping instructions related logic. but keep scratchpad stuff, as that is where the notes which are being chunked/integrated are coming from.

## Notes repository & config
> What are the concrete paths for `notes_directory`, `root_file`, and `inbox_file` in your environment (relative to repo or absolute), and should they be set via `config.json`, CLI flags, or environment variables (and which should take precedence)?

inbox file is just the scratchpad of root file, and root file is just the file provided via the cli, and notes directory is just whatever directory root file is in. inbox file is not a separate file, despite what the spec says.

> For `index_filename_pattern`, is the simple substring match in the spec (“index”) correct for your notes, or do you want a stricter rule (e.g., suffix/prefix/regex)?

if it ends in index.md, it is an index file.

> Should we treat non-note markdown files in the notes directory (templates, archives, etc.) as excluded by default? If yes, how do you want them identified (folder name, filename pattern, front matter flag)?

files are only "included" if they linked to, directly or indirectly (through another file) from the root file. only markdown files can be linked to. the fact that you ask me this seems to indicate you have some confusion though, as i don't see a need to exclude files just because they are in the dir. we don't scan all files in the dir. we only look at linked files and derive their path from the link text, .md and the root file dir (noting that links are case insensitive though)

## Chunking
> When numbering paragraphs in the inbox, do you want paragraphs split strictly on blank lines (current `normalize_paragraphs` behavior), or should we treat other separators (e.g., horizontal rules) as paragraph boundaries too?

just blank lines

> For non-contiguous chunk groups returned by the model, should we preserve original paragraph order when assembling each chunk, or should we follow the order returned by the model?

the latter.

> For the “15 randomly sampled filenames (>300 words)” calibration step, should the sample be deterministic (seeded) for reproducibility, and should word counts ignore front matter/code blocks?

doesn't need to be seeded. doesn't need to ignore front matter. note that these must only be files which are linked to (recursively/indirectly or directly) from the root note. not just any files in the dir.

## Summaries & cache
> For summary cache storage, should we use the spec’s `.summary_cache.json` at repo root, or do you prefer a different location/format (and should it be tracked in git or ignored)?

no put it outside the directory to avoid creating a mess. 

> For index-note summaries (summaries of linked notes), should we include links that resolve outside `notes_directory`, or only within it? How should broken/missing links be handled (error vs. skip)?

skip links for which no notes file exists. links are NOT capable of resolving outside of the notes directory, as the link text just specifies the file name and the directory is always implicitly notes directory.

> Should summary generation happen as a separate command (preprocessing), or on-demand during integration if a needed summary is stale/missing?

on-demand. but make sure it is maximally parallelised, to avoid needing to wait a long time.

## Markdown parsing & links
> Which link syntaxes should count as “outgoing links” for exploration: `[[wikilink]]`, `[text](file.md)`, bare `file.md`, or something else? Any special handling for anchors like `[[note#section]]`?

only wikilinks. if it has a #section in the wikilink, just ignore the section. too complex. 

> For headings extraction, should we only parse ATX `#` headings (ignore Setext), and should headings inside code blocks be ignored?

ignore setext. do not ignore headings inside code blocks. too complex

## LLM interaction & tools
> Do you want true tool calling via OpenAI Responses API tools (structured `action: view/checkout/edit`), or is strict JSON schema parsing of a normal response sufficient as long as the schema matches the spec?

yes use tools.

> Which model and reasoning level should we use for chunking, summaries, exploration, and edit phases? Should they all be the same (currently `gpt-5.2`/`medium`) or different per phase?

yes use medium for all.

> When an edit validation fails (0 or >1 matches), should we send the model just the failure details, or also include the specific matching candidates/snippets from the file to help it correct?

just failure details.

## Editing & validation
> The spec’s whitespace-normalized matching treats runs of whitespace as equivalent. Should we still allow the exact/trimmed match fallbacks currently used in `_locate_search_text`, or switch strictly to the whitespace-normalized approach only?

both (but simplify if possible). be quite permissive here. we like to avoid unnecessary matching errors.

> For `is_duplicate: true` edits, do you want them surfaced in output/logs (and how), or just kept internally for audit?

preserve the existing verification logic, but update it to support these new edit request formats. 

## Logging & runtime behavior
> Where should logs live (the spec doesn’t say, but project rules say “move logs outside src/”)? Is `./logs/integrate_notes.log` acceptable, or do you want a different path/rotation policy?

yes that is ok.

> Should the integration run be single-threaded for determinism, or is limited concurrency acceptable for summary generation/exploration?

concurrency is desired for any api calls.


also pls split up the code, including the existing logic, across multiple semantically-meaningful files each focusing on a different area, as the current file is wayy too long.

## Follow-ups
> You asked to keep the old `integrate_notes.py` untouched but also to split the existing logic into multiple files. Which should take priority: (a) leave `integrate_notes.py` as-is and only add new modules for the SPEC flow, or (b) refactor the existing logic into multiple files while keeping behavior unchanged?

keep the old one. but duplicate it and refactor + implement new logic in this duplicate version.

> For the summary cache “outside the directory,” do you want it outside the notes directory but still inside the repo (e.g., `./.summary_cache.json`), or in a user-level cache dir (e.g., `~/.cache/integrate_notes/summary_cache.json`)? If user-level, which exact path should we use on Fedora?

user level. up to you re: path
