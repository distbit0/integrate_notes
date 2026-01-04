# TODO 7: Multiline grouping approach (excluded from LLM body, protected from patches)

> What exact on-disk syntax do you want for the multiline grouping section? Please provide a concrete before/after example (including where it sits relative to `---` front matter and the `# -- SCRATCHPAD` heading).

after the front matter, before the scratchpad heading. it should be before any other content in the file after the front matter

> How should the end of the multiline grouping section be detected (e.g., first blank line, next heading, a closing marker, end-of-file)? Can the grouping text itself contain blank lines?

use explicit opening and closing marker syntax. if markdown frontmatter satisfies the criteria e.g. support for multiline content, use that. otherwise implement this in the way which makes the most sense / is the most aligned w/ good practice.

> For legacy documents that still use a single-line `Grouping approach: ...` prefix, should the tool leave that line as-is, or migrate it to the new multiline format when writing the file? If migration is desired, should it happen only when `--grouping` is provided / user is prompted, or always?

leave as is 

> The current CLI prompt uses `input()` (single line). How do you want multiline grouping input to be entered (e.g., read until a lone `.` line, read until EOF, open $EDITOR, allow literal `\n` escapes, etc.)?

yes figure out best/most simple but also useable way to support multiline input

> Should the grouping section be preserved verbatim (whitespace/indentation), or normalized (trim lines, collapse spaces) before inserting into the prompt’s “Maintain the grouping approach: …” line?

preserved verbatim 

> Do you want the grouping section to be strictly immutable during patch application (i.e., patches only apply to the body after removing the grouping block), or should we also detect and error if a patch’s SEARCH text matches inside the grouping section?

strictly immutable. it should be as if it didn't exist in the body at all. so if a search block matches it and nothing else, it results in an error > retry. there should not need to be any special handling logic for these cases. it simply isn't part of the document body for the purposes of search/replace or substitute blocks.

