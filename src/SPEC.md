# Zettelkasten Inbox Integration Script — Specification

## Overview

A script that automatically integrates new text from an inbox note into the most relevant existing notes in a zettelkasten-style markdown repository, using LLM-guided exploration of the note graph.

---

## Phase 1: Chunking

### Input

* Inbox file containing new text to integrate
* Filenames of 15 randomly sampled files (>300 words each, excluding index notes) from the repository

### Process

1. Number each paragraph in the inbox
2. Provide LLM with:

   * The numbered paragraphs
   * The 15 sampled filenames (for granularity calibration)
3. LLM returns groups of paragraph numbers representing semantically coherent chunks

### Constraints

* Max 600 words per chunk (but never split a single paragraph)
* Paragraphs within a chunk need not be contiguous
* Groups should only combine paragraphs that are clearly same topic/chain of thought

### Rationale

* LLM chunking ensures semantic coherence; mechanical chunking conflates proximity with relatedness
* Sampled filenames calibrate the LLM to match existing note granularity (filenames alone convey topic scope without token cost)
* Non-contiguous grouping allows related but separated paragraphs to be processed together

---

## Phase 2: Summary Generation (Preprocessing)

### Cache Invalidation

* Store `(file_path, content_hash, summary)` tuples
* Regenerate summary when `hash(current_content) != cached_hash`

### Summary Generation Rules

**Standard notes:**

```
Generate a 75-100 word summary of this note's content.
Focus on: main topics, key claims, what questions it answers.
```

**Index notes** (filename contains "index"):

```
Generate a summary based on these summaries of linked notes:
[summaries of all notes linked from this file]
Synthesize into 75-100 words describing what this index covers.
```

### Rationale

* Hash-based invalidation is precise—updates exactly when needed
* Index notes contain mostly links; summarizing their linked content is more informative than summarizing the links themselves

---

## Phase 3: Exploration

### State Model

Each file can be in one of three states:

| State           | What LLM sees                                  | How it gets there           |
| --------------- | ---------------------------------------------- | --------------------------- |
| **Available**   | Filename + summary                             | Linked from a viewed file   |
| **Viewed**      | Filename + summary + headings + outgoing links | LLM requested to view it    |
| **Checked out** | Full content                                   | LLM selected it for editing |

### Exploration Flow

```
1. Initialize:
   - Root file is automatically VIEWED (summary + headings + links shown)
   - All files linked from root are AVAILABLE (filename + summary shown)
   
2. Exploration loop:
   a. LLM sees: chunk + all VIEWED files (summary/headings/links) + AVAILABLE files (filename/summary)
   b. LLM returns: list of AVAILABLE files to VIEW (up to 4 per round)
   c. For each requested file:
      - Change state to VIEWED
      - Show summary + headings + outgoing links
      - Files it links to become AVAILABLE (if not already viewed)
   d. Repeat until LLM signals ready OR limits reached
   
3. Checkout:
   - LLM selects up to 3 VIEWED files to CHECK OUT
   - Full content of checked-out files shown
   
4. Edit:
   - LLM provides find/replace blocks for checked-out files
```

### Limits

* Max 3 exploration rounds
* Up to 4 files may be VIEWED per round (fewer is fine)
* Max 15 files VIEWED total
* Max 3 files CHECKED OUT

### Context Management

* Only summaries (not full content) accumulate during exploration
* Full content only loaded at checkout
* Keeps exploration cheap regardless of depth

### Rationale

* Three-state model separates cheap browsing from expensive content loading
* AVAILABLE shows summary so LLM can judge relevance; VIEWED adds structure (headings + links) for navigation decisions
* Summaries + headings provide enough signal for navigation decisions
* Root file treated identically to others; may itself be edited or contain no links

---

## Phase 4: Editing

### Edit Format

```json
{
  "edits": [
    {
      "file": "filename.md",
      "find": "exact text to locate",
      "replace": "replacement text",
      "is_duplicate": false
    },
    {
      "file": "other.md",
      "find": "text that already covers this",
      "is_duplicate": true
    }
  ]
}
```

### Edit Types

**Standard edit:** `find` + `replace` provided, content is modified

**Insertion:** `find` contains anchor text, `replace` contains anchor + new content

```json
{
  "find": "- Link B",
  "replace": "- Link B\n- Link C"
}
```

**Duplicate marker:** `is_duplicate: true`, only `find` required

* `find` contains existing text that already covers the chunk content
* No replacement made; serves as visibility into why content wasn't added

### Validation

1. For each edit, search for `find` text in specified file using a whitespace-normalized match (treat runs of spaces/tabs/newlines as equivalent, and ignore trivial leading/trailing whitespace differences) to increase match reliability
2. Must match exactly once (zero matches = error, multiple matches = error)
3. On validation failure: return error to LLM, request correction within same conversation

### Scope

* Edits can target any CHECKED OUT file
* This includes the root file and index notes

### Rationale

* Find/replace is simple and unambiguous; insertion is just a usage pattern, not a separate operation
* Single-match requirement prevents ambiguous edits
* Duplicate flag provides audit trail without cluttering output with identical find/replace pairs
* In-conversation correction leverages existing context rather than restarting

---

## Tool Schema

### Exploration Tools

```typescript
// Request to view files (see headings + links in addition to summary)
interface ViewFilesRequest {
  action: "view";
  files: string[];  // up to 4, must be AVAILABLE
}

// Signal ready to check out files for editing
interface CheckoutRequest {
  action: "checkout";
  files: string[];  // max 3, must be VIEWED
}
```

### Edit Tools

```typescript
interface Edit {
  file: string;
  find: string;
  replace?: string;      // omit if is_duplicate
  is_duplicate: boolean;
}

interface EditRequest {
  action: "edit";
  edits: Edit[];
}
```

---

## File Annotation Format

When displaying a VIEWED file:

```markdown
## [filename.md]

**Summary:** [75-100 word summary]

**Headings:**
- # Main Title
- ## Section One
- ## Section Two
- ### Subsection

**Links to:**
- [[other-note.md]] — [summary of other-note]
- [[another.md]] — [summary of another]
```

When displaying an AVAILABLE file:

```markdown
- [[filename.md]] — [75-100 word summary]
```

---

## Execution Flow Summary

```
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING (run periodically)                       │
│  - Update stale summaries (hash-based invalidation)     │
│  - Index notes: summarize from linked summaries         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  CHUNKING                                               │
│  - Show LLM: inbox paragraphs + 15 sample filenames     │
│  - LLM returns: paragraph groupings (max 600w each)     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  FOR EACH CHUNK:                                        │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ EXPLORE (max 3 rounds, up to 4 files per round,   │  │
│  │          max 15 files viewed total)               │  │
│  │ - AVAILABLE: see filename + summary               │  │
│  │ - VIEWED: see summary + headings + links          │  │
│  │ - Request more files or signal ready              │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ CHECKOUT (max 3 files)                            │  │
│  │ - Load full content of selected files             │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ EDIT                                              │  │
│  │ - LLM provides find/replace blocks                │  │
│  │ - Validate single-match constraint                │  │
│  │ - Apply edits or request correction               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Configuration

```yaml
# Limits
max_exploration_rounds: 3
max_files_viewed_per_round: 4
max_files_viewed_total: 15
max_files_checked_out: 3
max_chunk_words: 600
granularity_sample_size: 15
granularity_sample_min_words: 300

# Paths
root_file: "index.md"
inbox_file: "inbox.md"
notes_directory: "./notes"
summary_cache: "./.summary_cache.json"

# Summary
summary_target_words: 75-100
index_filename_pattern: "index"
```

---

## Out of Scope (Deliberate Simplifications)

| Feature                           | Reason excluded                                                 |
| --------------------------------- | --------------------------------------------------------------- |
| Create new note                   | Adds complexity; can be added later                             |
| Explicit insert_after operation   | Find/replace pattern sufficient                                 |
| Summary update debouncing         | Premature optimization                                          |
| Pre-routing with all summaries    | Doesn't scale; exploration achieves same goal                   |
| Confidence ratings / review queue | Adds friction; start simple                                     |
| Multiple root files / fallbacks   | Unnecessary if root is maintained                               |
| Heading-level routing             | Single dimension (files) is simpler than two (files + sections) |