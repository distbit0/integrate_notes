# Runtime Compatibility

- The dependency lock currently resolves `pydantic-core==2.33.2`, which ships Python 3.13 wheels but falls back to a PyO3 source build on Python 3.14. PyO3 0.24.1 rejects 3.14, so the project needs to stay pinned to Python 3.13 until the dependency set is upgraded to a 3.14-compatible release line.

# OpenRouter Env Routing

- OpenRouter clients load the repo `.env` with override enabled so the repo-specific `OPENROUTER_API_KEY` wins over inherited shell values. This prevents global keys from merging Integrate Notes usage with other projects.
## Notes filename slug migration

- The notes vault uses hyphen-slug filenames while keeping readable wikilinks. Note resolution and exploration file selection should compare by slug-normalized note title rather than exact filename text.

## Note directive frontmatter

- The grouping directive is now frontmatter (`grouping: |`) rather than body text near the top of notes. Continuous scratchpad integration is opt-in via `organise: continuous`; pending continuous notes must also have `grouping` frontmatter so batch mode does not prompt mid-run.
- `continuous-note-organisation.timer` runs `src/integrate_notes.py --continuous --notes-root /home/pimania/notes` daily at 09:00 as a systemd user timer. The timer unit is stored outside this repo under `~/.config/systemd/user/`.
- The timer depends on Git hooks in `/home/pimania/notes` being able to find `git-lfs`. On 2026-06-07, continuous integration failed at the pre-integration `git push` because user systemd's PATH omitted Homebrew (`/home/linuxbrew/.linuxbrew/bin`), where `git-lfs` is installed. The persistent fix is `~/.config/environment.d/10-user-path.conf`; the running user manager was also updated with `systemctl --user set-environment`.
- On 2026-06-07, manual reruns with an uncommitted `DEFAULT_MODEL = "minimax/minimax-m3"` change reached the LLM step but failed chunk 1 because the model repeatedly emitted malformed patch blocks, including extra `SEARCH` text inside the search span and a combined `<<<<<<< SEARCH DUPLICATE` marker. The script correctly refused to write these patches.
- Continuous mode now writes a default `grouping: |` frontmatter value, with a warning log, when a note is marked `organise: continuous` but has no grouping. This keeps scheduled runs non-interactive while making the default explicit in the note.
