# Runtime Compatibility

- The dependency lock currently resolves `pydantic-core==2.33.2`, which ships Python 3.13 wheels but falls back to a PyO3 source build on Python 3.14. PyO3 0.24.1 rejects 3.14, so the project needs to stay pinned to Python 3.13 until the dependency set is upgraded to a 3.14-compatible release line.
## Notes filename slug migration

- The notes vault uses hyphen-slug filenames while keeping readable wikilinks. Note resolution and exploration file selection should compare by slug-normalized note title rather than exact filename text.
