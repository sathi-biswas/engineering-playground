# Bug Report — BUG-1042 (codecracker iter_code_files max_files ordering)

**Codebase:** `/Users/sathibiswas/engineering-playground/engineering-playground/codecracker`
**GitHub:** https://github.com/sathi-biswas/engineering-playground/tree/main/codecracker

## Bug Summary

`iter_code_files` applies `settings.max_files` while walking `root.rglob("*")`, then sorts only the files already collected. `rglob` order follows the filesystem, so the cap keeps whichever code files the OS returns first. `return sorted(found)` only sorts that subset. On a repo larger than the cap (default 400), entrypoints and shallow source files can be omitted, and two machines can analyze different sets of files for the same repo.

`tests/test_ast_parser.py` only checks `len(paths) == 2` for `max_files=2`. It does not check which files were kept.

## Suspected Components

- `codecracker/ast_parser.py` — `iter_code_files`
- `tests/test_ast_parser.py` — `test_respects_max_files`

## How to Fix

In `iter_code_files`, remove the `break` on `len(found) >= settings.max_files`. Append every file that passes the skip, extension, and size checks.

After the walk, sort `found` so the order is stable. Prefer shallower paths, then path string:

```python
root = root.resolve()
found.sort(key=lambda p: (len(p.relative_to(root).parts), p.as_posix()))
```

Apply the cap after that sort:

```python
return found[: settings.max_files]
```

Extend `test_respects_max_files` so a tree with more than `max_files` code files asserts both the length and that the kept paths are the shallowest ones (for example `src/app.py` stays, a deep `tests/...` file is dropped when the cap is small).
