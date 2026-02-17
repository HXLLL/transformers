# AGENTS.md

Guidance for coding agents operating in this repo.

## Current Work: Fault-Tolerant LLM Inference

**Goal**: Make inference engine resistant to power loss. Save/recover states (KV cache, etc.) to minimize redundant work on restart.

**Repo Map**
- `src/transformers`: core library code.
- `src/transformers/models`: per-model implementations.
- `tests`: core tests and model tests.
- `docs/source/en`: documentation sources (MDX-like Markdown).

## Coding Style Preferences

This is an experimental and academic project, not user-facing software. Optimize for developer productivity:

- **Short code is better**: Short code implies simple logic and is faster to review. Avoid verbose abstractions.
- **No excessive error handling**: Skip edge cases that won't happen in practice. Basic asserts are fine; elaborate recovery logic is not.
- **No backward-compatibility code**: State files, configs, etc. are deleted between tests. Don't add version checks, optional field handling, or migration logic.
- **No polished UIs**: Command-line tools with minimal output are sufficient.
- **Main purpose first**: Implement the core functionality. Don't add "nice-to-have" features, configurability, or defensive code unless explicitly needed.
- **Maintainability through simplicity**: Fewer lines = fewer bugs = easier to understand and modify.

## State Persistence Rules
- State files are deleted between tests; do not add migrations or versioning
- Keep state format simple; minimal metadata only

## Safety & Hygiene
- Do not modify vendored/third_party unless requested
- Unless necessary, do not modify code other than examples/qwen3/ and mllm/mllm/models/qwen3_i. if you need to do so, confirm it with the user.
- Do not commit generated build outputs or large model files
- Prefer existing libraries over new dependencies


**Boundaries**
- Always: keep diffs minimal, follow existing patterns, prefer targeted tests.
- Ask first: adding dependencies, modifying CI, changing public APIs, large refactors.
- Never: edit generated modeling files when a `modular_<name>.py` exists.
**Tech Stack**
- Python package with PyTorch focus; tests run with `pytest`, formatting via `ruff`.
- Docs built with `doc-builder` (see `docs/README.md`).


**Testing (smallest relevant set)**
- Targeted model tests: `pytest tests/models/<name>/test_modeling_<name>.py`
- Tokenizers/processors: `pytest tests/models/<name>/test_tokenization_<name>.py` or `test_processing_<name>.py`
- For efficiency, you can test only the code you written. you do not need to ran the tests written by others.


**Output Example**
- Good: "Edited `src/transformers/models/bert/modeling_bert.py`, ran `make style`, ran `pytest tests/models/bert/test_modeling_bert.py`."
