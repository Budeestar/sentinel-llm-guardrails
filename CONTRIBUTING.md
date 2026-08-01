# Contributing

Small, focused PRs welcome.

## Rules
- Stay **stdlib-only** in `src/` (no new runtime deps)
- Add a test for every new detection / redaction path
- Prefer precision — noisy false positives will get reverted
- Run `pytest -q` before opening a PR

## Good first contributions
- Reproducible jailbreak bypass + a tight pattern or scorer tweak
- Missing secret formats with low false-positive risk
- Docs / examples (LangChain, LiteLLM, FastAPI)
