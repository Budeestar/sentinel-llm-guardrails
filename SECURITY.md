# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.2.x | Yes |
| < 0.2 | Best-effort |

## Reporting a vulnerability

Open a **private** GitHub security advisory on this repo, or email the maintainer via the GitHub profile.

Please include:
- Minimal repro input that bypasses a check (or leaks a secret past redaction)
- Expected vs actual behaviour
- Whether you want credit in the advisory

Do **not** open a public issue for active bypasses until a fix or disclosure window is agreed.

## Scope / non-goals

This library is **heuristic**. Novel jailbreaks and encoding tricks will evade it.
That is expected — report high-impact, reproducible bypasses; “train a model” requests are out of scope for this repo.
