# Adoption test (from the viability assessment)

**Decision rule:** publish this module. Give it 30–60 days.

| Signal | Treat as |
|---|---|
| Issues / PRs / forks from people you do not know | Demand evidence → consider Path 2 (air-gap host compliance) or Path 3 (audit-readiness service) |
| Stars only from friends / silence | Archive; do not spend a year packaging the full OS |
| Requests for classifiers / multi-tenant SaaS | Different product — do not bolt onto this repo without a new plan |

## Publish checklist

1. Create a **new public GitHub repo** (do not bury this inside the full SentinelOS tree as the only entry point).
2. Push this directory as the repo root.
3. Replace `YOUR_ORG` in `pyproject.toml` with your GitHub org/user.
4. Tag `v0.1.0`.
5. Optional: `twine` / PyPI upload as `sentinel-llm-guardrails`.
6. Post once where engineers already look (HN Show, r/MachineLearning, LinkedIn technical post) — **one** clear post, not a sales campaign.
7. Do not add features for 30 days unless a stranger files a concrete issue.

## What this test deliberately does not do

- Sell to companies that have never heard of you.
- Close feature gaps with Lakera / Bedrock Guardrails.
- Prove the full SentinelOS product is a fundable company.
