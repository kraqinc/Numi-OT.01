# Contributing to Numi-OT.01

1. Keep architecture free of circular imports (`architecture/` must not import `model/`).
2. No Qwen/other-vendor identity strings in prompts or seeds.
3. Add tests for new modules under `tests/`.
4. Run `make test` before pushing.
5. Secrets never in git — use `.env.example` only as a template.
6. Prefer small, reviewable commits.
