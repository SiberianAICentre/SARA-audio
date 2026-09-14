# Contributing

Start with [Development](docs/DEVELOPMENT.md) and
[Architecture](docs/ARCHITECTURE.md). Behavioral changes to MIC models must also
follow [the methodology acceptance procedure](docs/METHODOLOGY.md).

Every change should include focused tests, pass Ruff and pytest, preserve the
documented output contract, and avoid committing customer audio, transcripts,
results, virtual environments, model caches, or credentials. GPU/runtime changes
also require an end-to-end smoke test on the target deployment profile.

No public contribution or redistribution license is currently declared. Obtain
authorization from the repository owner before distributing code or derivative
work.
