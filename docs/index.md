# Shooter Performance Feature Engineering

Eyeviation home assignment: turn VR shooting-range `.event.json` logs into
3-5 interpretable shot-quality features, then present them to two
audiences — the shooter (plain-language dashboard) and a PM (why each
feature was chosen, how it's calculated, and how it could feed the product).

This site is a navigable companion to the actual deliverable
([`analysis.ipynb`](analysis.ipynb)) and the code that produces it. Nothing
here is duplicated by hand — the **Full Analysis** page renders the executed
notebook directly (prose + plots), and **Code Reference** renders the
current source + docstrings directly from `shooter/`. If either changes,
rebuild the site and this stays correct automatically.

## Where to go

- **[Data & Pipeline](data-and-pipeline.md)** — the event-log format, the
  parse → features → build pipeline, and the data quirks/assumptions worth
  knowing before reading anything else.
- **[Full Analysis](analysis.ipynb)** — the notebook itself: EDA, the
  shooter dashboard, feature explanations, validation, within-session
  drift, features considered but not built, and the PM summary.
- **[Code Reference](code-reference.md)** — every function in `shooter/`,
  auto-generated from its docstring and source.

## Running it locally

```bash
uv sync
uv run python -m shooter.build_features                 # rebuild features.csv
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
uv run mkdocs serve                                      # this site, live-reloading
```

See `README.md` in the repo root for the PDF-export commands.
