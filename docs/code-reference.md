# Code Reference

Auto-generated from the docstrings and source in `shooter/` — this page
always reflects the current code, not a snapshot. Each module starts with a
`ponytail:` comment explaining its design choice (plain functions over
DataFrames, no class hierarchies); see [Data & Pipeline](data-and-pipeline.md)
for the architecture these modules fit into.

## `shooter.parse`

NDJSON loading, event-time tie-breaking, and the event-replay join that
builds one row per shot.

::: shooter.parse

## `shooter.features`

One function per performance domain; each takes the shot table (+ frame
dict where relevant) and returns it with new feature columns.

::: shooter.features

## `shooter.dashboard`

Plotting + scoring helpers used by the shooter dashboard and PM validation
sections of the notebook.

::: shooter.dashboard

## `shooter.reporting`

Exploratory per-user reporting metrics (drift/bias, split times,
session trend, postural sway) — validated but not promoted into
`features.csv`.

::: shooter.reporting

## `shooter.build_features`

Runs the pipeline over the data folder and writes `features.csv`.

::: shooter.build_features
