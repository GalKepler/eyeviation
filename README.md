# Shooter Performance Feature Engineering

Eyeviation home assignment: turn VR shooting-range event logs into 3-5
interpretable shot-quality features and a two-audience report (shooter +
PM). See `Data Scientist -  HA Eyeviation.pdf` for the brief and `CLAUDE.md`
for the full data-format reference.

## Layout

```
shooter/
  parse.py           NDJSON streaming reader + the event-replay join that
                      builds one row per shot (drill/repetition/buzzer/
                      requirement state tracking, target-path matching).
  features.py        4 feature domains: reaction time (timing), precision/
                      grouping (motor), trigger control, gaze/pupil/Quiet Eye
                      (eye/cognitive); plus decision-quality (cognitive,
                      derived).
  dashboard.py        Scoring + plotting helpers used by the notebook.
  reporting.py        Exploratory per-user metrics (drift/bias, split times,
                      speed/accuracy tradeoff, session trend, postural sway)
                      on top of the headline features; validated but not
                      promoted into features.csv.
  build_features.py   Runs the pipeline over the data folder -> features.csv.
analysis.ipynb         The report: EDA, feature explanations, validation,
                       shooter dashboard, PM summary. The actual deliverable.
make_notebook.py       Generates analysis.ipynb from source (one-off; edit
                        this and rerun if the notebook needs structural
                        changes, otherwise edit analysis.ipynb directly).
features.csv            Cached per-shot feature table (output of the pipeline).
```

## Running it

```
uv sync
uv run python -m shooter.build_features        # writes features.csv
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
uv run mkdocs serve                            # browsable docs site (http://127.0.0.1:8000)
```

`mkdocs.yml` + `docs/` build a small site (notebook + code reference +
data/pipeline overview) on top of the same sources — see `docs/index.md`.

### Producing the PDF report

No LaTeX toolchain or `playwright` is assumed to be installed, so the PDF is
produced via HTML + a headless browser instead of nbconvert's native
`--to pdf`:

```
uv run jupyter nbconvert --to html --no-input --HTMLExporter.mathjax_url="" \
  analysis.ipynb --output analysis_report.html
google-chrome --headless --disable-gpu --no-sandbox \
  --print-to-pdf="$(pwd)/analysis_report.pdf" --print-to-pdf-no-header \
  "file://$(pwd)/analysis_report.html"
```

(Swap `google-chrome` for `chromium`/`chromium-browser` if that's what's
available; any Chromium-family browser's `--headless --print-to-pdf` works.)

## Data assumptions worth knowing before reading the notebook

- **Event tie-breaking:** a `shooting_requirement_change(clear)` can share
  the exact same millisecond `Time` as the `shot_start` it was triggered by.
  `parse.py` breaks such ties so `shot_start` is processed first — otherwise
  the requirement looks like it was already cleared before the shot that
  satisfied it. After this fix, our recomputed valid-hit flag agrees with the
  engine's own `object_type` on 100% of shots across all 8 sample files.
- **Eye-tracking fallback:** one user's stated `dominant_eye` has
  systematically poor tracking validity in the data; gaze features fall back
  to the other eye per-shot when the dominant eye doesn't clear a minimum
  valid-frame threshold, rather than dropping that user's gaze data entirely.
- **Cohort scope:** 7 of the 8 sample files are the same `TargetAcquisition`
  scenario (used for cross-user comparison); the 8th (`WarmUp`, user 243) is
  excluded from norms as a different task — see analysis.ipynb Section 4.

Full reasoning for every feature/assumption is in `analysis.ipynb`, not
duplicated here.
