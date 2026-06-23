# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## What this is

Eyeviation home-assignment data project: turns VR shooting-range NDJSON event
logs into 3-5 interpretable shot-quality features and a two-audience
(shooter + PM) report. `analysis.ipynb` is the actual deliverable; everything
in `shooter/` is the pipeline that feeds it.

## Commands

```bash
uv sync                                                  # install deps
uv run python -m shooter.build_features                 # rebuild features.csv from raw event files
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # rerun the notebook
```

Per-module self-checks (each module's `if __name__ == "__main__"` block is
its own smoke test, not a pytest suite):

```bash
uv run python -m shooter.parse <path-to-one.event.json>      # validates shot table join invariants
uv run python -m shooter.features <path-to-one.event.json>   # validates feature ranges/non-NaN
```

PDF report (no LaTeX/playwright assumed; uses HTML + headless Chromium instead):

```bash
uv run jupyter nbconvert --to html --no-input --HTMLExporter.mathjax_url="" \
  analysis.ipynb --output analysis_report.html
google-chrome --headless --disable-gpu --no-sandbox \
  --print-to-pdf="$(pwd)/analysis_report.pdf" --print-to-pdf-no-header \
  "file://$(pwd)/analysis_report.html"
```

Raw event data lives in `20260601072221_238_2566_11639.event/` (gitignored,
shared via the assignment brief's Drive link, not in this repo).

## Architecture

Pipeline: `parse.py` (NDJSON -> shot table) -> `features.py` (shot table ->
feature columns) -> `build_features.py` (runs both over every file in the
data dir -> `features.csv`) -> `analysis.ipynb` / `dashboard.py` (consume
`features.csv`).

### Data format (`.event.json` files)

Filename encodes identity: `Timestamp_UserId_SessionId_ScenarioID.event.json`
(parsed by `parse_filename`). Each file is NDJSON: one `{Time, Type, Data}`
object per line. `Type` is one of:

- **Frame types** (`controller`, `headset`, `eye_data`): high-frequency
  per-tick sensor data (trigger pull value; gaze origin/direction/pupil per
  eye, gated by `valid.<side>`/`open.<side>`).
- **Event types**: `session_metadata` (dominant hand/eye, device, gun
  geometry), `scenario_start/end`, `location_metadata`, `drill_state_change`,
  `repetition_start/end`, `buzzer` (go signal), `shooting_requirement_change`
  (`add`/`clear` of a live target path + bullets required),
  `target_state_change`, `shot_start`, `shot_resolved` (hit location, object
  hit, engine's own `object_type`/`score`).

Records across types are interleaved in the file and **not** causally
ordered (a `shot_resolved` line can appear before its `shot_start` line) --
always re-sort by `Time` first, which `load_raw` does.

Target path format: `LINE_<row>/<col>/<HitZone>/<ScoreZone>`. A
`shooting_requirement_change` target with `ScoreZone == "*"` is a wildcard on
that segment only; matching logic is `parse._target_match`.

### Known data quirks the code works around

- **Same-millisecond tie**: a `shooting_requirement_change(clear)` can share
  the exact `Time` as the `shot_start` it was triggered by. `load_raw` sorts
  `shot_start` before `shooting_requirement_change` on ties so the
  live-target snapshot at shot time isn't already cleared. Validated:
  recomputed valid-hit flag agrees with the engine's own `object_type` on
  100% of shots across all 8 sample files.
- **IDs reset per repetition**: `shot_start`/`shot_resolved` ids reset to 0
  at the start of each repetition; `repetition_start.id` is always 0 in the
  sample data (so repetitions are tracked by arrival order, not by id).
  `build_shot_table` resets its `open_shots` dict on every `repetition_start`.
- **Per-user eye-tracking fallback**: one sample user's stated
  `dominant_eye` has systematically poor tracking validity; gaze/Quiet Eye
  features fall back to the other eye for that shot if the dominant eye
  doesn't clear `MIN_GAZE_FRAMES` (`features._select_eye`). `gaze_eye_used`/
  `quiet_eye_eye_used` record which eye actually backed each shot.
- **Cohort scope**: 7 of 8 sample files are the same `TargetAcquisition`
  scenario (cross-user norms); the 8th (`WarmUp`, user 243, scenario 11601)
  is a different drill and is excluded via `build_features.EXCLUDED_SCENARIOS`.

### Feature domains (`features.py`)

One function per domain, each taking the shot table (+ frame dict where
relevant) and returning the shot table with new columns:

- `add_reaction_time` -> `reaction_time_s` (timing): buzzer start -> shot_start.
- `add_precision` -> `radial_error_m` (motor): hit vs target center, x/y only.
- `add_shot_quality` -> `shot_quality_pct`: Gaussian-decay transform of
  `radial_error_m`, fit so it tracks the engine's discrete score; a
  derived/display feature, not one of the headline five.
- `add_trigger_control` -> `trigger_smoothness` (+ `trigger_press_duration_s`):
  shape of the final trigger pull before the shot, windowed back to
  `rep_start_time` so it never bleeds into a previous shot.
- `add_gaze_and_pupil` -> `gaze_target_angle_deg` (+ `pupil_diameter_mm`):
  mean aim-misalignment angle over the last `GAZE_WINDOW_MS`.
- `add_quiet_eye` -> `quiet_eye_duration_s`: length of the final continuous
  on-target fixation right up to the shot (Vickers 1996 Quiet Eye paradigm);
  `0.0` means gaze was already off-target at the shot (a real result), `NaN`
  means too few valid gaze frames to tell.
- `decision_quality_by_user`: per-user rate of shots at a non-live target,
  derived from `valid_hit_recomputed`, not one of the per-shot columns above.

`MAIN_FEATURES` (5, one per domain) vs `SUPPLEMENTARY_FEATURES` (2, validated
but not promoted) are listed explicitly at the top of `features.py` --
update both lists together if a feature's status changes, and explain why in
`analysis.ipynb` Section 2, not just here.

All windowed features look back from `shot_start_time`, clipped to
`rep_start_time` (never bleed across repetitions). `_window`/`frames_between`
implement the clipped slice.

### Style note (already in place, keep it)

This codebase deliberately avoids class hierarchies (`Feature`/`Extractor`/
`Plot` classes, event classes) in favor of plain functions over DataFrames --
see the `ponytail:` comments at the top of each module for the reasoning.
Don't introduce abstractions/config objects for values that don't vary at
runtime; match the existing style when extending the pipeline.
