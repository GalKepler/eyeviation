# Data & Pipeline

## Pipeline

```
parse.py            NDJSON -> shot table (one row per resolved shot)
   |
features.py         shot table -> feature columns (one function per domain)
   |
build_features.py   runs both over every file in the data dir -> features.csv
   |
analysis.ipynb / dashboard.py   consume features.csv
```

`shooter/dashboard.py` and `shooter/reporting.py` sit alongside `features.py`:
dashboard holds the plotting/scoring helpers the notebook calls; reporting
holds exploratory metrics (drift, split time, session trend, postural sway)
that are validated but not promoted into `features.csv`. See
[Code Reference](code-reference.md) for every function in each module.

## Data format (`.event.json` files)

The filename encodes identity: `Timestamp_UserId_SessionId_ScenarioID.event.json`
(parsed by `parse_filename`). Each file is NDJSON — one `{Time, Type, Data}`
object per line. `Type` is one of:

- **Frame types** (`controller`, `headset`, `eye_data`): high-frequency
  per-tick sensor data — trigger pull value; gaze origin/direction/pupil per
  eye, gated by `valid.<side>`/`open.<side>`.
- **Event types**: `session_metadata` (dominant hand/eye, device, gun
  geometry), `scenario_start/end`, `location_metadata`, `drill_state_change`,
  `repetition_start/end`, `buzzer` (go signal), `shooting_requirement_change`
  (add/clear of a live target path + bullets required), `target_state_change`,
  `shot_start`, `shot_resolved` (hit location, object hit, the engine's own
  `object_type`/`score`).

Records across types are interleaved in the file and **not** causally
ordered — a `shot_resolved` line can appear before its `shot_start` line.
Always re-sort by `Time` first, which `load_raw` does.

Target path format: `LINE_<row>/<col>/<HitZone>/<ScoreZone>`. A
`shooting_requirement_change` target with `ScoreZone == "*"` is a wildcard on
that segment only — matching logic is `parse._target_match`.

## Known data quirks the code works around

- **Same-millisecond tie.** A `shooting_requirement_change(clear)` can share
  the exact `Time` as the `shot_start` it was triggered by. `load_raw` sorts
  `shot_start` before `shooting_requirement_change` on ties so the
  live-target snapshot at shot time isn't already cleared. Validated:
  recomputed valid-hit flag agrees with the engine's own `object_type` on
  100% of shots across all 8 sample files.
- **IDs reset per repetition.** `shot_start`/`shot_resolved` ids reset to 0
  at the start of each repetition; `repetition_start.id` is always 0 in the
  sample data, so repetitions are tracked by arrival order, not by id.
  `build_shot_table` resets its `open_shots` dict on every `repetition_start`.
- **Per-user eye-tracking fallback.** One sample user's stated `dominant_eye`
  has systematically poor tracking validity; gaze/Quiet Eye features fall
  back to the other eye for that shot if the dominant eye doesn't clear
  `MIN_GAZE_FRAMES` (`features._select_eye`). `gaze_eye_used`/
  `quiet_eye_eye_used` record which eye actually backed each shot.
- **Cohort scope.** 7 of 8 sample files are the same `TargetAcquisition`
  scenario (cross-user norms); the 8th (`WarmUp`, user 243, scenario 11601)
  is a different drill and is excluded via `build_features.EXCLUDED_SCENARIOS`
  — it's used instead in the notebook's within-session drift section, where
  its 72 repetitions answer a question the 7-user cohort can't.
- **Non-`LINE_` object hits.** A handful of shots resolve against scene
  props (`PrpSandBagsWallA1_LOD2.11`, `WallBack_Collider`) or a second
  target-naming scheme (`B_LINE_...` in the `WarmUp` file). `_target_match`
  never hardcodes `"LINE_"`; it only compares segments, so both cases are
  handled without special-casing.
- **A duplicate repetition id** appears once in the `WarmUp` file's
  `SuppressiveFire` drill — read as a retried repetition, not a parser bug.
  Arrival-order `rep_idx` is unaffected since it never relies on id uniqueness.
- **`eye_data.convergence.distance` is flat at `100.0`** across every frame
  sampled — likely unimplemented for this build/scenario rather than a real
  signal; not used in any feature.

## Feature domains (`features.py`)

One function per domain, each taking the shot table (+ frame dict where
relevant) and returning the shot table with new columns:

| Function | Column | Domain |
|---|---|---|
| `add_reaction_time` | `reaction_time_s` | Timing |
| `add_precision` | `radial_error_m` | Motor control |
| `add_shot_quality` | `shot_quality_pct` | Consistency / display (derived from `radial_error_m`) |
| `add_trigger_control` | `trigger_smoothness` (+ `trigger_press_duration_s`) | Trigger control |
| `add_gaze_and_pupil` | `gaze_target_angle_deg` (+ `pupil_diameter_mm`) | Eye behavior |
| `add_quiet_eye` | `quiet_eye_duration_s` | Cognitive (Quiet Eye, Vickers 1996) |
| `decision_quality_by_user` | — | Cognitive, derived per-user, not a per-shot column |

`MAIN_FEATURES` (the 5 headline columns, one per domain) vs
`SUPPLEMENTARY_FEATURES` (2 validated but not promoted) are listed explicitly
at the top of `shooter/features.py`. The full reasoning for every feature —
why it was chosen, how it validates, why two were set aside — is in
[Full Analysis](analysis.ipynb), Section 2.

All windowed features look back from `shot_start_time`, clipped to
`rep_start_time` so they never bleed across repetitions
(`features._window` / `parse.frames_between`).

## Commands

```bash
uv sync
uv run python -m shooter.build_features                 # rebuild features.csv
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb

# per-module self-checks
uv run python -m shooter.parse <path-to-one.event.json>
uv run python -m shooter.features <path-to-one.event.json>
uv run python -m shooter.reporting
```
