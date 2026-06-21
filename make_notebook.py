"""One-off generator for analysis.ipynb. Not a deliverable itself -- run once,
then edit analysis.ipynb directly if changes are needed (or rerun this after
editing this script). Kept out of the repo's "clean" surface conceptually,
but left in place so the notebook's structure is reproducible from source.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# Shooter Performance Feature Engineering - VR Shooting Simulation

This notebook turns raw VR shooting logs (controller/headset/eye-tracking
frames plus drill/repetition/shot events) into five interpretable
shot-quality features, one per performance domain, plus a derived sixth
(decision quality). Two more metrics were built and validated along the way
but didn't get promoted to the main set; Section 2 covers why.

It then presents those features twice: as a plain-language dashboard for the
shooter, and as a PM-facing writeup of why each one was chosen and how it
could map to product feedback. See `CLAUDE.md` for the data dictionary and
`README.md` for how to run this.

**Cohort:** 7 users, each running the same `TargetAcquisition` scenario
(about 31 shots and 24 repetitions per session, roughly 6 minutes). An 8th
file (user 243, `WarmUp` scenario, 350 shots) is a different drill and is
left out of the cross-user comparisons - see Section 4.
""")

# ---------------------------------------------------------------------------
md("## 0. Data loading & EDA")
code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from shooter.parse import parse_file
from shooter.build_features import build, DATA_DIR, EXCLUDED_SCENARIOS
from shooter.features import MAIN_FEATURES, SUPPLEMENTARY_FEATURES
from shooter.dashboard import skill_scores, plot_scorecard, plot_good_vs_bad, plot_hit_map, plot_feature_vs_score

pd.set_option("display.width", 140)
plt.rcParams["figure.figsize"] = (7, 4)
""")

md("""
**Filenames** follow `Timestamp_UserId_SessionId_ScenarioID.event.json`.
Each file is NDJSON (one JSON object per line), so we read it by streaming
`json.loads` per line instead of `pandas.read_json` - files run 20-75MB, and
loading the whole thing into memory at once buys nothing for a
line-oriented format.

First, a look at one file's raw structure: event-type volumes and frame
rate.
""")
code("""
sample_path = sorted(DATA_DIR.glob("*.event.json"))[3]  # the 238 file used to explore the data
frames, shots_sample = parse_file(sample_path)
print(sample_path.name)
for t, df in sorted(frames.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(df):>7}  {t}")
""")

code("""
# Frame rate: gaps between consecutive controller frames (ms). Frame data is
# logged at a steady rate -- this just confirms it and sets expectations for
# window sizes used later (TRIGGER_WINDOW_MS, GAZE_WINDOW_MS in features.py).
gaps = frames["controller"]["Time"].diff().dropna()
print(f"median frame gap: {gaps.median():.1f} ms  (~{1000/gaps.median():.0f} Hz)")
gaps.clip(upper=50).hist(bins=30)
plt.xlabel("ms between controller frames"); plt.ylabel("count"); plt.title("Frame rate check");
""")

md("""
**Timestamp note:** `Time` is an integer millisecond epoch, but in this
sample it decodes to a date around 2026 - almost certainly a synthetic or
shifted clock rather than a real recording date. We only ever use
*differences* between `Time` values (reaction time, window slicing), never
the absolute wall-clock value, so this has no effect on any feature.

**Event ordering:** events of different `Type` are interleaved in the file
and aren't guaranteed to be causally ordered even after sorting by `Time`.
We found a concrete case of this: a `shooting_requirement_change` "clear"
can land at the exact same millisecond as the `shot_start` that triggered
it, and a plain stable sort leaves "clear" first - making it look like the
requirement was already gone before the shot that satisfied it.
`shooter/parse.py` breaks ties at equal timestamps by processing
`shot_start` before `shooting_requirement_change`. After that fix, our
recomputed valid-hit flag (`object_id` matched against whichever
requirement was live at shot time) agrees with the engine's own
`object_type` label on **100% of shots** across all 8 files - good evidence
that the event model (drill -> repetition -> buzzer -> requirement -> shot)
was reconstructed correctly.
""")
code("""
frames, shots_sample = parse_file(sample_path)  # rebuilt with the tie-break fix already in parse.py
mismatch = (shots_sample["valid_hit_recomputed"] != (shots_sample["object_type"] == "valid")).mean()
print(f"disagreement between recomputed valid-hit and engine object_type: {mismatch:.0%}")
""")

md("""
**Eye-tracking data quality:** `valid.<eye>` / `open.<eye>` record per-frame
tracking confidence. One user (243) has consistently lower left-eye
validity (about 75-78%) than right-eye (about 90-96%) across both of their
sessions, despite `session_metadata` listing `dominant_eye = "Left"` - that
pattern looks like a per-user calibration or sensor issue rather than
ordinary blinking. `add_gaze_and_pupil` in `shooter/features.py` handles
this by preferring the dominant eye but falling back to the other eye, per
shot, whenever the dominant eye doesn't clear a minimum valid-frame count in
its window, instead of just returning NaN for that user's gaze features.
""")
code("""
rows = []
for path in sorted(DATA_DIR.glob("*.event.json")):
    f, _ = parse_file(path)
    eye = f["eye_data"]
    uid = f["session_metadata"]["user_id"].iloc[0]
    dom = f["session_metadata"]["dominant_eye"].iloc[0]
    rows.append({"user_id": uid, "dominant_eye": dom,
                 "left_valid_rate": eye["valid.left"].mean(), "right_valid_rate": eye["valid.right"].mean()})
pd.DataFrame(rows).round(2)
""")

# ---------------------------------------------------------------------------
md("""
## Building the per-shot feature table

The core join (`shooter/parse.py: build_shot_table`) replays every event in
`Time` order, keeping drill, repetition, buzzer, and "live" shooting
requirements as running state, and emits one row per `shot_resolved` joined
to its matching `shot_start`, repetition, and active requirement.
`shooter/features.py` then adds the feature columns on top of that table.
""")
code("""
shots = build(DATA_DIR)
print(f"{len(shots)} shots from {shots['user_id'].nunique()} users "
      f"(scenario {sorted(shots['scenario_id'].unique())}; excluded scenarios: {EXCLUDED_SCENARIOS})")
shots[["user_id", "rep_idx", "reaction_time_s", "radial_error_m", "trigger_press_duration_s",
       "trigger_smoothness", "gaze_target_angle_deg", "quiet_eye_duration_s",
       "valid_hit_recomputed", "score"]].head()
""")

# ---------------------------------------------------------------------------
md("""
## 1. Shooter Dashboard

A plain-language summary for one shooter (user 238, chosen because their
profile mixes clear strengths and clear weaknesses, which makes for a more
useful example than someone who's simply good or bad across the board).
Scores are **0-100 percentiles against the other 6 users in this cohort**:
100 is best in the group, 50 is the middle. With only 7 users this is a
coarse ranking, not a calibrated population norm (see Section 4).
""")
code("""
DEMO_USER = 238
raw, scores = skill_scores(shots)
raw.round(2)
""")
code("""
scores.round(0)
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_scorecard(scores, DEMO_USER, axes[0])
plot_hit_map(shots, DEMO_USER, axes[1])
fig.tight_layout()
""")
md("""
**Plain-language feedback for user 238:** Your trigger finger is fast. You're
the quickest in the group to fire once you're allowed to shoot (top score on
speed). But that speed isn't paying off in steadiness: your trigger pull is
the jerkiest in the group (lowest stability score), and your eyes drift the
furthest from the target in the moment before you shoot (low focus score).
Accuracy and decision-quality both sit below the group median. **Bottom
line: you're shooting fast, but slow down on the final trigger squeeze and
keep your eyes locked on the target through the shot - right now speed is
costing you precision.**
""")

# ---------------------------------------------------------------------------
md("""
## 2. Feature Explanation

The brief asks for 3-5 features, each from a different performance domain.
Building the pipeline, we ended up with seven candidate metrics across five
domains - more than one per domain in a couple of cases. Below is the **main
set of five**, one per domain, followed by a derived sixth (decision
quality) and then the two extra metrics we tested but didn't promote, along
with the reasoning for leaving each one out.

### Main features

| # | Feature | Domain | What it measures | Data source |
|---|---------|--------|-------------------|--------------|
| 1 | `reaction_time_s` | Timing / target acquisition | Seconds from `buzzer(start)` to `shot_start` | `buzzer`, `shot_start` |
| 2 | `radial_error_m` | Motor control / consistency | Distance from hit point to target center (x,y plane) | `shot_resolved.hit_local_location`/`center_local_location` |
| 3 | `trigger_smoothness` | Trigger control | Jaggedness of the final trigger pull before the shot | `controller.trigger_pull_value` |
| 4 | `gaze_target_angle_deg` | Eye behavior | Angle between gaze direction and target, averaged over the 500ms before the shot | `eye_data` |
| 5 | `quiet_eye_duration_s` | Cognitive | Duration of the final continuous on-target fixation right up to the trigger pull ("Quiet Eye") | `eye_data` |

#### 1. Reaction time (timing / target acquisition)
Time from the buzzer's "start" signal to the shot's trigger break - the most
direct measure we have of how quickly a shooter engages a presented target,
which is a tactical skill in its own right.

One assumption worth flagging: in multi-bullet repetitions, the second
bullet's "reaction time" includes whatever time was already spent on the
first bullet, so it reads as *time-to-fire-this-bullet-since-allowed* rather
than a clean reflex measure. We're calling that out rather than treating the
two cases as equivalent.

For product, this maps directly to a "draw speed" or "time-to-first-shot"
metric for the shooter, and could back a pass/fail threshold in
time-limited scenarios.

#### 2. Precision and grouping (motor control / consistency)
Euclidean distance between the hit point and the target's center, in the
plane perpendicular to the line of fire (x = lateral, y = vertical; we drop
z, the depth into the target, since the engine already snaps it to the
target plane and it carries no aim-quality signal here).

This is about as basic a shot-quality measure as exists: even a "valid" hit
can land far from center, which matters for follow-up shot speed and
lethality in a realistic scenario. Looking at `radial_error_m` across
repeated shots at the same target row also gives a grouping/consistency
read, not just a single-shot number.

For product, this maps straight onto the "shot grouping" visualizations
shooters already expect from range training.

#### 3. Trigger control
From the `trigger_pull_value` stream in the second before the shot, we take
`trigger_smoothness`: the standard deviation of frame-to-frame change during
the final press (lower = smoother). We're assuming the shot fires at or near
the local max of the pull within that window.

A jerky or "slapped" trigger pull is a well-known cause of low-and-left
misses in real shooting, and it's coachable independently of aim. A
below-peers smoothness score is a concrete, drillable cue ("work on
dry-fire trigger control") in a way that "shoot more accurately" isn't.

#### 4. Gaze and target alignment (eye behavior)
Angle between the dominant eye's `world_gaze_direction` and the vector to
the target center, averaged over the 500ms before the shot. This assumes
gaze and target-center coordinates share the same world frame, which we
checked by cross-referencing Section 0's hit-location coordinates against
`location_metadata.target_base`.

"Look where you want to shoot" is foundational marksmanship coaching, and a
large gaze-target angle right before the shot turns out to be a leading
indicator of a miss (Section 3). Of everything in this report, this is the
strongest candidate for a real-time in-VR cue ("eyes on target"), since it
precedes the shot rather than just describing it afterward.

#### 5. Quiet Eye duration (cognitive)
How long, continuously, gaze stayed within 3° of the target right up to the
trigger pull. We walk backward frame-by-frame from `shot_start` until gaze
leaves that tolerance or a frame gap signals a blink or tracking dropout.

This is the "Quiet Eye" measure from sports-vision research (Vickers, 1996):
the final fixation before a critical motor action, replicated specifically
in marksmanship and police-shooting studies (Causer, Bennett & Williams,
2010), where a longer pre-shot Quiet Eye duration predicts a higher hit
rate.

It's tempting to assume this just duplicates `gaze_target_angle_deg`, but
the two ask different questions. The angle feature averages over a fixed
500ms window, so a shooter whose gaze wanders onto target for most of that
window but breaks away right before firing still scores well on it. Quiet
Eye asks specifically how long gaze was locked on, uninterrupted, through
the moment of the shot - catching the failure mode the average can hide. A
`0.0s` value means gaze was already off-target at the shot and is a real
result, not missing data; `NaN` is reserved for windows with too few valid
gaze frames to judge.

The 3° tolerance and the trigger-pull instant as the Quiet Eye "offset" are
both literature-standard choices rather than something measured from this
dataset - a tighter or looser tolerance would shift the absolute durations
but not the cohort ranking (checked in Section 3). For product, this pairs
with the gaze-angle cue as a single, well-validated number that's easy to
coach directly - "hold your sight picture on target for at least half a
second before you break the shot" is a known drill in real-range Quiet Eye
training.

### Bonus: decision quality (cognitive, derived)
Beyond the five main features, we also compute the share of a user's shots
fired at a target that wasn't currently a "live" shooting requirement (for
example, hitting a no-score or off-target object while the intended target
was active). It's derived from the same event data as the others rather
than measured directly off a sensor, which is why we're treating it as a
bonus rather than one of the five.

It catches a failure mode none of the other features can see: "couldn't hit
the right target" (a motor/aim problem) is different from "shot the wrong
target" (a target-discrimination or impulse-control problem), and those
call for different coaching entirely. A shooter with a high non-valid rate
but otherwise good accuracy and reaction time probably needs discrimination
drills, not marksmanship ones.

### Supplementary features (tested, not in the main set)
Two more per-shot metrics came out of the same pipeline. Both are real,
validated columns in `features.csv` - they just didn't make the headline
five, for reasons specific to each:

- **`trigger_press_duration_s`** (time from a half-way pull threshold to the
  shot) sits in the same domain as `trigger_smoothness` and is correlated
  with it. Smoothness is the more diagnostic and more obviously coachable of
  the two, so it became the main trigger-control feature, and duration is
  kept as a supporting number rather than giving trigger control two
  headline slots.
- **`pupil_diameter_mm`** (mean pupil size in the pre-shot gaze window, as a
  cognitive-load/arousal proxy) is conceptually interesting, but unlike
  every main feature it never showed a clean separation between valid hits
  and misses or a correlation with the engine's `score` field (Section 3
  only runs that check on the main five). Without that evidence we're not
  comfortable presenting it as a validated shot-quality signal yet - flagged
  here as worth a second look once there's a clearer hypothesis for what
  pupil size should predict in this task. (`gaze_stability_deg`, the
  per-shot standard deviation of the gaze angle, is in the same boat: a
  reasonable idea, computed alongside the others, but not separately
  validated or promoted.)
""")

# ---------------------------------------------------------------------------
md("""
## 3. Feature Validation / Supporting Analysis

The core trust check for every feature: does it actually separate good shots
from bad ones?
""")
code("""
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_good_vs_bad(shots, "radial_error_m", axes[0, 0], "radial error (m)")
plot_good_vs_bad(shots, "gaze_target_angle_deg", axes[0, 1], "gaze-target angle (deg)")
plot_good_vs_bad(shots, "quiet_eye_duration_s", axes[0, 2], "Quiet Eye duration (s)")
plot_good_vs_bad(shots, "trigger_smoothness", axes[1, 0], "trigger smoothness (std)")
plot_good_vs_bad(shots, "reaction_time_s", axes[1, 1], "reaction time (s)")
fig.delaxes(axes[1, 2])
fig.suptitle("Feature distributions: valid hit vs non-valid shot")
fig.tight_layout()
""")
md("""
Radial error and gaze-target angle both show a clear, large separation
(misses have much larger radial error and much larger gaze-target angle):
strong validation for both. Quiet Eye duration also separates the groups in
the expected direction (valid hits have a longer final on-target fixation),
though with more overlap than the angle feature, consistent with it
capturing a related but distinct, noisier signal (a single critical instant
versus an averaged window). Trigger smoothness and reaction time show weaker
or no separation by hit/miss, which is expected: a shooter can have a clean,
smooth, fast trigger pull and still miss for purely aim-related reasons.
These two are technique measures, not outcome measures, and get validated
differently below (against each other and against between-user variation,
not against the binary hit/miss label).
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
plot_feature_vs_score(shots, "radial_error_m", axes[0], "radial error (m)")
plot_feature_vs_score(shots, "gaze_target_angle_deg", axes[1], "gaze-target angle (deg)")
plot_feature_vs_score(shots, "quiet_eye_duration_s", axes[2], "Quiet Eye duration (s)")
fig.tight_layout()
""")
md("""
All three correlate with the engine's own `score` field in the expected
direction (more error or a wider gaze miss means a lower score; a longer
Quiet Eye means a higher score), without us ever telling the pipeline what
`score` was. That's independent validation that these features track real
shot quality rather than an artifact of how they were built.
""")
code("""
# Redundancy check: are the domains actually telling us different things, or
# are they all just proxies for the same underlying "good session" effect?
# Covers both the main features and the two supplementary ones, so we can
# see whether the metrics we set aside are at least adding distinct
# information, even though they didn't clear the bar above to be promoted.
feature_cols = MAIN_FEATURES + SUPPLEMENTARY_FEATURES
shots[feature_cols].corr(method="spearman").round(2)
""")
md("""
Most correlations are modest (|ρ| mostly under 0.3), which suggests each
feature is contributing something distinct rather than all of them
re-measuring "had a good day." That's what the brief is after - different
performance domains, not several views of the same underlying thing - and
it holds for the two supplementary features too, not just the main five.

The one clear exception is `quiet_eye_duration_s` vs `gaze_target_angle_deg`
(ρ ≈ -0.84 per shot). Both measure gaze-target alignment from overlapping
data, so a strong per-shot relationship isn't surprising. What actually
matters is whether they *rank shooters* the same way - that's checked
directly below.
""")
code("""
# Same question, but at the level a coach actually cares about: does ranking
# shooters by focus agree with ranking them by Quiet Eye?
rank_corr = scores[["focus", "quiet_eye"]].corr(method="spearman").iloc[0, 1]
print(f"per-user skill-score rank correlation (focus vs quiet_eye): {rank_corr:.2f}")
scores[["focus", "quiet_eye"]].round(0)
""")
md("""
At ρ ≈ 0.5, the per-user rankings agree only moderately, well below the
per-shot correlation. User 240 is the clearest case: worst in the cohort on
average gaze-target angle, but mid-pack on Quiet Eye duration. That's a
shooter whose gaze wanders during the approach but snaps onto target right
at the moment of the shot. A coaching report built on `gaze_target_angle_deg`
alone would flag this person as "not looking at the target," but Quiet Eye
duration tells a more precise story: eyes are on target when it counts,
just not before that. That's the complementary information that justifies
keeping both as main features instead of picking one.
""")
code("""
# Case study: best and worst single shots by radial error, to see the features
# explaining concrete behavior rather than just summary statistics.
best = shots.loc[shots["radial_error_m"].idxmin()]
worst = shots.loc[shots["radial_error_m"].idxmax()]
pd.DataFrame([best, worst], index=["best shot", "worst shot"])[
    ["user_id", "object_id", "score", "radial_error_m", "reaction_time_s",
     "trigger_smoothness", "gaze_target_angle_deg", "valid_hit_recomputed"]]
""")
md("""
The worst shot by radial error also has a far larger gaze-target angle than
the best one. That's a single concrete case of the gaze feature explaining a
miss that the precision number alone can't diagnose: was this a motor
problem, or did the shooter just look at the wrong place?
""")
code("""
# Cross-user norms: cohort-wide view of the 0-100 percentile scores.
fig, ax = plt.subplots(figsize=(8, 4.5))
scores.plot.bar(ax=ax)
ax.set_ylabel("score (0-100, percentile vs cohort)")
ax.set_title("Skill scores across the 7-user cohort")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
""")

# ---------------------------------------------------------------------------
md("""
## 4. Features Considered But Not Built

Beyond the two supplementary features in Section 2 (computed, just not
promoted), a couple of other ideas didn't make it into the pipeline at all:

- **Headset/controller path stability before the shot** (hand/head sway).
  The frame data would support this, but conceptually it overlaps heavily
  with `trigger_smoothness`: both are really asking "how steady was the
  shooter right before the shot?" With only seven users, adding a third
  highly-correlated steadiness metric felt more likely to dilute
  interpretability than add signal. Worth revisiting if the cohort grows.
- **Convergence distance** (`eye_data.convergence`). It's in the schema, but
  was flat at 100.0 across the sample we explored, which suggests it isn't
  actively computed in this build or scenario - not reliable enough to build
  a feature on yet.

**Valuable, but needs more data than this sample has:**
- *Real per-user norms.* The 0-100 scores in this report are percentiles
  within a group of 7 people on one scenario - useful for comparing this
  specific cohort, but not a stable population baseline. Turning "70th
  percentile" into something that means the same thing for a new user next
  month would need norms built from dozens to hundreds of sessions per skill
  level.
- *Learning-curve features* - does reaction time drop across repetitions
  within a session, or across sessions for the same user? The sample is one
  session per user (243 has two), which isn't enough repeated sessions to
  fit a trend.
- *Recoil/follow-through and stance/posture features* would need sensor
  logs this schema doesn't have (full-body tracking, weapon recoil
  telemetry).
- *Audio* (verbal commands, ambient distraction) isn't referenced anywhere
  in the schema and would need an entirely new log type.

**Logging changes that would help:** an explicit `shot_break_time` captured
directly off the trigger sensor (instead of inferring it from the
`trigger_pull_value` curve) would make the trigger-control features exact
rather than approximated. A per-frame `gaze_confidence` score, instead of
the binary `valid`/`open` flags, would also remove the need to infer the
eye-tracking fallback in Section 0 from raw validity rates.
""")

# ---------------------------------------------------------------------------
code("""
print(f"reaction time range across cohort: {raw['speed'].min():.1f}s - {raw['speed'].max():.1f}s")
print(f"gaze-target angle range across cohort: {raw['focus'].min():.1f}deg - {raw['focus'].max():.1f}deg "
      f"({raw['focus'].max() / raw['focus'].min():.1f}x spread)")
print(f"Quiet Eye duration range across cohort: {raw['quiet_eye'].min():.2f}s - {raw['quiet_eye'].max():.2f}s "
      f"({raw['quiet_eye'].max() / max(raw['quiet_eye'].min(), 0.01):.0f}x spread)")
""")
md("""
## 5. Product Manager Summary

**Scope:** five features, one per domain (timing, motor control, trigger
control, eye behavior, cognitive), plus a derived decision-quality metric.
Two more metrics were built and validated alongside these but set aside as
supplementary - see Section 2 for the reasoning on each.

**What we learned about this cohort:** reaction time is fairly uniform
across the 7 shooters (roughly 2.0-2.5s); everyone engages at about the same
speed once the buzzer goes. The real spread is in eye behavior. Gaze-target
angle varies by roughly 6x between the most and least target-focused
shooter, and Quiet Eye duration spreads even wider (see the numbers above).
Eye behavior is the single most differentiating skill in this cohort, and
it's also the domain that's easiest to turn into a real-time in-VR coaching
cue: "look at the target," "hold your sight picture before you fire."

**Strong and weak skills:** decision quality (shooting the right target) and
eye behavior (focus and Quiet Eye) separate the cohort the most; reaction
time separates it the least, with trigger control and accuracy in between.
Focus and Quiet Eye only agree moderately on how they rank shooters (ρ ≈ 0.5,
Section 3), so it's worth surfacing both rather than treating one as "the"
eye metric - each catches a shooter the other would miss. One user, for
example, ranks worst on average gaze alignment but mid-pack on Quiet Eye:
eyes wander during the approach, but lock on right when it counts.

**Most reliable features:** `radial_error_m` and `gaze_target_angle_deg` are
the two we'd trust most. Both show a large, sensible gap between valid hits
and misses (Section 3), and both correlate with the engine's own shot score
without ever having been told what that score was. `quiet_eye_duration_s`
shows the same pattern, just more weakly, since it's a single-instant
measure and noisier per shot - but it adds a distinct, literature-backed
angle the others don't cover. `trigger_smoothness` and `reaction_time_s` are
technique measures rather than outcome measures, so they validate
differently (by between-user spread, not hit/miss separation): trust them as
*process* feedback, not as a stand-in for "did they hit."

**What would improve logging or product feedback:** capture an explicit
trigger-break event instead of inferring it from the raw pull curve; log
per-frame eye-tracking confidence instead of a binary valid/open flag, so
gaze features don't need an inferred per-user eye fallback; and collect
multiple sessions per user before showing anyone a percentile score, so
"better than 70% of users" reflects a real population rather than these
7 people.
""")

nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("wrote analysis.ipynb with", len(cells), "cells")
