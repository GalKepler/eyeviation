"""Exploratory per-user reporting metrics, on top of the headline features in
features.py. Nothing here is promoted to MAIN_FEATURES or features.csv -- see
analysis_explorations.ipynb for how these get used.

Axis convention (validated against location_metadata.target_base: world x
rises monotonically with column index; gun/headset position.y sits at
gun/eye height): +x = shooter's right, +y = up. Matches what plot_hit_map
already assumes.

ponytail: bare functions over DataFrames, same style as dashboard.py/
features.py -- no Report/Metric classes for a fixed, small set of one-off
aggregates.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from shooter.parse import frames_between, parse_file, parse_filename

# 8-way compass label for a (x, y) offset, walked counter-clockwise from
# "right" (angle 0) in 45-degree steps to match atan2's convention.
_DIRECTIONS = ["right", "upper-right", "up", "upper-left", "left", "lower-left", "down", "lower-right"]


def _compass(x, y):
    if x == 0 and y == 0:
        return "centered"
    angle = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return _DIRECTIONS[int((angle + 22.5) // 45) % 8]


def drift_summary(shots):
    """Per-user mean miss vector (bias) vs std of radial_error_m (spread).

    Same shots, two different prescriptions: a consistent offset in one
    direction (bias) is a sight/grip/stance fix; spread with ~zero bias is a
    consistency problem that only practice fixes. Collapsing to radial_error_m
    alone (as the headline feature does) can't tell these apart.
    """
    err = shots.assign(_ex=shots["hit_x"] - shots["center_x"], _ey=shots["hit_y"] - shots["center_y"])
    summary = err.groupby("user_id").agg(bias_x_m=("_ex", "mean"), bias_y_m=("_ey", "mean"),
                                          spread_m=("radial_error_m", "std"))
    summary["bias_mag_m"] = np.hypot(summary["bias_x_m"], summary["bias_y_m"])
    summary["bias_direction"] = [_compass(x, y) for x, y in zip(summary["bias_x_m"], summary["bias_y_m"])]
    return summary


def split_times(shots):
    """Time between consecutive shots within the same repetition (s).

    reaction_time_s for bullet 2+ is buzzer-to-shot, so it includes bullet
    1's time (features.py docstring). split_time_s instead measures
    bullet-to-bullet pace -- the standard practical-shooting cadence metric,
    and unaffected by that caveat. NaN for each rep's first bullet (nothing
    to measure from).
    """
    shots = shots.sort_values("shot_start_time").copy()
    g = shots.groupby(["user_id", "session_id", "rep_idx"])
    shots["bullet_idx_in_rep"] = g.cumcount()
    shots["split_time_s"] = (shots["shot_start_time"] - g["shot_resolved_time"].shift()) / 1000
    return shots


def speed_accuracy_tradeoff(shots):
    """Per-user Spearman rho of reaction_time_s vs radial_error_m: does
    rushing cost this shooter accuracy? Positive rho = yes (faster shots
    land worse); near zero = speed and accuracy are independent for them.
    """
    return shots.groupby("user_id").apply(
        lambda d: d[["reaction_time_s", "radial_error_m"]].corr(method="spearman").iloc[0, 1],
        include_groups=False,
    ).rename("speed_accuracy_rho")


# One trend per performance domain rather than a single composite: reaction
# time often improves with warm-up while precision degrades with fatigue, so
# averaging them into one number would hide the more interesting story.
TREND_DOMAINS = ["reaction_time_s", "radial_error_m", "trigger_smoothness", "gaze_target_angle_deg", "quiet_eye_duration_s"]


def session_trend(shots, min_shots=20):
    """Per-user, per-domain change from the first third to the last third of
    a session, after subtracting each shot's target-row mean (row-2 targets
    score worse than row-1 regardless of skill -- features.py) so the trend
    isn't just "which rows came up late."

    Sign: for reaction_time_s/radial_error_m/trigger_smoothness/
    gaze_target_angle_deg, lower is better, so a negative trend = improved.
    quiet_eye_duration_s is the opposite (higher is better).

    Below min_shots the first/last-third split is too noisy to call --
    returns NaN there rather than a confident-looking number from ~7 shots.
    """
    rows = []
    for (user, session), s in shots.groupby(["user_id", "session_id"]):
        s = s.sort_values("shot_start_time")
        n = len(s)
        rec = {"user_id": user, "session_id": session, "n_shots": n}
        if n < min_shots:
            rec.update({f"{d}_trend": np.nan for d in TREND_DOMAINS})
        else:
            third = n // 3
            for d in TREND_DOMAINS:
                adjusted = s[d] - s.groupby("target_row")[d].transform("mean")
                rec[f"{d}_trend"] = adjusted.iloc[-third:].mean() - adjusted.iloc[:third].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


# Same look-back as the other pre-shot windows (features.GAZE_WINDOW_MS).
SWAY_WINDOW_MS = 500


def postural_sway(data_dir):
    """Pre-shot head-position wobble (m): magnitude of the std of headset
    x/y/z position over the SWAY_WINDOW_MS before each shot.

    headset frames are parsed but otherwise unused by the pipeline -- body
    sway right before the shot is an established marksmanship-quality signal
    (steadier stance -> tighter groups) this taps for the first time. Kept
    out of features.csv since it needs a second pass over raw frame data the
    main build doesn't retain; merge onto the shot table by
    (user_id, session_id, shot_start_time) instead.

    Skips the headset's rotation quaternion (angular head wobble) -- position
    sway alone is the simpler, standard version of this signal; add angular
    wobble only if position sway turns out not to separate hits from misses.
    """
    rows = []
    for path in sorted(Path(data_dir).glob("*.event.json")):
        frames, shots = parse_file(path)
        head = frames.get("headset")
        if head is None or shots.empty:
            continue
        meta = parse_filename(path)
        for _, shot in shots.iterrows():
            t0 = max(shot["rep_start_time"], shot["shot_start_time"] - SWAY_WINDOW_MS)
            win = frames_between(head, t0, shot["shot_start_time"] + 1)
            sway = np.linalg.norm(win[["position.x", "position.y", "position.z"]].std()) if len(win) >= 3 else np.nan
            rows.append({"user_id": meta["user_id"], "session_id": meta["session_id"],
                         "shot_start_time": shot["shot_start_time"], "head_sway_m": sway})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from shooter.build_features import DATA_DIR, build

    shots = build(DATA_DIR)

    drift = drift_summary(shots)
    assert (drift["bias_mag_m"] >= 0).all() and (drift["spread_m"] >= 0).all(), "bias/spread can't be negative"

    st = split_times(shots)
    assert (st["split_time_s"].dropna() >= 0).all(), "split time can't be negative"

    rho = speed_accuracy_tradeoff(shots)
    assert rho.dropna().between(-1, 1).all(), "Spearman rho out of [-1, 1]"

    trend = session_trend(shots)
    assert (trend.loc[trend["n_shots"] < 20, [f"{d}_trend" for d in TREND_DOMAINS]].isna().all(axis=None)), \
        "underpowered sessions should report NaN trend, not a number"

    sway = postural_sway(DATA_DIR)
    assert (sway["head_sway_m"].dropna() >= 0).all(), "sway can't be negative"

    print(f"drift:\n{drift.round(3)}\n")
    print(f"speed/accuracy rho per user:\n{rho.round(2)}\n")
    print(f"session trend (n_shots, then per-domain):\n{trend.round(3)}\n")
    print(f"head sway: {sway['head_sway_m'].notna().sum()}/{len(sway)} shots, "
          f"mean {sway['head_sway_m'].mean():.4f} m")
