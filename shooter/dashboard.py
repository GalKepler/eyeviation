"""Plotting + scoring helpers shared by the shooter dashboard and PM validation
sections of analysis.ipynb. Minimal matplotlib -- the brief explicitly says
not to spend time on visual polish.

ponytail: bare functions over the features DataFrame; no Plot/Report classes,
no config object for things like colors that never change here.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# skill name -> (raw column, higher raw value = better performance). Used to
# pick the percentile-rank direction in skill_scores.
SKILL_METRICS = {
    "accuracy": ("radial_error_m", False),
    "speed": ("reaction_time_s", False),
    "stability": ("trigger_smoothness", False),
    "focus": ("gaze_target_angle_deg", False),
    "quiet_eye": ("quiet_eye_duration_s", True),
    "decision": ("valid_hit_recomputed", True),
}


def skill_scores(shots):
    """Per-user mean of each raw metric, plus a 0-100 cohort-percentile score
    per skill (100 = best in cohort).

    Percentile, not z-score: with only 7 users a rank-based score is more
    robust to outliers and easier to explain to a shooter ("better than N of
    your peers") than a standardized score.
    """
    cols = {name: col for name, (col, _) in SKILL_METRICS.items()}
    raw = shots.groupby("user_id")[list(cols.values())].mean()
    raw.columns = list(cols.keys())

    scores = pd.DataFrame(index=raw.index)
    for name, (_, higher_is_better) in SKILL_METRICS.items():
        pct = raw[name].rank(pct=True)
        scores[name] = 100 * (pct if higher_is_better else (1 - pct))
    return raw, scores


def plot_scorecard(scores, user_id, ax=None):
    """Bar chart of one user's 0-100 skill scores vs the cohort."""
    ax = ax or plt.gca()
    row = scores.loc[user_id]
    ax.bar(row.index, row.values, color="#4C72B0")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="cohort median")
    ax.set_ylim(0, 100)
    ax.set_ylabel("score (0-100, percentile vs cohort)")
    ax.set_title(f"User {user_id}: skill scorecard")
    ax.legend()
    return ax


def plot_good_vs_bad(shots, feature_col, ax=None, label=None):
    """Boxplot of a feature split by whether the shot was a valid hit -- the
    core "why trust this feature" check: it should separate hits from misses.
    """
    ax = ax or plt.gca()
    groups = [shots.loc[shots["valid_hit_recomputed"], feature_col].dropna(),
              shots.loc[~shots["valid_hit_recomputed"], feature_col].dropna()]
    ax.boxplot(groups, tick_labels=["valid hit", "non-valid"])
    ax.set_ylabel(label or feature_col)
    ax.set_title(f"{label or feature_col}: valid hit vs non-valid")
    return ax


def plot_hit_map(shots, user_id, ax=None):
    """Scatter of hit point relative to target center (meters), colored by
    valid hit -- shows both precision (spread) and bias (off-center direction).
    """
    ax = ax or plt.gca()
    u = shots[shots["user_id"] == user_id]
    colors = np.where(u["valid_hit_recomputed"], "#2ca02c", "#d62728")
    ax.scatter(u["hit_x"] - u["center_x"], u["hit_y"] - u["center_y"], c=colors, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("horizontal miss (m)")
    ax.set_ylabel("vertical miss (m)")
    ax.set_title(f"User {user_id}: hit points relative to target center")
    ax.set_aspect("equal")
    return ax


def plot_feature_vs_score(shots, feature_col, ax=None, label=None):
    """Scatter of a feature against the engine's own shot score, to sanity
    check the feature actually relates to shot quality."""
    ax = ax or plt.gca()
    ax.scatter(shots[feature_col], shots["score"], alpha=0.5)
    ax.set_xlabel(label or feature_col)
    ax.set_ylabel("shot score")
    rho = shots[[feature_col, "score"]].corr(method="spearman").iloc[0, 1]
    ax.set_title(f"{label or feature_col} vs score (Spearman ρ={rho:.2f})")
    return ax
