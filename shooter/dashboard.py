"""Plotting + scoring helpers shared by the shooter dashboard and PM validation
sections of analysis.ipynb.

ponytail: bare functions over the features DataFrame; no Plot/Report classes,
no config object for things like colors that never change here. Seaborn for
the statistical chart types (violin/reg) it gives us for free over hand-rolled
matplotlib; styling is one sns.set_theme() call, not a custom theme system.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

GOOD_COLOR = "#2ca02c"
BAD_COLOR = "#d62728"
ACCENT_COLOR = "#4C72B0"

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
    """Horizontal bar chart of one user's 0-100 skill scores vs the cohort,
    colored red-to-green by score so weak/strong spots pop out at a glance.
    """
    ax = ax or plt.gca()
    row = scores.loc[user_id].sort_values()
    colors = sns.color_palette("RdYlGn", as_cmap=True)(row.values / 100)
    bars = ax.barh(row.index, row.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontweight="bold")
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, label="cohort median")
    ax.set_xlim(0, 100)
    ax.set_xlabel("score (0-100, percentile vs cohort)")
    ax.set_title(f"User {user_id}: skill scorecard", fontweight="bold")
    ax.legend(loc="lower right")
    sns.despine(ax=ax, left=True)
    return ax


def plot_user_radar(scores, user_id, ax):
    """Radar chart of one user's 0-100 skill scores (colored) against the
    cohort median (dashed gray), for use as one panel of a small-multiples
    grid -- one ax per user reads easier than every user overlaid on one.
    """
    skills = list(scores.columns)
    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    angles += angles[:1]

    values = scores.loc[user_id].tolist()
    values += values[:1]
    median = scores.median().tolist()
    median += median[:1]

    ax.plot(angles, median, color="gray", linestyle="--", linewidth=1, label="cohort median")
    ax.plot(angles, values, color=ACCENT_COLOR, linewidth=1.5)
    ax.fill(angles, values, color=ACCENT_COLOR, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.set_title(f"user {user_id}", fontweight="bold", fontsize=10)
    return ax


def plot_good_vs_bad(shots, feature_col, ax=None, label=None):
    """Violin + strip plot of a feature split by whether the shot was a valid
    hit -- the core "why trust this feature" check: it should separate hits
    from misses.
    """
    ax = ax or plt.gca()
    df = shots[[feature_col, "valid_hit_recomputed"]].dropna().copy()
    df["outcome"] = df["valid_hit_recomputed"].map({True: "valid hit", False: "non-valid"})
    order = ["valid hit", "non-valid"]
    sns.violinplot(data=df, x="outcome", y=feature_col, order=order, ax=ax,
                    hue="outcome", palette={"valid hit": GOOD_COLOR, "non-valid": BAD_COLOR},
                    legend=False, alpha=0.5, inner=None, cut=0)
    sns.stripplot(data=df, x="outcome", y=feature_col, order=order, ax=ax,
                   color="black", alpha=0.4, size=3, jitter=0.2)
    ax.set_xlabel("")
    ax.set_ylabel(label or feature_col)
    ax.set_title(f"{label or feature_col}", fontweight="bold")
    sns.despine(ax=ax)
    return ax


def plot_hit_map(shots, user_id, ax=None):
    """Scatter of hit point relative to target center (meters), colored by
    valid hit -- shows both precision (spread) and bias (off-center direction).
    """
    ax = ax or plt.gca()
    u = shots[shots["user_id"] == user_id].copy()
    u["outcome"] = u["valid_hit_recomputed"].map({True: "valid hit", False: "non-valid"})
    sns.scatterplot(data=u, x=u["hit_x"] - u["center_x"], y=u["hit_y"] - u["center_y"],
                     hue="outcome", palette={"valid hit": GOOD_COLOR, "non-valid": BAD_COLOR},
                     s=60, alpha=0.8, edgecolor="white", linewidth=0.5, ax=ax)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.scatter([0], [0], marker="+", color="black", s=120, linewidth=1.5, zorder=5)
    ax.set_xlabel("horizontal miss (m)")
    ax.set_ylabel("vertical miss (m)")
    ax.set_title(f"User {user_id}: hits relative to target center", fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(title=None, loc="best")
    sns.despine(ax=ax)
    return ax


def plot_feature_vs_score(shots, feature_col, ax=None, label=None, outcome_col="score"):
    """Scatter + trend line of a feature against a shot-quality outcome (the
    engine's own discrete score by default, or shot_quality_pct), to sanity
    check the feature actually relates to shot quality."""
    ax = ax or plt.gca()
    df = shots[[feature_col, outcome_col]].dropna()
    sns.regplot(data=df, x=feature_col, y=outcome_col, ax=ax, lowess=True,
                scatter_kws=dict(alpha=0.4, s=25, color=ACCENT_COLOR),
                line_kws=dict(color=BAD_COLOR, linewidth=2))
    ax.set_xlabel(label or feature_col)
    ax.set_ylabel(outcome_col)
    rho = df.corr(method="spearman").iloc[0, 1]
    ax.set_title(f"{label or feature_col} vs {outcome_col} (Spearman ρ={rho:.2f})", fontweight="bold")
    sns.despine(ax=ax)
    return ax
