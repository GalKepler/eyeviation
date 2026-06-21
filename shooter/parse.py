"""Stream .event.json (NDJSON) files into per-Type DataFrames and a per-shot table.

ponytail: plain functions + dicts, no event classes. The schema is fixed by the
data and small (a handful of Types) -- a class hierarchy would just rename
dict access without adding behavior.
"""
import json
from pathlib import Path

import pandas as pd

FRAME_TYPES = ("controller", "headset", "eye_data")
EVENT_TYPES = (
    "session_metadata", "scenario_start", "scenario_end", "location_metadata",
    "drill_state_change", "repetition_start", "repetition_end", "buzzer",
    "shooting_requirement_change", "target_state_change", "shot_start", "shot_resolved",
)


def parse_filename(path):
    """Timestamp_UserId_SessionId_ScenarioID.event.json -> dict."""
    stem = Path(path).name.removesuffix(".event.json")
    ts, user_id, session_id, scenario_id = stem.split("_")
    return {"timestamp": ts, "user_id": int(user_id), "session_id": int(session_id), "scenario_id": int(scenario_id)}


# Tie-break priority for records sharing the same millisecond Time. Observed
# in the data: a shooting_requirement_change(clear) can be logged at the same
# Time as the shot_start it was satisfied by -- if we let "clear" win the tie,
# build_shot_table would see the requirement as already gone before the shot
# it was meant to score. shot_start must process first so it snapshots live
# targets before same-tick clears apply.
_TIME_TIE_PRIORITY = {"shot_start": 0, "shooting_requirement_change": 1}


def load_raw(path):
    """Read NDJSON line by line -> list of {Time, Type, Data} dicts, sorted by
    (Time, tie-break priority).

    Events across Types are interleaved in the file and not causally ordered
    (e.g. a shot_resolved line can appear before its shot_start line), so we
    always re-sort by Time before doing anything else.
    """
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: (r["Time"], _TIME_TIE_PRIORITY.get(r["Type"], 1)))
    return records


def to_frames(records):
    """Split sorted records by Type into one flat DataFrame per Type.

    Returns {type_name: DataFrame with a "Time" column + flattened Data columns}.
    """
    by_type = {}
    for r in records:
        by_type.setdefault(r["Type"], []).append({"Time": r["Time"], **r["Data"]})
    return {t: pd.json_normalize(rows).sort_values("Time").reset_index(drop=True) for t, rows in by_type.items()}


def _target_match(object_id, requirement_target):
    """Does a shot_resolved.object_id satisfy an active requirement target path?

    Path = LINE_row/col/HitZone/ScoreZone. ScoreZone "*" is a wildcard; the
    other three segments must match exactly.
    """
    obj = object_id.split("/")
    req = requirement_target.split("/")
    if len(obj) != 4 or len(req) != 4:
        return False
    return obj[0] == req[0] and obj[1] == req[1] and obj[2] == req[2] and (req[3] == "*" or obj[3] == req[3])


def build_shot_table(records, meta):
    """Replay sorted events to join shot_start/shot_resolved with their
    repetition/drill/buzzer/requirement context. One row per resolved shot.

    Assumptions (data is ambiguous on these points -- see README):
    - repetition_start.id is always 0 in the sample data, so repetitions are
      identified by arrival order, not by id.
    - shot_start/shot_resolved ids reset to 0 at the start of each repetition
      and increment per bullet within it; resolved events follow their start
      event immediately once sorted by Time, so id-keyed pairing within the
      current repetition is reliable.
    - A requirement target is "live" between an "add" for it and the next
      "clear" that covers it (empty target = clear all). The snapshot of
      "live" used for matching is taken at shot_start time, not shot_resolved
      time: a clear event can share the same millisecond timestamp as the
      shot_start it satisfied (and sorts first, since a stable sort keeps
      file order for ties), so checking at resolve time would wrongly see an
      already-cleared requirement.
    """
    rows = []
    drill_id = None
    rep_idx = -1
    rep_start_time = None
    bullets_n = None
    buzzer_start_time = None
    live_targets = {}  # target_path -> bullets allowed
    open_shots = {}  # shot id -> (shot_start time, live_targets snapshot)

    for r in records:
        t, typ, data = r["Time"], r["Type"], r["Data"]

        if typ == "drill_state_change":
            drill_id = data["id"]

        elif typ == "repetition_start":
            # shot_start/shot_resolved ids reset to 0 per repetition (see
            # docstring), so open_shots must reset here too -- otherwise a
            # leftover id from the previous repetition could collide.
            rep_idx += 1
            rep_start_time = t
            bullets_n = data["bullets_n"]
            open_shots = {}

        elif typ == "buzzer":
            if data["type"] == "start":
                buzzer_start_time = t

        elif typ == "shooting_requirement_change":
            # "live_targets" is a running set of target-path -> bullets-required,
            # built up by "add" and torn down by "clear" (a clear with no
            # target wipes everything -- the engine's way of resetting between
            # phases of a repetition).
            if data["status"] == "add":
                live_targets[data["target"]] = data["bullets"]
            elif data["status"] == "clear":
                if data.get("target"):
                    live_targets.pop(data["target"], None)
                else:
                    live_targets = {}

        elif typ == "shot_start":
            # Snapshot (copy, not reference) live_targets as of this exact
            # moment, since a same-millisecond "clear" can still mutate the
            # dict afterward -- the snapshot is what shot_resolved checks
            # against later, regardless of what happens to live_targets next.
            open_shots[data["id"]] = (t, dict(live_targets))

        elif typ == "shot_resolved":
            shot_id = data["id"]
            object_id = data["object_id"]
            shot_start_time, targets_at_start = open_shots.pop(shot_id, (None, {}))
            # A hit only "counts" if the resolved object_id matches one of
            # the target paths that were live at shot_start time -- not
            # whatever was live by the time this resolve event arrived.
            valid_hit = any(_target_match(object_id, target) for target in targets_at_start)
            rows.append({
                **meta,
                "drill_id": drill_id,
                "rep_idx": rep_idx,
                "bullets_n": bullets_n,
                "rep_start_time": rep_start_time,
                "buzzer_start_time": buzzer_start_time,
                "shot_start_time": shot_start_time,
                "shot_resolved_time": t,
                "object_id": object_id,
                "object_type": data.get("object_type"),
                "score": data.get("score"),
                "valid_hit_recomputed": valid_hit,
                "live_targets": ",".join(targets_at_start) or None,
                "center_x": data["center_local_location"]["x"],
                "center_y": data["center_local_location"]["y"],
                "center_z": data["center_local_location"]["z"],
                "hit_x": data["hit_local_location"]["x"],
                "hit_y": data["hit_local_location"]["y"],
                "hit_z": data["hit_local_location"]["z"],
            })

    return pd.DataFrame(rows)


def parse_file(path):
    """Load one .event.json file -> (frames_by_type dict, shot_table DataFrame)."""
    meta = parse_filename(path)
    records = load_raw(path)
    frames = to_frames(records)
    shots = build_shot_table(records, meta)
    return frames, shots


def frames_between(df, t0, t1):
    """Slice a per-Type frame DataFrame to the time window [t0, t1)."""
    return df[(df["Time"] >= t0) & (df["Time"] < t1)]


def _self_check(path):
    frames, shots = parse_file(path)
    n_resolved = len(frames.get("shot_resolved", []))
    n_start = len(frames.get("shot_start", []))
    assert len(shots) == n_resolved, f"shot table rows ({len(shots)}) != shot_resolved count ({n_resolved})"
    assert n_start == n_resolved, f"shot_start count ({n_start}) != shot_resolved count ({n_resolved})"
    assert shots["shot_start_time"].notna().all(), "every shot should have a matched shot_start"
    assert shots["rep_idx"].ge(0).all(), "every shot should belong to a repetition"
    disagree = (shots["valid_hit_recomputed"] != (shots["object_type"] == "valid")).mean()
    print(f"OK: {len(shots)} shots parsed from {path.name}. "
          f"valid_hit_recomputed disagrees with object_type on {disagree:.0%} of shots.")


if __name__ == "__main__":
    import sys
    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path(".").glob("**/*.event.json"))
    _self_check(sample)
