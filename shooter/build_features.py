"""Run the pipeline over the TargetAcquisition cohort -> features.csv.

ponytail: one script, no CLI framework -- there's exactly one job (build the
csv from the data folder) and no second use case to abstract for.
"""
from pathlib import Path

import pandas as pd

from shooter.features import add_all_features
from shooter.parse import parse_file

DATA_DIR = Path(__file__).parent.parent / "20260601072221_238_2566_11639.event"
OUT_PATH = Path(__file__).parent.parent / "features.csv"

# The WarmUp file (user 243, scenario 11601) runs a different drill than the
# other 7 TargetAcquisition sessions -- excluded from the cross-user cohort
# used for norms/validation. See README "Features not used / future work".
EXCLUDED_SCENARIOS = {11601}


def build(data_dir=DATA_DIR):
    all_shots = []
    for path in sorted(data_dir.glob("*.event.json")):
        frames, shots = parse_file(path)
        if shots.empty or shots["scenario_id"].iloc[0] in EXCLUDED_SCENARIOS:
            continue
        dominant_eye = frames["session_metadata"]["dominant_eye"].iloc[0]
        shots = add_all_features(shots, frames, dominant_eye)
        all_shots.append(shots)
    return pd.concat(all_shots, ignore_index=True)


if __name__ == "__main__":
    full = build()
    full.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(full)} shots from {full['user_id'].nunique()} users -> {OUT_PATH}")
