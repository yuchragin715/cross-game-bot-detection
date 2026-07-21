import numpy as np
import pandas as pd

feature_cols = [
    "total_movement", "avg_speed", "n_events", "mean_dt",
    "idle_ratio", "avg_turn_angle", "speed_std", "speed_max", "dist_std",
]

cross_game_feature_cols = [
    "avg_speed", "mean_dt", "idle_ratio", "avg_turn_angle",
    "speed_std", "speed_max", "dist_std",
]

SCALE_FREE_COLS = ["speed_cv", "speed_peak", "dist_cv", "turn_angle", "idle_ratio"]
_EPS = 1e-9


def to_scale_free(feat_df):
    out = pd.DataFrame(index=feat_df.index)
    out["speed_cv"] = feat_df["speed_std"] / (feat_df["avg_speed"] + _EPS)
    out["speed_peak"] = feat_df["speed_max"] / (feat_df["avg_speed"] + _EPS)
    mean_dist = feat_df["total_movement"] / (feat_df["n_events"] + _EPS)
    out["dist_cv"] = feat_df["dist_std"] / (mean_dist + _EPS)
    out["turn_angle"] = feat_df["avg_turn_angle"]
    out["idle_ratio"] = feat_df["idle_ratio"]
    return out

def extract_features(mouse_df):
    df = mouse_df.sort_values("time")
    if len(df) < 2:
        return None

    distance = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    dt = df["time"].diff()
    valid = dt > 0
    speed = distance[valid] / dt[valid]
    
    raw_turn = np.arctan2(df["dy"], df["dx"]).diff()
    angles = ((raw_turn + np.pi) % (2 * np.pi) - np.pi).abs()[valid]

    if speed.empty:
        return None

    return {
        "total_movement": distance.sum(),
        "avg_speed": speed.mean(),
        "n_events": len(df),
        "mean_dt": dt[valid].mean(),
        "idle_ratio": (distance < 1).mean(),
        "avg_turn_angle": angles.mean(),
        "speed_std": speed.std(),
        "speed_max": speed.max(),
        "dist_std": distance.std(),
    }
